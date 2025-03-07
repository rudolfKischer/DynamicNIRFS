import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import networks
import data
import torchode
import json
import logging
import time
import datetime
import shutil
import os

# disable torch FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

device = None

def setup_logger(log_file_path):
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter('[%(levelname)s][%(asctime)s]| %(message)s')
  if not log_file_path.exists():
    log_file_path.touch()
  file_handler = logging.FileHandler(log_file_path)
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  console_handler = logging.StreamHandler()
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)

  # disable degault handler
  logger.propagate = False



  return logger

def prepare_data(experiment_config, epoch):
  data_config = experiment_config["data_config"]
  train_config = experiment_config["training_configuration"]
  tot_epochs = train_config["num_epochs"]
  num_trials = data_config["num_trials"]
  num_frames = data_config["num_frames"]
  sdf_samples_per_frame = data_config["samples_per_frame"]
  frames_per_batch = train_config["frames_per_batch"]
  segment_schedule = train_config["segment_schedule"]
  training_num_samples_per_frame = train_config["sdf_samples_per_frame"]
  
  # Get the current segment length for this point in the training schedule
  p_epoch = epoch / tot_epochs
  # sort by p value
  segment_schedule = sorted(segment_schedule, key=lambda x: x["p"], reverse=True)
  cur_segment_interval = segment_schedule[0]
  for i, segment_interval in enumerate(segment_schedule):
    if p_epoch > segment_interval["p"]:
      break
    cur_segment_interval = segment_interval
  seg_len = cur_segment_interval["seg_length"]
  samples_per_seg = cur_segment_interval["sampled_frames"]

  logger.info(f"Epoch {epoch} - Preparing data set...")
  
  # get the dataset
  sdfTrainData = data.NeuralSimTrainingDataSet(
    data_folder=experiment_config["training_data_folder"],
    frames_per_segment=seg_len,
    sampled_frames_per_segment=samples_per_seg,
    num_samples_per_frame=training_num_samples_per_frame
  )

  logger.info(f"Epoch {epoch} - Data set prepared.")
  logger.info(f"Epoch {epoch} - preparing dataloader...")

  # prepare the dataloader
  segs_per_batch = max(round(frames_per_batch / seg_len), 2)
  total_frames = num_trials * num_frames
  num_batches = total_frames // frames_per_batch
  total_segments_per_epoch = num_batches * segs_per_batch
  
  # log all numerical sizes of the settings for the training
  logger.info(f"Epoch {epoch} - Segment length: {seg_len}, Samples per segment: {samples_per_seg}, Segments per batch: {segs_per_batch}, Batches per epoch: {num_batches}, sdf samples per frame: {sdf_samples_per_frame}")

  batches_per_ode_update = train_config["sdf_batches_per_ode_update"]
  ode_updates_per_epoch = num_batches // batches_per_ode_update
  batches_per_ode_update = num_batches // ode_updates_per_epoch

  # log the number of ode updates per epoch
  logger.info(f"Epoch {epoch} - ODE updates per epoch: {ode_updates_per_epoch}, Batches per ODE update: {batches_per_ode_update}")

  data_sampler = torch.utils.data.RandomSampler(
    sdfTrainData, 
    replacement=False, 
    num_samples=total_segments_per_epoch
  )

  num_data_workers = train_config["data_loader_workers"]

  data_loader = torch.utils.data.DataLoader(
    sdfTrainData,
    batch_size=segs_per_batch,
    sampler=data_sampler,
    num_workers=num_data_workers,
    pin_memory=True,
    prefetch_factor=2
  )
  logger.info(f"Epoch {epoch} - Dataloader prepared.")

  logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

  train_info = {
    "num_batches": num_batches,
    "total_frames": total_frames,
    "total_segments_per_epoch": total_segments_per_epoch,
    "ode_updates_per_epoch": ode_updates_per_epoch,
    "batches_per_ode_update": batches_per_ode_update,
    "seg_len": seg_len,
    "samples_per_seg": samples_per_seg,
    "sdf_samples_per_frame": sdf_samples_per_frame,
    "frames_per_batch": frames_per_batch,
    "segs_per_batch": segs_per_batch
  }

  return sdfTrainData, data_loader, train_info

def load_experiment_config(experiment_config_path):
  with open(experiment_config_path, "r") as f:
    experiment_config = json.load(f)
  
  data_folder = experiment_config["training_data_folder"]
  data_config_file_name = experiment_config["training_data_config_file_name"]
  data_config = data_folder + '/' + data_config_file_name

  with open(data_config, "r") as f:
    data_config = json.load(f)
  
  experiment_config["data_config"] = data_config
  return experiment_config

logger = None 

def initialize_decoder(experiment_config):
  embedding_len = experiment_config["embedding_config"]["latent_dimensions"]
  auto_decoder_config = experiment_config["auto_decoder_config"]
  auto_decoder_config["embedding_length"] = embedding_len
  auto_decoder_config = networks.AutoDecoderConfig(**auto_decoder_config)
  auto_decoder = networks.AutoDecoder(auto_decoder_config)
  return auto_decoder

def initialize_ode_model(experiment_config):
  embedding_len = experiment_config["embedding_config"]["latent_dimensions"]
  ode_model_config = experiment_config["latent_ode_config"]
  ode_model_config["embedding_length"] = embedding_len
  ode_model_config = networks.LatentODEConfig(**ode_model_config)
  ode_model = networks.LatentODE(ode_model_config)
  return ode_model

def initialize_latent_buffer(experiment_config):
  num_trials = experiment_config["data_config"]["num_trials"]
  num_frames = experiment_config["data_config"]["num_frames"]
  embedding_config = experiment_config["embedding_config"]
  order = experiment_config["latent_ode_config"]["order"]
  embed_dims = embedding_config["latent_dimensions"]
  gauss_init = embedding_config["gauss_init"]
  guass_mean = embedding_config["gauss_mean"]
  guass_std = embedding_config["gauss_std"]
  # latent shape: (num_seq, num_frames_per_seq, latent_dim)
  buffer_shape = (num_trials, num_frames, embed_dims * order)
  if gauss_init and False:
    # TODO: fix guassian latent initialization, right now causes nan values

    latent_buffer = torch.normal(mean=guass_mean, std=guass_std, size=buffer_shape)
    latent_buffer = torch.clamp(latent_buffer, min=-10.0, max=10.0)

    # set from embedding length to 2 * embedding length to 0
    start_of_p = embed_dims
    end_of_p = 2 * embed_dims
    latent_buffer[:, :, start_of_p:end_of_p] = 0.0
  else:
    # latent_buffer = torch.zeros(buffer_shape)
    # TODO: Initializing to zero causes nans?
    latent_buffer = torch.randn(buffer_shape)
  latent_buffer.requires_grad = True
  return latent_buffer

integration_methods = {
  "tsit5": torchode.Tsit5,
  "dopri5": torchode.Dopri5
}

class LoggingIntegralController(torchode.IntegralController):
  def __init__(self, *args, **kwargs):
    super(LoggingIntegralController, self).__init__(*args, **kwargs)
  
  def adapt_step_size(self, *args, **kwargs):
    # log the args
    # logger.info(f"Adapt step size args: {args}")


    return super(LoggingIntegralController, self).adapt_step_size(*args, **kwargs)

def initialize_ode_solver(
    ode: nn.Module,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    method: str = "rk4",
):
  ode_term = torchode.ODETerm(ode)
  stepper = integration_methods[method](term=ode_term)
  # step_controller = torchode.IntegralController(atol=atol, rtol=rtol, term=ode_term)
  step_controller = LoggingIntegralController(atol=atol, rtol=rtol, term=ode_term)
  ode_solver = torchode.AutoDiffAdjoint(stepper, step_controller).to(device)
  # still needs to be sent to device
  return ode_solver


def prepare_ode_problem(z_buffer, seg_sequences, seg_frame_times, seg_len, fps):
    ode_start_frames = seg_frame_times[:, 0]
    ode_end_frames = seg_frame_times[:, -1]
    # insteaf of seg_len, it should be sampled frames per segment

    t = torch.linspace(0, 1, seg_len, device=device).unsqueeze(0)
    ode_time_evals = ode_start_frames.unsqueeze(1) + t * (ode_end_frames.unsqueeze(1) - ode_start_frames.unsqueeze(1))
    ode_time_evals = ode_time_evals
    # logger.info(f"ODE time evaluations: {ode_time_evals}")
    # run ode solver
    y0 = z_buffer[seg_sequences, seg_frame_times[:, 0]]
    # logger.info(f"Epoch {epoch} - Batch {iter} - ODE initial state: {y0}")
    ode_problem = torchode.InitialValueProblem(y0=y0, t_eval=ode_time_evals / fps)
    # we need to unroll the ode_time_evals, to get the gt latent vectors
    # and then we roll them back up so they are the same shape as the ode solution
    # and we have to expand the seg_sequences to match the shape of the ode_time_evals , so we expand by seg_len
    num_segments = seg_sequences.shape[0]
    latent_dim = z_buffer.shape[-1]
    # we want to get our latent gt in this shape
    # ode_gt = torch.zeros(num_segments, seg_len, latent_dim, device=device)
    expanded_seg_sequences = seg_sequences.unsqueeze(1).expand(num_segments, seg_len).reshape(-1)
    ode_gt = z_buffer[expanded_seg_sequences, ode_time_evals.view(-1).int()].view(num_segments, seg_len, latent_dim)



    return ode_problem, ode_gt

def ode_loss_fn(z_ode_solution, z_gt):
  # take the sequared difference
  # then take the mean across the frames
  # logger.info(f"ODE solution shape: {z_ode_solution.shape}, GT shape: {z_gt.shape}")

  # normalize both latents, to prevent the model
  # from learning to reduce loss by reducing the latent magnitude
  # z_ode_solution = z_ode_solution / torch.norm(z_ode_solution, dim=-1, keepdim=True).to(device, non_blocking=True)
  z_gt = z_gt / torch.norm(z_gt, dim=-1, keepdim=True).to(device, non_blocking=True)

  return torch.mean((z_ode_solution - z_gt) ** 2).to(device, non_blocking=True)

def prepare_sdf_batch():
  pass

def sdf_loss_fn(sdf_pred, sdf_gt):
  # mean squared error
  return torch.mean((sdf_pred - sdf_gt) ** 2).to(device, non_blocking=True)


def plot_loss(loss_histories, fig_path):
  fig, ax = plt.subplots(1, 1, figsize=(10, 5))

  # smooth the loss history
  window_size = 10
  smoothed_loss_histories = {}

  for key, loss_history in loss_histories.items():
    new_loss_history = []
    for i in range(len(loss_history)):
      start = max(0, i - window_size)
      end = min(len(loss_history), i + window_size)
      new_loss_history.append(np.mean(loss_history[start:end]))
    smoothed_loss_histories[key] = new_loss_history
  
  loss_histories = smoothed_loss_histories

  for key, loss_history in loss_histories.items():
    ax.plot(loss_history, label=key)
  ax.legend()
  ax.set_xlabel("Batch")
  ax.set_ylabel("Loss")
  ax.set_title("Loss over training")
  fig.savefig(fig_path)
  plt.close(fig)

def plot_latent_variances(sequence_num, latent_buffer, fig_path):
  latent_variances = torch.var(latent_buffer, dim=1)
  # we want a bar plot of the latent variance for that sequence
  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  ax.bar(range(latent_variances.shape[1]), latent_variances[sequence_num])
  ax.set_xlabel("Latent dimension")
  ax.set_ylabel("Variance")
  ax.set_title(f"Latent variance for sequence {sequence_num}")
  fig.savefig(fig_path)
  plt.close(fig)

def plot_latent_trajectories(sequence_num, latent_buffer, fig_path):
  # we want to plot each latent vector over time as a line plot
  # each dimension should have its own line
  latent_trajectory = latent_buffer[sequence_num]
  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  for i in range(latent_trajectory.shape[1]):
    ax.plot(latent_trajectory[:, i], label=f"Latent dimension {i}")
  ax.legend()
  ax.set_xlabel("Frame")
  ax.set_ylabel("Latent value")
  ax.set_title(f"Latent trajectory for sequence {sequence_num}")
  fig.savefig(fig_path)
  plt.close(fig)

def plot_latent_trajectoryPCA(sequence_num, latent_buffer, fig_path):
  # plot only the 2 first principal components of the latent vectors
  # one dimension on on x axis, the other on y axis
  # make sure to label the axes dimensions
  sequence_latents = latent_buffer[sequence_num]
  latent_variances = torch.var(sequence_latents, dim=0)
  # get the 2 most variant dimensions
  _, top_dims = torch.topk(latent_variances, 2)
  # out plot should be a xy point at each from, connected by a line in order
  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  ax.plot(sequence_latents[:, top_dims[0]], sequence_latents[:, top_dims[1]])
  ax.set_xlabel(f"Latent dimension {top_dims[0]}")
  ax.set_ylabel(f"Latent dimension {top_dims[1]}")
  ax.set_title(f"Latent trajectory PCA for sequence {sequence_num}")
  fig.savefig(fig_path)
  plt.close(fig)




def train_neural_sim(experiment_config):
  global device

  # Experiment file structure
  # <experiment_name>/
  #   experiment_config.json
  #   models/
  #    ode_model.pt
  #    sdf_model.pt
  #    latent_buffer.pt
  #   logs/
  #   figures/

  #==================================INITIALIZATION===================================================================
  # log experiment config
  # pretty print the experiment config
  logger.info(f"Experiment config: {json.dumps(experiment_config, indent=4)}")
  
  # initialize models
  auto_decoder = initialize_decoder(experiment_config)
  ode_model = initialize_ode_model(experiment_config)
  z_buffer = initialize_latent_buffer(experiment_config) # z buffer as in latent state, not depth buffer

  # initialize ode solver
  ode_solver_config = experiment_config["ode_solver_config"]
  ode_solver = initialize_ode_solver(ode_model, **ode_solver_config)
  logger.info(f"ODE solver initialized with config: {ode_solver_config}")

  training_configs = experiment_config["training_configuration"]
  num_epochs = training_configs["num_epochs"]
  lr = training_configs["learning_rate"]

  
  # initialize learning rate scheduler

  # initialize optimizer
  ode_optimizer = optim.Adam(ode_model.parameters(), lr=lr, weight_decay=1e-5)
  grouped_params = [
    {
      "params": auto_decoder.parameters(),
      "lr": lr
    },
    {
      "params": z_buffer,
      "lr": lr
    }
  ]


  # adam optimizer with learning rate scheduler
  group_optimizer = optim.Adam(grouped_params, lr=lr, weight_decay=1e-5)
  group_scheduler= optim.lr_scheduler.ReduceLROnPlateau(group_optimizer, mode='min', factor=0.85, patience=200)

  cur_device = training_configs["device"]
  if cur_device == "cuda":
    cur_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Falling back to CPU since CUDA is not available.")
  
  device = cur_device
  logger.info(f"Device: {device}")

  ode_model.train()
  auto_decoder.train()
  # z_buffer.train()

  z_buffer = z_buffer.to(device)
  z_buffer.requires_grad = True
  order = experiment_config["latent_ode_config"]["order"]

  # send models to device
  auto_decoder.to(device)
  ode_model.to(device)

  batch_loss_history = {
    "ode_loss": [],
    "sdf_loss": [],
    "joint_loss": [],
    "z_weight_loss": []
  }
  epoch_loss_history = {
    "ode_loss": [],
    "sdf_loss": [],
    "joint_loss": []
  }

  # log the model params
  for name, param in ode_model.named_parameters():
    logger.info(f"ODE model parameter {name} - {param}")
  for name, param in auto_decoder.named_parameters():
    logger.info(f"Auto decoder parameter {name} - {param}")
  
  ode_final_lambda = training_configs["ode_final_lambda"]
  ode_initial_lambda = training_configs["ode_init_lambda"]
  sdf_lambda = training_configs["sdf_lambda"]

  try:
    # enter epoch loop
    for epoch in range(num_epochs):

      # prepare dataset for epoch
      # initialize dataloader
      sdfTrainData, data_loader, train_info = prepare_data(experiment_config, epoch)
      # log the training info
      # pretty print the training info
      logger.info(f"Training info: {json.dumps(train_info, indent=4)}")
      # update schedules for learning rate + lambda + segment length

      batches_per_ode_update = train_info["batches_per_ode_update"]
      ode_update_per_epoch = train_info["ode_updates_per_epoch"]
      seg_len = train_info["seg_len"]
      fps = experiment_config["data_config"]["fps"]
      is_gauss_latents = experiment_config["embedding_config"]["gauss_init"]
      latent_gauss_mean = experiment_config["embedding_config"]["gauss_mean"]
      latent_gauss_std = experiment_config["embedding_config"]["gauss_std"]
      embedding_len = experiment_config["embedding_config"]["latent_dimensions"]
      sampled_frames_per_segment = train_info["samples_per_seg"]

      ode_optimizer.zero_grad(set_to_none=True)
      group_optimizer.zero_grad(set_to_none=True)

      #==================================TRAINING LOOP===================================================================

      # enter iteration loop (batch loop)
      # unload data 
      for iter, (seg_sequences, seg_frame_times, seg_samples) in enumerate(data_loader):
        p_epoch = (epoch / num_epochs) * 100
        p_batch = (iter / train_info["num_batches"]) * 100
        iter_header = f"Epoch {epoch}/{num_epochs} Batch {iter}/{train_info['num_batches']} ({p_batch:.2f}%) -"
        logger.info(f"{iter_header} Data loaded to device: {device}")
        # send the data to the device
        seg_sequences = seg_sequences.to(device, non_blocking=True)
        seg_frame_times = seg_frame_times.to(device, non_blocking=True)
        seg_samples = seg_samples.to(device, non_blocking=True)

        # normalize just the Z buffer
        # z_buffer = z_buffer / torch.norm(z_buffer, dim=-1, keepdim=True).to(device, non_blocking=True)
        # z_buffer.retain_grad()


    

        #==================================ODE SOLVER===================================================================
        # if batch_n % ode_update == 0:
        # set gradients to zero
        if iter % batches_per_ode_update == 0 and iter > 0:
          ode_optimizer.zero_grad(set_to_none=True)
          logger.info(f"{iter_header} ODE optimizer gradients zeroed.")
        group_optimizer.zero_grad(set_to_none=True)

        # prepare data for ode solver

        # logg latent buffer
        # logger.info(f"{iter_header} Latent buffer: {z_buffer}")

              # check if the ode_model weights contain any nan values
        for name, param in ode_model.named_parameters():
          if torch.isnan(param).any():
            logger.warning(f"{iter_header} ODE model parameter {name} has NaN values.")
            # print the parameter
            logger.warning(f"{iter_header} ODE model parameter {name}: {param}")
            # set the parameter to 0
            param.data.fill_(0.0)
        
        if torch.isnan(z_buffer).any():
          logger.warning(f"{iter_header} Z buffer has NaN values.")
          # print the parameter
          logger.warning(f"{iter_header} Z buffer: {z_buffer}")
          # set the parameter to 0
          z_gt.data.fill_(0.0)
          raise ValueError("Z buffer has NaN values.")
  

        logger.info(f"{iter_header} Preparing ODE problem...")
        ode_problem, z_gt = prepare_ode_problem(
          z_buffer, 
          seg_sequences, 
          seg_frame_times, 
          sampled_frames_per_segment,
          fps
        )
        # check if any of the ode_params contain any nan values
        for name, param in ode_model.named_parameters():
          if torch.isnan(param).any():
            logger.warning(f"{iter_header} ODE model parameter {name} has NaN values.")
            # print the parameter
            logger.warning(f"{iter_header} ODE model parameter {name}: {param}")
            # set the parameter to 0
            param.data.fill_(0.0)
        # check if the z_gt has any nan values
        if torch.isnan(z_gt).any():
          logger.warning(f"{iter_header} ODE GT has NaN values.")
          # print the parameter
          logger.warning(f"{iter_header} ODE GT: {z_gt}")
          # set the parameter to 0
          z_gt.data.fill_(0.0)

        logger.info(f"{iter_header} ODE problem prepared.")
        logger.info(f"{iter_header} Solving ODE...")
        ode_solution = ode_solver.solve(ode_problem)
        logger.info(f"{iter_header} ODE solved.")

        z_sol = ode_solution.ys
        # if z_sol has any nan values log them as a warning
        ode_lambda = ode_initial_lambda + ode_final_lambda * (p_epoch / 100)
        ode_batch_loss = ode_loss_fn(z_sol, z_gt) * ode_lambda #/ sampled_frames_per_segment
        if torch.isnan(z_sol).any():
          logger.warning(f"{iter_header} ODE SOLUTION HAS NaN VALUES.")
          # print the solution
          # set ode_bathch loss to 0
          ode_batch_loss = torch.tensor(0.0, device=device)
        
          logger.warning(f"{iter_header} ODE solution: {z_sol}")

          # raise an exception
          raise ValueError("ODE solution has NaN values.")








        # calculate ode loss

        #==================================SDF DECODER===================================================================

        # get the latent vectors for the ode samples
        expanded_seg_sequences = seg_sequences.unsqueeze(1).expand(seg_sequences.shape[0], sampled_frames_per_segment).reshape(-1)
        batch_latents = z_buffer[expanded_seg_sequences, seg_frame_times.view(-1)]
        batch_latents_view = batch_latents.view(seg_sequences.shape[0], sampled_frames_per_segment, -1)
        # create the gaussian latent vectors
        if is_gauss_latents:
          batch_gauss_latents = torch.distributions.Normal(loc=batch_latents, scale=latent_gauss_std).rsample()
          batch_latents_view = batch_gauss_latents.view(seg_sequences.shape[0], sampled_frames_per_segment, -1)
        
        # only pass the state into the decoder, which is the 0 - embedding_len, not 0 - 2 * embedding_len
        # expand to the number 
        batch_q = batch_latents_view[:, :, :embedding_len]
        batch_xyz = seg_samples[:, :, :, :3]
        batch_sdf = seg_samples[:, :, :, 3]
        sdf_samples_per_frame = batch_xyz.shape[2]
        logger.info(f"{iter_header} Batch SDF before unsqueeze shape: {batch_sdf.shape}")

        batch_sdf_gt = batch_sdf.unsqueeze(-1)
        # log the head of the batch sdf_gt, and the batch_q
        # print the shape
        logger.info(f"{iter_header} Batch SDF GT shape: {batch_sdf_gt.shape}")
        logger.info(f"{iter_header} Batch Q shape: {batch_q.shape}")
        # logger.info(f"{iter_header} Batch SDF GT: {batch_sdf_gt[0, 0, 0, :5]}")
        # logger.info(f"{iter_header} Batch Q: {batch_q[0, 0]}")
        batch_q = batch_q.unsqueeze(2)
        batch_q_repeat = batch_q.expand(-1, -1, sdf_samples_per_frame, -1)
        logger.info(f"{iter_header} Batch Q repeat shape: {batch_q_repeat.shape}")
        # logger.info(f"{iter_header} Batch Q repeat: {batch_q_repeat[0, 0, :5]}")
        sdf_pred = auto_decoder(batch_xyz, batch_q_repeat)

        logger.info(f"{iter_header} SDF prediction shape: {sdf_pred.shape}")

        # ==================================BACKPROPAGATION Error===================================================================

        # log the head of
        # log the sdf_pred and the batch_sdf_gt
        # [sequence, frame, sample, sdf]
        logger.info(f"{iter_header} SDF prediction: {sdf_pred[0, 0, :5]}")
        logger.info(f"{iter_header} SDF GT: {batch_sdf_gt[0, 0, :5]}")
        # print the head of the corresponing xyz values
        # print without scientifi notation

        torch.set_printoptions(sci_mode=False)

        logger.info(f"{iter_header} XYZ: {batch_xyz[0, 0, :5]}")
        # print the tail

        # We are having problems getting negative sdf samples from the auto encoder
        # print what percentage of the predictions are negative values, and what percentage of the gt are negative values
        logger.info(f"{iter_header} SDF prediction negative percentage: {torch.mean((sdf_pred < 0).float())}")
        logger.info(f"{iter_header} SDF GT negative percentage: {torch.mean((batch_sdf_gt < 0).float())}")

        

        # calculate the sdf loss
        sdf_batch_loss = sdf_loss_fn(sdf_pred, batch_sdf_gt) * sdf_lambda

        # subtract the z buffer magnitude from the sdf loss
        # z_weight_loss = torch.mean(torch.abs(batch_latents)).to(device, non_blocking=True)
        z_weight_loss = torch.tensor(0.0, device=device)

        

        # calculate the joint loss
        joint_loss = sdf_batch_loss + ode_batch_loss # + z_weight_loss

        torch.nn.utils.clip_grad_norm_(ode_model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(auto_decoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(z_buffer, 1.0)

        # backpropogate the joint loss
        joint_loss.backward()

        logger.info(f"{iter_header} ODE loss: {ode_batch_loss.item():2e}, SDF loss: {sdf_batch_loss.item():2e}, Joint loss: {joint_loss.item():2e}")
        # check the gradients for nans
        for name, param in ode_model.named_parameters():
          if torch.isnan(param.grad).any():
            logger.warning(f"{iter_header} ODE model parameter {name} has NaN gradients.")
            # print the parameter
            logger.warning(f"{iter_header} ODE model parameter {name}: {param.grad}")
            # set the parameter to 0
            param.grad.data.fill_(0.0)
        
        for name, param in auto_decoder.named_parameters():
          if torch.isnan(param.grad).any():
            logger.warning(f"{iter_header} Auto decoder parameter {name} has NaN gradients.")
            # print the parameter
            logger.warning(f"{iter_header} Auto decoder parameter {name}: {param.grad}")
            # set the parameter to 0
            param.grad.data.fill_(0.0)
        
        if torch.isnan(z_buffer.grad).any():
          logger.warning(f"{iter_header} Z buffer has NaN gradients.")
          # print the parameter
          logger.warning(f"{iter_header} Z buffer: {z_buffer.grad}")
          # set the parameter to 0
          z_buffer.grad.data.fill_(0.0)
          raise ValueError("Z buffer has NaN gradients.")

        # check if any of the gradients from backpropagation contain any nan values



        # clip the gradients
        torch.nn.utils.clip_grad_norm_(ode_model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(auto_decoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(z_buffer, 1.0)

        # step the latent buffer optimizer and the ode solver
        group_optimizer.step()
        logger.info(f"{iter_header} Group optimizer stepped.")

        # if batch_n % ode_update == 0:
        # step the ode optimizer
        if iter % batches_per_ode_update == 0 and iter > 0:
          ode_optimizer.step()

        # prinnt the   params
        # for name, param in ode_model.named_parameters():
          # logger.info(f"{iter_header} ODE model parameter {name} - grad: {param}")

          logger.info(f"{iter_header} ODE optimizer stepped.")

        # log the batch losses

        # detach and clone th batch losses
        ode_batch_loss = ode_batch_loss.detach().clone()
        sdf_batch_loss = sdf_batch_loss.detach().clone()
        joint_loss = joint_loss.detach().clone()
        z_weight_loss = z_weight_loss.detach().clone()

        # append the batch losses to the history
        batch_loss_history["ode_loss"].append(ode_batch_loss.item())
        batch_loss_history["sdf_loss"].append(sdf_batch_loss.item())
        batch_loss_history["joint_loss"].append(joint_loss.item())
        batch_loss_history["z_weight_loss"].append(z_weight_loss.item())

        group_scheduler.step(joint_loss)
        logger.info(f"{iter_header} Current LR: {group_optimizer.param_groups[0]['lr']}")

        # plot loss and save to fig
        if iter % 30 == 0:
          figures_folder = experiment_config["experiment_folder_path"] + "/figures"
          if not Path(figures_folder).exists():
            Path(figures_folder).mkdir(parents=True, exist_ok=True)
          figure_name = f"loss_plot.png"
          figure_path = figures_folder + "/" + figure_name
          plot_loss(batch_loss_history, figure_path)

          # evaluate the model on the first sequence
          # we take all the samples from the first frame
          # get the prediction from the auto decoder
          # then we plot the samples, in th unit x z 0 to 1 space
          # we set the sdf value for the color of the point
          # then we save to a folder called sdf plot with the batch and epoch number

          sdf_plot_folder = experiment_config["experiment_folder_path"] + "/figures" + "/sdf_plot"
          if not Path(sdf_plot_folder).exists():
            Path(sdf_plot_folder).mkdir(parents=True, exist_ok=True)
          sdf_plot_name = f"sdf_plot_epoch_{epoch}_batch_{iter}.png"
          sdf_plot_path = sdf_plot_folder + "/" + sdf_plot_name

          # get the first frame samples from the first sequence
          # to do this we load up the samples file from the first sequence
          trial_folder = experiment_config["training_data_folder"] + "/trial_0"
          samples_file = trial_folder + "/samples.pt"
          s0_samples = torch.load(samples_file)
          frame_no = 0
          f0_s0_samples = s0_samples[frame_no]
          f0_s0_xyz = f0_s0_samples[:, :3]
          f0_s0_sdf = f0_s0_samples[:, 3]

          # get the latent vector for the first frame
          f0_s0_latent = z_buffer[0, frame_no]
          f0_s0_latent = f0_s0_latent.unsqueeze(0).unsqueeze(0)
          f0_s0_latent = f0_s0_latent.expand(1, f0_s0_xyz.shape[0], -1)
          # get just the first half of the latent vector
          f0_s0_latent = f0_s0_latent[:, :, :embedding_len]
          # print(f'f0_s0_xyz shape: {f0_s0_xyz.shape}')
          # expand unsqueeze xyz
          f0_s0_xyz = f0_s0_xyz.unsqueeze(0)
          # get the prediction from the auto decoder
          # print f0_s0_xyz.shape 
          # print(f'f0_s0_xyz shape: {f0_s0_xyz.shape}')

          f0_s0_pred = auto_decoder(f0_s0_xyz, f0_s0_latent)

          f0_s0_pred = f0_s0_pred.detach().clone().squeeze().cpu().numpy()

          # plot the samples
          fig, ax = plt.subplots(1, 1, figsize=(10, 10))
          # use color red blue, and set max sdf color to 0.1 and min to -0.1
          ax.scatter(f0_s0_xyz[:, :, 0], f0_s0_xyz[:, :, 2], c=f0_s0_pred, cmap='RdBu', s=100)
          # make  the points bigger

          ax.set_xlabel("X")
          ax.set_ylabel("Z")
          ax.set_title("SDF samples for first frame of first sequence")
          fig.savefig(sdf_plot_path)
          plt.close(fig)




      
        

  except KeyboardInterrupt:
    logger.info("Training interrupted by user.")

  # end iteration loop

  # log learning rate, lambda, segment length

  # end epoch loop

  #==================================END OF TRAINING===================================================================


  # save the model
  # create models folder if it does not exist
  models_folder = experiment_config["experiment_folder_path"] + "/models"
  if not Path(models_folder).exists():
    Path(models_folder).mkdir(parents=True, exist_ok=True)



  
  # save the models
  ode_model_file_name = "ode_model.pt"
  auto_decoder_file_name = "auto_decoder.pt"
  z_buffer_file_name = "z_buffer.pt"
  ode_model_path = models_folder + "/" + ode_model_file_name
  auto_decoder_path = models_folder + "/" + auto_decoder_file_name
  z_buffer_path = models_folder + "/" + z_buffer_file_name
  ode_model.save(ode_model_path)
  auto_decoder.save(auto_decoder_path)
  torch.save(z_buffer, z_buffer_path)

  # add the model file names to the experiment config
  experiment_config["ode_model_file_name"] = ode_model_file_name
  experiment_config["auto_decoder_file_name"] = auto_decoder_file_name
  experiment_config["z_buffer_file_name"] = z_buffer_file_name

  # add loss history to experiment config
  experiment_config["loss_history"] = batch_loss_history


  # save all configuration info, and generated training info (losses, lr, lambda, segment length, etc)
  # convert experiment config to json
  experiment_config_path = experiment_config["experiment_folder_path"] + "/experiment_config.json"
  with open(experiment_config_path, "w") as f:
    json.dump(experiment_config, f)
  
  # evaluate all sdf sequences
  evaluate_all_sdf_sequences(auto_decoder, z_buffer, experiment_config)
  
  
  
  #evaluate the experiment
  evaluate_experiment(experiment_config["experiment_folder_path"])
  
  







def plot_all_latent_variances(latent_buffer, latent_variance_fig_folder):
  # plot the variance of the latent vectors for each sequence
  if not Path(latent_variance_fig_folder).exists():
    Path(latent_variance_fig_folder).mkdir(parents=True, exist_ok=True)
  for seq_num in range(latent_buffer.shape[0]):
    fig_name = f"latent_variance_seq_{seq_num}.png"
    fig_path = latent_variance_fig_folder + "/" + fig_name
    plot_latent_variances(seq_num, latent_buffer, fig_path)

def plot_all_latent_trajectories(latent_buffer, latent_trajectory_fig_folder):

  #plot latent trajectories over time for each sequence
  if not Path(latent_trajectory_fig_folder).exists():
    Path(latent_trajectory_fig_folder).mkdir(parents=True, exist_ok=True)
  for seq_num in range(latent_buffer.shape[0]):
    fig_name = f"latent_trajectory_seq_{seq_num}.png"
    fig_path = latent_trajectory_fig_folder + "/" + fig_name
    plot_latent_trajectories(seq_num, latent_buffer, fig_path)

def plot_all_latent_trajectoryPCA(latent_buffer, latent_pca_fig_folder):
  # plot the pca of the latent vectors
  if not Path(latent_pca_fig_folder).exists():
    Path(latent_pca_fig_folder).mkdir(parents=True, exist_ok=True)
  for seq_num in range(latent_buffer.shape[0]):
    fig_name = f"latent_pca_seq_{seq_num}.png"
    fig_path = latent_pca_fig_folder + "/" + fig_name
    plot_latent_trajectoryPCA(seq_num, latent_buffer, fig_path)

def evaluate_all_sdf_sequences(
    auto_decoder,
    z_buffer,
    experiment_config
  ):
  data_folder = experiment_config["training_data_folder"]
  # create a folder to save the sdf samples
  sdf_sample_folder = experiment_config["experiment_folder_path"] + "/sdf_samples"
  if not Path(sdf_sample_folder).exists():
    Path(sdf_sample_folder).mkdir(parents=True, exist_ok=True)
  # evaluate the sdf sequences
  num_trials = experiment_config["data_config"]["num_trials"]
  for seq_num in range(num_trials):
    evaluate_sdf_sequence(
      seq_num,
      auto_decoder,
      z_buffer,
      data_folder,
      sdf_sample_folder,
      experiment_config
    )

def evaluate_sdf_sequence(
    seq_num, 
    auto_decoder, 
    z_buffer, 
    data_folder, 
    sdf_sample_folder, 
    experiment_config
    ):
    # get the original samples for that sequence
    trial_folder = data_folder + f"/trial_{seq_num}"
    trial_sdf_samples = torch.load(trial_folder + "/samples.pt")

    xyz = trial_sdf_samples[:, :, :3]  # (num_frames, num_samples, 3)
    sdf_gt = trial_sdf_samples[:, :, 3]  # (num_frames, num_samples)

    num_frames, num_samples, _ = xyz.shape
    embedding_len = experiment_config["embedding_config"]["latent_dimensions"]

    # Extract latent vectors for the sequence
    z_seq = z_buffer[seq_num, :num_frames, :embedding_len]  # (num_frames, latent_dim)

    # Expand z_seq to match the number of samples
    z_seq_expanded = z_seq.unsqueeze(1).expand(-1, num_samples, -1)  # (num_frames, num_samples, latent_dim)

    # Flatten the tensors for batch processing
    xyz_flat = xyz.reshape(-1, 3).to(device)  # (num_frames * num_samples, 3)
    z_flat = z_seq_expanded.reshape(-1, embedding_len).to(device)  # (num_frames * num_samples, latent_dim)

    # Ensure the auto_decoder is in evaluation mode
    auto_decoder.eval()
    with torch.no_grad():
        # Predict SDF values
        print(f'xyz_flat shape: {xyz_flat.shape}')
        # unsqueese xyz_flat on first dimension, and z_flat on first dimension
        xyz_flat = xyz_flat.unsqueeze(0)
        z_flat = z_flat.unsqueeze(0)
        sdf_pred = auto_decoder(xyz_flat, z_flat)


    # sdf_pred = auto_decoder(xyz, q_seq)
    sdf_pred = sdf_pred.reshape(num_frames, num_samples)
    

    # calculate the MSE and print it out
    sdf_loss = sdf_loss_fn(sdf_pred, sdf_gt)
    print(f'SDF loss for sequence {seq_num}: {sdf_loss.item()}')

    # save the sdf samples
    print(f'sdf_pred shape: {sdf_pred.shape}')
    print(f'sdf shape: {sdf_gt.shape}')
    print(f'xyz shape: {xyz.shape}')
    sdf_pred = sdf_pred.unsqueeze(-1)
    sdf_samples = torch.cat((xyz, sdf_pred), dim=-1)
    # detach and clone the sdf samples
    sdf_samples = sdf_samples.detach().clone()
    # save the sdf samples
    pred_sdf_samples_folder = sdf_sample_folder + f"/trial_{seq_num}"
    if not Path(pred_sdf_samples_folder).exists():
      Path(pred_sdf_samples_folder).mkdir(parents=True, exist_ok=True)
    sdf_samples_path = pred_sdf_samples_folder + "/samples.pt"
    torch.save(sdf_samples, sdf_samples_path)

    # create symbolic links to the original mesh sequence and dynamics files
    mesh_sequence_path = trial_folder + "/mesh_sequence.pkl"
    new_mesh_sequence_path = sdf_sample_folder + f"/trial_{seq_num}" + "/mesh_sequence.pkl"

    # if the new file alread exists, delete it 
    if Path(new_mesh_sequence_path).exists():
      Path(new_mesh_sequence_path).unlink()
    os.symlink(mesh_sequence_path, new_mesh_sequence_path)

    dynamics_path = trial_folder + "/dynamics.pkl"
    new_dynamics_path = sdf_sample_folder + f"/trial_{seq_num}" + "/dynamics.pkl"
    if Path(new_dynamics_path).exists():
      Path(new_dynamics_path).unlink()
    os.symlink(dynamics_path, new_dynamics_path)


def evaluate_experiment(experiment_folder_path):

  #load configuration
  experiment_config_path = experiment_folder_path + "/experiment_config.json"
  with open(experiment_config_path, "r") as f:
    experiment_config = json.load(f)
  # load models and latent buffer
  model_folder = experiment_folder_path + "/models"
  ode_model_name = experiment_config["ode_model_file_name"]
  auto_decoder_name = experiment_config["auto_decoder_file_name"]
  z_buffer_name = experiment_config["z_buffer_file_name"]

  ode_model_path = model_folder + "/" + ode_model_name
  auto_decoder_path = model_folder + "/" + auto_decoder_name
  z_buffer_path = model_folder + "/" + z_buffer_name

  ode_model = networks.LatentODE.load(ode_model_path)
  auto_decoder = networks.AutoDecoder.load(auto_decoder_path)
  z_buffer = torch.load(z_buffer_path, weights_only=True)

  ode_solver_config = experiment_config["ode_solver_config"]
  ode_solver = initialize_ode_solver(ode_model, **ode_solver_config)




  # plot the pca of the latent vectors
  # plot the latent vector trajectory for each sequence for sdf evaluation
  # plot the latent vector trajectory for each sequence for ode + sdf evaluation
  # plot the latent vecotor variation 
  # plot the latent vector variation over time while training
    # evaluate the experiment
  z_buffer = z_buffer.detach().clone()
  
  # plot the variance of the latent vectors for each sequence
  latent_variance_fig_folder = experiment_config["experiment_folder_path"] + "/figures" + "/latent_variance"
  plot_all_latent_variances(z_buffer, latent_variance_fig_folder)
  
  #plot latent trajectories over time for each sequence
  latent_trajectory_fig_folder = experiment_config["experiment_folder_path"] + "/figures" + "/latent_trajectory"
  plot_all_latent_trajectories(z_buffer, latent_trajectory_fig_folder)
  
  # plot the pca of the latent vectors
  latent_pca_fig_folder = experiment_config["experiment_folder_path"] + "/figures" + "/latent_pca"
  plot_all_latent_trajectoryPCA(z_buffer, latent_pca_fig_folder)
  
  # setup ode solver, and, evaluate each sequence, (step from the first frame to the last frame)
  # we will use this to create a new pred_latent_buffer

  fps = experiment_config["data_config"]["fps"]
  # set segment length to total frames in a sequence
  seg_len = experiment_config["data_config"]["num_frames"]
  num_sequences = experiment_config["data_config"]["num_trials"]
  
  # we now want to create a seg_seq , thats just the sequence number,
  # and the seg_frame_times, which is all frame from first to last
  seg_sequences = torch.arange(num_sequences, device=device)
  seg_frame_times = torch.arange(seg_len, device=device)
  seg_frame_times = seg_frame_times.unsqueeze(0).expand(num_sequences, -1)


  # print the shapes
  print(f'z_buffer shape: {z_buffer.shape}')
  print(f'seg_sequences shape: {seg_sequences.shape}')
  print(f'seg_frame_times shape: {seg_frame_times.shape}')
  # prepare the ode problem
  ode_problem, z_gt = prepare_ode_problem(z_buffer, seg_sequences, seg_frame_times, seg_len, fps)
  # solve the ode
  ode_solution = ode_solver.solve(ode_problem)
  z_pred = ode_solution.ys

  z_pred = z_pred.detach().clone()

  # print the shape of z_pred
  print(f'z_pred shape: {z_pred.shape}')

  # plot all the same things for z_pred
  # plot the variance of the latent vectors for each sequence
  latent_variance_fig_folder = experiment_config["experiment_folder_path"] + "/figures" + "/latent_variance_ode"
  plot_all_latent_variances(z_pred, latent_variance_fig_folder)

  #plot latent trajectories over time for each sequence
  latent_trajectory_fig_folder = experiment_config["experiment_folder_path"] + "/figures" + "/latent_trajectory_ode"
  plot_all_latent_trajectories(z_pred, latent_trajectory_fig_folder)

  # plot the pca of the latent vectors
  latent_pca_fig_folder = experiment_config["experiment_folder_path"] + "/figures" + "/latent_pca_ode"
  plot_all_latent_trajectoryPCA(z_pred, latent_pca_fig_folder)

  # create sdf sample pt files sp we can visualize the sdf
  # we will create a new folder that has the sdf samples for each sequence
  # we will use the sample points from the original data set

  # evaluate_all_sdf_sequences(
  #   auto_decoder, 
  #   z_buffer, 
  #   experiment_config)







def run_experiment(experiment_config_path):
  global logger
  experiment_config = load_experiment_config(experiment_config_path)
  experiments_folder = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/modeling/experiments/results"
  # experiment name is the current time stamp
  experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  # create the experiment folder
  experiment_folder = Path(experiments_folder) / experiment_name
  experiment_folder.mkdir(parents=True, exist_ok=True)
  experiment_config["experiment_folder_path"] = str(experiment_folder)
  experiment_config["experiment_name"] = str(experiment_name)

  log_file_path = experiment_folder / "experiment.log"
  logger = setup_logger(log_file_path)
  logger.info(f"Experiment {experiment_name} started.")

  train_neural_sim(experiment_config)

def main():
  global logger
  experiment_config_path = sys.argv[1]
  run_experiment(experiment_config_path)

  # experiment_folder_path = sys.argv[1]
  # evaluate_experiment(experiment_folder_path)

  




if __name__ == "__main__":
  main()