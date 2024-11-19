import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import torch
import json
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import os
import PIL
import shutil
import math

from plotters import *
from integrators import *
from networks import *


if torch.backends.mps.is_available():
  device = torch.device("mps")  # Use the Metal Performance Shaders (MPS) backend on Apple Silicon
  print("Using MPS for GPU acceleration")
else:
  device = torch.device("cpu")
  print("Using CPU as MPS is not available")



"""
we want to try training an MLP to predict the sdf of a single mesh as a test.
- move on to generalizing over many shapes with a decoder only network
- setup an mlp with torch
- load in the data we are trying to learn
- split the data into a training and evaluation set
- setup a loss function and a training loop
- train the model, and collect the loss and accuracy over time
- save the model to the models folder
- evaluate the models performance
- inference the model and visualse the sdf

- expected input data format: 
   - .pt with shape (num_frames, num_samples, 4), 4 -> (x, y, z, sdf)

"""



    



class EncodedDataset(Dataset):

  def __init__(self, X, y, code_ids, decoder):
    assert X.shape[0] == y.shape[0] == code_ids.shape[0]
    self.X = X
    self.y = y
    self.code_ids = code_ids
    self.decoder = decoder
  
  
  def __len__(self):
    return self.X.shape[0]
  
  def __getitem__(self, idx):
    x = self.X[idx]
    y = self.y[idx]
    code_id = self.code_ids[idx]
    code = self.decoder.get_code(code_id)
    combined = torch.cat((x, code), dim=0)
    # combined = x
    return combined, y

class EncodedDHNODEDataset(Dataset):

  def __init__(self, num_steps, decoder, latent_dim, training=True):
    self.num_steps = num_steps
    self.decoder = decoder
    self.embeddings = decoder.embeddings
    self.latent_dim = latent_dim
    self.num_of_embeddings = self.embeddings.weight.shape[0]
    self.training = training

    self.stead_state_padding = int(self.num_of_embeddings * 0.1)
  
  def __len__(self):
    return self.num_of_embeddings

  def __getitem__(self, idx):
    # idx may be a  slice
    # X = (q, p, num_steps)
    # Y = (Q_h, P_h) , shape = (num_steps, latent_dim * 2)
    # to get Y we just get the codes from idx to idx + num_steps
    # print(f'embedding_shape: {self.embeddings.weight.shape}') 

    if self.training:
      idx = torch.randint(0, self.num_of_embeddings, (1,)).item()

    code = self.embeddings(torch.tensor([idx]))
    # print(f'code shape: {code.shape}') # [m, 32]
    # print(f'num_steps: {torch.tensor([self.num_steps]).shape}') # [1]
    # expand the num steps to be [1, m] where m is the batch size

    cur_steps = self.num_steps

    num_steps = torch.tensor([cur_steps]).expand(code.shape[0], 1)
    x = torch.cat((code, num_steps), dim=1)
    # print(f'x_shape_ dataset: {x.shape}')
    start = idx
    end = idx + cur_steps

    end = min(end, self.num_of_embeddings)
    # pad with the last id if idx + cur_steps > num_of_embeddings
    # pad it (idx + cur_steps - num_of_embeddings) times


    y = self.embeddings(torch.arange(start, end))

    last_code = self.embeddings(torch.tensor([end - 1]))

    if (idx + cur_steps) > self.num_of_embeddings:
      padding = idx + cur_steps - self.num_of_embeddings
      y = torch.cat((y, last_code.expand(padding, -1)), dim=0)
    
    # # the steady state is the first code in the list of codes
    # steady_state = self.embeddings(torch.tensor([0]))

    # # we append this to the end of the y tensor
    # y = torch.cat((y, steady_state.expand(self.stead_state_padding, -1)), dim=0)



    
    return x, y

def train(model,
          dataset,
          epochs, 
          optimizer,
          batch_size,
          loss_fn,
          plotter=None,
          val_split= 0.2,
          output_model_folder="models",
          output_model_name="model"
          ):
  # split the data into a training and validation set
  val_size = int(val_split * len(dataset))
  train_size = len(dataset) - val_size
  train_data, val_data = random_split(dataset, [train_size, val_size])


  train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
  val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
  val_test_p = 0.1
  val_test_amount = int(val_test_p * len(val_data))
  val_test_amount = min(max(val_test_amount, 3), len(val_data))

  loss_history = {
    "train": [],
    "val": []
  }
  with tqdm(total=epochs*len(train_dataloader)) as pbar:
    iteration = 0
    for epoch in range(epochs):
      batch_loss_history = {
        "train": [],
        "val": []
      }
      for i, (X_batch, y_batch) in enumerate(train_dataloader):
        # move the data to the device
        # X_batch = X_batch.to(device)
        # code_ids = code_ids.to(device)
        # y_batch = y_batch.to(device)

        # print(next(model.parameters()).device)

        # model.to(device)
        # concat 
        # codes = model.get_code(code_ids)
        # X_batch = torch.cat((X_batch, codes), dim=1)



        iteration += 1
        optimizer.zero_grad()
        y_pred = model(X_batch)
        t_loss = loss_fn(y_pred, y_batch)

        # sample a random subset of the validation data to evaluate the model
        val_test_x, val_test_y = next(iter(val_dataloader))

        # val_test_x = val_test_x.to(device)
        # val_code_ids = val_code_ids.to(device)
        # val_test_y = val_test_y.to(device)

        # val_test_codes = model.get_code(val_code_ids)
        # val_test_x = torch.cat((val_test_x, val_test_codes), dim=1)

        val_test_y_pred = model(val_test_x)
        v_loss = loss_fn(val_test_y_pred, val_test_y)

        t_loss.backward()
        optimizer.step()
        pbar.update(1)
        pbar.set_description(f"Epoch: {epoch}, Loss: {t_loss.item():.2f} batch: {i}/{len(train_dataloader)} val_loss: {v_loss.item():.2f}")
        batch_loss_history["train"].append((epoch, i, t_loss.item()))
        batch_loss_history["val"].append((epoch, i, v_loss.item()))
      # take the average loss over the epoch
      avg_train_loss = np.mean([l[2] for l in batch_loss_history["train"]])
      avg_val_loss = np.mean([l[2] for l in batch_loss_history["val"]])
      loss_history["train"].append((epoch, avg_train_loss))
      loss_history["val"].append((epoch, avg_val_loss))
      if plotter:
        plotter.update(loss_history, epoch)
  plotter.finish()
  model.save(output_model_name)
  print()
  return loss_history, train_data, val_data

def l1_loss(y_pred, y_true):
  return torch.mean(torch.abs(y_pred - y_true))

def clamped_l1_loss(y_pred, y_true, clamp_val=0.1):
  # clamp y_pred to -clamp_val, clamp_val
  # clamp y_true to -clamp_val, clamp_val
  y_pred = torch.clamp(y_pred, -clamp_val, clamp_val)
  y_true = torch.clamp(y_true, -clamp_val, clamp_val)
  return l1_loss(y_pred, y_true)

def train_deep_sdf():

  # load in the data
  # test model : stanford bunny, 2d slice
  # format: (num_frames, num_samples, 4), 4 -> (x, y, z, sdf)
  # sdf_samples = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/data/tank_2d_motion_2024-11-02_23-28-02/tank_2d_motion_2024-11-02_23-28-02_simplified_samples_10000.pt"
  # output_name = 'tank_2d_motion'

  sdf_samples = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/stanford-bunny_samples_10000.pt"
  output_name = 'bunny_sdf'

  print(f"Loading data from {sdf_samples}")
  pt_tensor = torch.load(sdf_samples)

  # get just the first frame
  pt_tensor = pt_tensor[0:1, :, :]

  X, Y = pt_tensor[:, :, 0:3], pt_tensor[:, :, 3]
  X = X.reshape(-1, 3)
  Y = Y.reshape(-1, 1)
  data = (X, Y)
  # setup the model
  width, height = 512, 8
  layer_dims = [width] * height
  model = DeepSDFDecoder(
    layer_dims,
    time_stamp_saved_model=False
    )

  loss_fn = clamped_l1_loss
  # setup the optimizer
  optimizer = optim.Adam(model.parameters(), lr=0.00001)
  batch_size = 32
  epochs = 200
  loss_plotter = LossPlotter()
  sdf_plotter = SDFPlotter(model, 
                      data,
                      epochs,
                      num_of_snapshots=100,
                      output_folder="figures",
                      snapshot_folder_name=output_name,
  )
  # plotter = loss_plotter
  plotter = LossSDFPlotter(sdf_plotter, loss_plotter)





  output_model_name = "deep_sdf_decoder_tank_2d_motion"
  output_model_folder = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/modeling/models"

  # print the params
  print(f"Training model with params:\n")
  param_dict = {
    "width": width,
    "height": height,
    "batch_size": batch_size,
    "epochs": epochs,
    "optimizer": optimizer.__class__.__name__,
    "loss_fn": loss_fn.__name__,
    "output_model_name": output_model_name,
    "output_model_folder": output_model_folder
  }
  print(json.dumps(param_dict, indent=4))

  X, y = data
  dataset = TensorDataset(X, y)

  loss_history, train_data, val_data = train(model, 
                                             dataset,
                                             epochs, 
                                             optimizer, 
                                             batch_size, 
                                             loss_fn, 
                                             plotter=plotter, 
                                              output_model_folder=output_model_folder,
                                              output_model_name=output_model_name
                                              )

  # evaluate the model on every point in the dataset, and save to pt file

  y_e = model(X)
  # save to pt tensor together with X
  num_samples = X.shape[0]
  y_e = y_e.reshape((1, num_samples, 1)).detach()
  X_e = X.reshape((1, num_samples, 3)).detach()

  eval_pt_tensor = torch.cat((X_e, y_e), dim=2)
  eval_pt_tensor_path = f"{output_model_folder}/{output_model_name}_eval.pt"
  torch.save(eval_pt_tensor, eval_pt_tensor_path)
  print(f"Model evaluation saved to {eval_pt_tensor_path}")


def flatten_models_samples(pt_tensor):
  num_models, num_samples, _ = pt_tensor.shape # (num_models, num_samples, 4)

  model_ids = torch.arange(num_models).view(-1, 1, 1)
  model_ids = model_ids.expand(num_models, num_samples, 1)

  pt_tensor_with_ids = torch.cat((model_ids, pt_tensor), dim=2)
  # flatten the tensor
  pt_tensor_flat = pt_tensor_with_ids.reshape(-1, 5)

  code_ids = pt_tensor_flat[:, 0].long()
  pt_tensor = pt_tensor_flat[:, 1:]
  return pt_tensor, code_ids


def train_deep_sdf_auto_decoder():

  tank_samples = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/data/tank_2d_motion_2024-11-02_23-28-02/tank_2d_motion_2024-11-02_23-28-02_simplified_samples_1000.pt"
  bunny_samples = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/stanford-bunny_samples_1000.pt"
  output_name = 'decoder_test'

  samples_file_name = tank_samples

  print(f"Loading data from {samples_file_name}")
  # tank_pt_tensor = torch.load(tank_samples)
  # bunny_pt_tensor = torch.load(bunny_samples)
  pt_tensor = torch.load(samples_file_name)

  # get just the first frames
  # tank_pt_tensor_1 = tank_pt_tensor[0:100, :, :]
  # tank_pt_tensor_2 = tank_pt_tensor[99:100, :, :]

  # bunny_pt_tensor = bunny_pt_tensor[0:1, :, :]

  # combine the two datasets
  pt_tensor = pt_tensor[0:100, :, :]
  # pt_tensor = torch.cat((tank_pt_tensor_1, bunny_pt_tensor), dim=0)
  # pt_tensor = torch.cat((pt_tensor, tank_pt_tensor_2), dim=0)

  # remove about 90 % of the sampels
  # new_num_samples = int(pt_tensor.shape[1] * 0.4)
  # pt_tensor = pt_tensor[:, 0:new_num_samples, :]


  # pt_tensor_shape = (#num_models, #num_samples, 4)

  # we want to reformat into the following shape
  # pt_tensor_shape = (#num_samples * #num_models, 5) # where the new column is the index of the model in the original

  # then we will want to split of the ids, so code_ids = (num_samples * num_models, 1), pt_tensor = (num_samples * num_models, 4)

  pt_tensor_flat, code_ids = flatten_models_samples(pt_tensor)

  X, Y = pt_tensor_flat[:, :3], pt_tensor_flat[:, 3].reshape(-1, 1)
  data = (X, Y)

  embedding_len = 32

  width, height = 512, 6
  layer_dims = [width] * height
  mlp = DeepSDFDecoder(
    layer_dims,
    input_dim=(X.shape[1] + embedding_len),
    time_stamp_saved_model=False
    )
  
  model = AutoDecoder(pt_tensor.shape[0], embedding_len, mlp)
  
  loss_fn = clamped_l1_loss
  optimizer = optim.Adam(model.parameters(), lr=0.00005)
  batch_size = 1024
  epochs = 30
  loss_plotter = LossPlotter()
  plotter = loss_plotter

  output_model_name = "autodecoder_tank_2d_motion"
  output_model_folder = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/modeling/models"

  encoded_dataset = EncodedDataset(X, Y, code_ids, model)
  loss_history, train_data, val_data = train(model, 
                                             encoded_dataset,
                                             epochs, 
                                             optimizer, 
                                             batch_size, 
                                             loss_fn, 
                                             plotter=plotter, 
                                              output_model_folder=output_model_folder,
                                              output_model_name=output_model_name
                                              )
  
  # evaluate the model on every model in the dataset, and save to pt file
  # get encoded_X from the encoded dataset
  # we need the actual codes, the ids are not useful

  model.eval()
  with torch.no_grad():
      data_loader = DataLoader(encoded_dataset, batch_size=pt_tensor_flat.shape[0], shuffle=False)
      X_encoded, y_encoded = next(iter(data_loader))
      y_e = model(X_encoded)

  # data_loader = DataLoader(encoded_dataset, batch_size=pt_tensor_flat.shape[0], shuffle=False)
  # X_encoded, y_encoded = next(iter(data_loader))

  # y_e = model(X_encoded)



  # reshape so it is no longer flat
  # we want the first dimension to be used to index into the specific model
  # we want the second dimension to be the samples, and the third to be the the input or output
  num_samples = pt_tensor.shape[1]
  num_models = pt_tensor.shape[0] #num_models : num_frames
  y_e = y_e.reshape((num_models, num_samples, 1)).detach()
  print(X_encoded.shape)
  X_e = X_encoded[:, :3].reshape((num_models, num_samples, 3)).detach()


  # remove the code from the input

  eval_pt_tensor = torch.cat((X_e, y_e), dim=2)
  eval_pt_tensor_path = f"{output_model_folder}/{output_model_name}_eval.pt"
  torch.save(eval_pt_tensor, eval_pt_tensor_path)
  print(f"Model evaluation saved to {eval_pt_tensor_path}")

  # save a new version of the samples, but with the codes included so they can be used later
  # we want to save it to the same folder as the original samples file
  
  encoded_samples_path = samples_file_name.replace(".pt", f"_encoded_{output_model_name}.pt")
  # we want the output to be F X E where F is the number of frames, and E is the embedding length
  print(f'pt_tensor_shape: {pt_tensor.shape}') # (num_models, num_samples, 4)
  # so we have to get the codes from the model
  encoded_pt_tensor = torch.zeros([num_models, embedding_len])
  for i in range(num_models):
    code = model.get_code(torch.tensor(i))
    encoded_pt_tensor[i] = code
  
  # save the encoded samples
  torch.save(encoded_pt_tensor, encoded_samples_path)






def train_odenet():

  deep_sdf_model_file = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/modeling/models/autodecoder_tank_2d_motion_auto_decoder_2024-11-18-14-09-09.pt"
  mlp = DeepSDFDecoder(layer_dims=[512] * 6, input_dim=35, output_dim=1)
  model = AutoDecoder(100, 32, mlp)
  model.load(deep_sdf_model_file)

  latent_vectors = model.codes.weight

  # to train our model we will construct an X and Y
  # 
  # X = (q, p, num_steps) , where q is the position, p is the velocity, and num_steps is the number of rollout steps into the future
  # Z = (Q_h, P_h) , where q_h is the position, and p_h is the velocity at timestep t + num_steps

  # for now we will just use a fixed number of steps for all samples
  # the q is unknown to start, so we can set it to 0 or random values
  # then to create Y we can create views of X, where is the current index up to i + num_steps
  # if i + 10 < len(X), then we can set the num of steps to len(X) - i for those samples
  num_steps = 4
  # X.shape = (latent_dim * 2 + 1) . x = (q, p, num_steps)
  # copy the latent vectors to the first half of X

  # Z = (Q_h, P_h) , where q_h is the position, and p_h is the velocity at timestep t + num_steps
  # Y.shape = [N, num_steps, latent_vectors.shape[1] * 2]
  # we want to create a view of X for each sample, where we take the current index up to i + num_steps
  # we want to create an encoded dataset
  shared_embeddings = nn.Embedding(latent_vectors.shape[0], latent_vectors.shape[1] * 2)
  # append latent_vector of 0s to the second half, these are the p values
  shared_latent_vectors = torch.cat((latent_vectors, torch.zeros_like(latent_vectors)), dim=1)
  shared_embeddings.weight = nn.Parameter(shared_latent_vectors)

  # detach the shared embeddings so they cannot be updated
  shared_embeddings.weight.requires_grad = False

  # print the share embeddings shape
  print(f'shared_embeddings: {shared_embeddings.weight.shape}') # 100 64

  # create view of the embedding weights for X 

  

  dhnode_mlp = MLP(input_dim=latent_vectors.shape[1], output_dim=1, layer_dims=[512] * 4)
  dhnode = DHNODE(dhnode_mlp)
  integrator = RK2(dhnode)
  # integrator = RK4(dhnode)
  dhnode_integrator = DHNODEIntegrator(dhnode, integrator, shared_embeddings)


  def dhnode_loss(Z_p, Z):
    # for now just use Q_p - Q , instead of Q_p - Q, P_p - P

    steady_state = shared_embeddings(torch.tensor([0]))
    steady_state = steady_state[:, :steady_state.shape[-1] // 2]

    dims = Z.shape[-1] // 2
    # print(f'Z_p_shape: {Z_p.shape}')
    # print(f'Z_shape: {Z.shape}')
    # squeeze axis 2 for z_p
    Z_p = Z_p.squeeze(2) 
    Z_p = Z_p[:, :, :dims]
    Z = Z[:, :, :dims]

    # add extra weight to the very last two points in the sequence


    return torch.mean((Z_p - Z)**2) #+ 0.5 * torch.mean((Z_p[-1] - steady_state)**2)


  loss_fn = dhnode_loss
  optimizer = optim.Adam(dhnode_integrator.parameters(), lr=0.0001)
  batch_size = 10
  loss_plotter = LossPlotter()
  plotter = loss_plotter


  # for i in tqdm(range(1, num_steps + 1, 1)):
  epochs = 150 #+ int(i / float(num_steps) * 50) 
  cur_num_steps = num_steps
  dhnode_dataset = EncodedDHNODEDataset(cur_num_steps, dhnode_integrator, shared_latent_vectors.shape[1])
  loss_history, train_data, val_data = train(dhnode_integrator,
                                              dhnode_dataset,
                                              epochs,
                                              optimizer,
                                              batch_size,
                                              loss_fn,
                                              plotter=plotter,
                                              output_model_folder="models",
                                              output_model_name="dhnode_tank_2d_motion"
                                              )
    
  

  
  # evaluate the model on the dataset with a fixed number of steps
  # evaluate the model for the length of the whole dataset


  eval_data_set = EncodedDHNODEDataset(len(dhnode_dataset), dhnode_integrator, shared_latent_vectors.shape[1], training=False)
  eval_data_loader = DataLoader(eval_data_set, batch_size=1, shuffle=False)
  eval_data = next(iter(eval_data_loader))
  X, Z = eval_data
  Z_p = dhnode_integrator(X)

  print(f'z_p: {Z_p.shape}')
  # collapse the axis 2, because its just 1, right now shape [1, N, 1, latent_dim] -> [1, N, latent_dim]
  Z_p = Z_p.squeeze(2)
  Z_p = Z_p.squeeze(0)
  Q_p = Z_p[:, :Z_p.shape[-1] // 2]

  # now use test_auto_decoder to evaluate the model
  # but make sure to use only the Q_h, not the P_h
  test_auto_decoder(embeddings=Q_p)













def test_auto_decoder(embeddings=None):
    model_file = '/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/modeling/models/autodecoder_tank_2d_motion_auto_decoder_2024-11-18-14-09-09.pt'
    model_file_loaded = torch.load(model_file)
    for key in model_file_loaded.keys():
      print(f'{key}: {model_file_loaded[key].shape}')

    mlp = DeepSDFDecoder(layer_dims=[512] * 6, input_dim=35, output_dim=1)
    model = AutoDecoder(100, 32, mlp)
    model.load(model_file)
    
    if embeddings is None:
      embeddings = model.codes.weight
    print(embeddings.shape)

    f = embeddings.shape[0]
    e = embeddings.shape[1]

    embeddings = embeddings.detach()

    # variance = torch.var(embeddings, dim=0)
    # mean = torch.mean(embeddings, dim=0)
    # # sort the variance
    # variance, _ = torch.sort(variance, descending=True)

    # # only keep the top k dimensions with the most variance, rest go to 0
    # k = 30
    # variance = variance[:k]
    # # get the indices of the top k dimensions
    # indices = embeddings.var(dim=0).topk(k).indices
    # # set the embeddings to mean for the indices that are not in the top k
    # # get the indices that are not the top k
    # not_top_k = [i for i in range(e) if i not in indices]
    # for i in not_top_k:
    #   embeddings[:, i] = mean[i]

    # uniformly sample the unit cube for each frame
    # X = F x [N^3] x [3], Y = F x [N^3] x [1]
    n = 100
    # x = [x, code]
    X = torch.zeros((f, n**2, 3 + e))
    # fill the positions with random values between 0 and 1, except for y axis, set to 0.5
    X[:, :, :3] = torch.rand((f, n**2, 3))
    X[:, :, 1] = 0.5
    # fill the codes with the embeddings
    print(f'X_shape: {X.shape}')
    print(f'embeddings_shape: {embeddings.shape}')
    for i in range(f):
      X[i, :, 3:] = embeddings[i].repeat(n**2, 1)

    
    # evaluate the model
    model.eval()
    Y = torch.zeros((f, n**2, 1))
    for i in tqdm(range(f)):
      Y[i] = model(X[i])
    
    # remove the code from the input ,and attach the input to the output
    eval_pt_tensor = torch.cat((X[:, :, :3], Y), dim=2)

    # we need to attach the frame number to the index

    output_eval = 'test_eval.pt'
    # detach grad
    eval_pt_tensor = eval_pt_tensor.detach()
    torch.save(eval_pt_tensor, output_eval)

    np_embed = embeddings.detach()
    plot_embeddings(np_embed)









def main():
  # train_deep_sdf()
  # train_deep_sdf_auto_decoder()
  # test_auto_decoder()
  train_odenet()

  
  




if __name__ == "__main__":
  main()