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


activation_functions = {
  "relu": nn.ReLU(),
  "tanh": nn.Tanh()
}

class MLP(nn.Module):
  """
  PARAMS:
  - layer_dims: list of integers, the dimensions of the hidden layers
  - input_dim: int, the dimension of the input
  - output_dim: int, the dimension of the output
  """
  def __init__(self,
                layer_dims,
                input_dim,
                output_dim,
                hidden_activation="relu",
                output_activation="tanh",
                model_folder="models",
                weight_norm=True,
                time_stamp_saved_model=True
                ):
    super(MLP, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.layer_dims = layer_dims
    self.activation = activation_functions[hidden_activation]
    self.output_activation = activation_functions[output_activation]
    self.model_folder = model_folder
    self.time_stamp = time_stamp_saved_model
    self.weight_norm = weight_norm

    self.init_weights()
  
  def init_weights(self):
    self.layers = nn.ModuleList()
    self.layers.append(nn.Linear(self.input_dim, self.layer_dims[0]))
    for i in range(1, len(self.layer_dims)):
      layer = nn.Linear(self.layer_dims[i-1], self.layer_dims[i])
      if self.weight_norm:
        layer = nn.utils.weight_norm(layer)
      self.layers.append(layer)
    layer = nn.Linear(self.layer_dims[-1], self.output_dim)
    if self.weight_norm:
      layer = nn.utils.weight_norm(layer)
    self.layers.append(layer)


    


  def forward(self, x):
    for layer in self.layers[:-1]:
      x = self.activation(layer(x))
    x = self.output_activation(self.layers[-1](x))
    return x
  
  def save(self, model_name):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = f"{self.model_folder}/{model_name}.pt"
    if self.time_stamp:
      output_path = f"{self.model_folder}/{model_name}_{time_str}.pt"
    torch.save(self.state_dict(), output_path)
  
  def load(self, model_name):
    self.load_state_dict(torch.load(f"{self.model_folder}/{model_name}.pt"))
  

class DeepSDFDecoder(MLP):
  def __init__(self, layer_dims,
                input_dim=3,
                output_dim=1,
                **kwargs):
    super(DeepSDFDecoder, self).__init__(layer_dims, input_dim, output_dim, **kwargs)

  def forward(self, x):
    return super(DeepSDFDecoder, self).forward(x)
  



class AutoDecoder(nn.Module):
  """
  An auto decoder works, by attaching a code to each sample in a particular set of the samples that in the same category.
  The code is shared by all the samples in the category during training.
  The codes are initialized to random values. But during training,  we back propogate the loss from the decoder to the code.
  This will encourage the decoder to codes closer together, if their samples are similar.
  """

  def __init__(self, num_codes, code_dim, mlp):
    super(AutoDecoder, self).__init__()
    self.num_codes = num_codes
    self.code_dim = code_dim
    self.mlp = mlp

    self.codes = nn.Embedding(num_codes, code_dim)
    nn.init.normal_(self.codes.weight, mean=0, std=0.1)
  
  def load(self, model_name):
    # self.mlp.load(model_name)
    self.load_state_dict(torch.load(model_name))

  def save(self, model_name):
    self.mlp.save(model_name)
    # save this model as well
    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = f"{self.mlp.model_folder}/{model_name}_auto_decoder_{time_str}.pt"
    if self.mlp.time_stamp:
      output_path = f"{self.mlp.model_folder}/{model_name}_auto_decoder_{time_str}.pt"
    torch.save(self.state_dict(), output_path)


  def get_code(self, code_idx):
    return self.codes(code_idx)

  def forward(self, x):
    return self.mlp(x)

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
  

class LossPlotter():
  """
  Used to show loss while training a model.
  We can have an active plotter while the model is updating.
  """
  def __init__(self, 
                fig_folder="figures",
                fig_name="loss_plot.png",
                plot_title="Loss Plot",
                plot_x_label="Steps",
                plot_y_label="Loss",
                ):
    self.loss_history = None
    self.fig_folder = fig_folder
    self.fig_name = fig_name
    self.plot_title = plot_title
    self.plot_x_label = plot_x_label
    self.plot_y_label = plot_y_label


    if not os.path.exists(fig_folder):
      os.makedirs(fig_folder)

  
  def plot(self):
    plt.figure(figsize=(10, 10))
    # format: (epoch, batch, loss)
    # X axis: steps
    # Y axis: loss

    for label, loss_history in self.loss_history.items():
      steps = [i for i in range(len(loss_history))]
      losses = [l[-1] for l in loss_history]
      sns.lineplot(x=steps, y=losses, label=label)
    
    plt.xlabel(self.plot_x_label)
    plt.ylabel(self.plot_y_label)
    plt.title(self.plot_title)
    plt.legend()
    plt.savefig(f"{self.fig_folder}/{self.fig_name}")
    plt.close()


  def update(self, data, epoch):
    self.loss_history = data
    self.plot()
  
  def finish(self):
    pass

class SDFPlotter():

  """
  We want to evaluate the model at every point in the dataset while its learning
  and plot the results to create a gif of the model learning.
  We will also save the the results at each step to a pt file.
  the sdf plotter will need:
  - the model
  - the dataset
  - the output folder
  - snapshot_folder_name
  (assume 2d points for now, and ignore y axis)
  at each update, we evaluate the model at every point in the dataset
  then we add the evaluated samples to the frame samples
  we create a snapshot folder in the output folder
  then we create a figure where we plot the points in 2d space using the the coordinates 
  and use the sdf value to color the points
  then after were done we combine all the frames into a gif
  """
  def __init__(self, model, 
               dataset, 
               epochs,
               num_of_snapshots = 100,
               output_folder = "figures",
               snapshot_folder_name="sdf_snapshots",
               remove_snapshots=False,
               plot_3d=True
               ):
    self.model = model
    self.dataset = dataset
    self.X, self.y = dataset
    self.X_orig = self.X
    self.y_orig = self.y
    self.plot_3d = plot_3d
    
    # create new X data set with more samples
    # keep the y the same
    # we want regulare samples over 0 to 1 in x and z
    # if N is the number of samples per dim, then we will have N * N samples
    N = 40 if plot_3d else 100
    # # generate the new X data, y axis as 0.5 (axis 1)
    vals = np.linspace(0, 1, N)
    X = np.array(np.meshgrid(vals, [0.5], vals)).T.reshape(-1, 3)
    if plot_3d:
      # add the y axis as well
      X = np.array(np.meshgrid(vals, vals, vals)).T.reshape(-1, 3)

    
    self.X = torch.tensor(X).float()




    self.output_folder = output_folder
    self.snapshot_folder_name = snapshot_folder_name

    # self.evaluations = self.evaluate_model()
    num_samples = self.X.shape[0]
    y_e = self.model(self.X).reshape((1, num_samples, 1)).detach()
    X_e = self.X.reshape((1, num_samples, 3)).detach()
    self.evaluations = torch.cat((X_e, y_e), dim=2)

    self.snapshot_files = []
    self.snapshot_folder = f"{output_folder}/{snapshot_folder_name}"
    if not os.path.exists(self.snapshot_folder):
      os.makedirs(self.snapshot_folder)
    self.snapshot_interval = max(epochs // num_of_snapshots, 1)
    self.num_of_snapshots = num_of_snapshots
    self.epochs = epochs
    self.step_count = 0
    self.snapshot_count = 0
    self.remove_snapshots = remove_snapshots



  
  def evaluate_model(self):
    num_samples = self.X.shape[0]
    y_e = self.model(self.X).reshape((1, num_samples, 1)).detach()
    # use the original labels for testing
    # y_e = self.y.reshape((1, num_samples, 1)).detach()
    X_e = self.X.reshape((1, num_samples, 3)).detach()
    eval_pt_tensor = torch.cat((X_e, y_e), dim=2)
    return eval_pt_tensor
  
  def plot_snapshot_3d(self, eval_pt_tensor, epoch):
    # we want to similiar to the 2d plot
    # but our data points ore in the unit cube

    # set the positions of the points to 0,0,0 if the sdf value is positive
    # the sdf val is in the 4th column
    # reshape the original
    original_pt_tensor = torch.cat((self.X_orig.reshape((1, self.X_orig.shape[0], 3)), self.y_orig.reshape((1, self.y_orig.shape[0], 1))), dim=2)

    # but shift their x values by + 1.0
    original_pt_tensor[0, :, 0] += 1.0
    # add the original labels to the plot as well
    eval_pt_tensor = torch.cat((eval_pt_tensor, original_pt_tensor), dim=1)


    clamp_val =0.005
    eval_pt_tensor[0, :, 0][eval_pt_tensor[0, :, 3] > clamp_val] = 0
    eval_pt_tensor[0, :, 1][eval_pt_tensor[0, :, 3] > clamp_val] = 0
    eval_pt_tensor[0, :, 2][eval_pt_tensor[0, :, 3] > clamp_val] = 0





    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_box_aspect(aspect=[2, 1, 1])

    # remove border from figure and zoom in
    ax.dist = 3


    # fix axis between 0 and 1

    # adjust view angle
    ax.view_init(elev=20, azim=100)

    norm = plt.Normalize(-0.01, 0.00)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    scatter = ax.scatter(eval_pt_tensor[0, :, 0].numpy(), 
                          eval_pt_tensor[0, :, 1].numpy(),
                          eval_pt_tensor[0, :, 2].numpy(),
                          c=eval_pt_tensor[0, :, 3].numpy(),
                          cmap="coolwarm",
                          s=250,
                          linewidths=0,
                          norm=norm,
                          # make filled circles markers
                          marker="o"

                          )
    


    ax.set_title(f"Epoch: {epoch}")
    output_path = f"{self.snapshot_folder}/epoch_{epoch}.png"
    try:
      plt.savefig(output_path)
      self.snapshot_files.append(output_path)
    except Exception as e:
      print(f"Error saving snapshot: {e}")
    plt.close()
  
  def plot_snapshot(self, eval_pt_tensor, epoch):
    # use the blue red color map
    # plot the points in 2d space
    # use the sdf value to color the points
    # x and z axis are between 0 and 1, assume 2d points
    # we plot them over x and y axis on the plot
    # we use the sdf value to color the points
    plt.figure(figsize=(10, 10))
    x = eval_pt_tensor[0, :, 0].numpy()
    z = eval_pt_tensor[0, :, 2].numpy()
    sdf = eval_pt_tensor[0, :, 3].numpy()
    # set the cool warm range for the sdf values [-0.1, 0.1]
    clamp_dist = 0.1 * 0.5
    sns.scatterplot(x=x, y=z, 
                    hue=sdf, 
                    palette="coolwarm", 
                    hue_norm=(-clamp_dist, clamp_dist),
                    legend=False,
                    # increase the point size
                    s=400,
                    # make them squares, no border
                    marker="s", linewidth=0
                    )
    plt.title(f"Epoch: {epoch}")
    output_path = f"{self.snapshot_folder}/epoch_{epoch}.png"
    try:
      plt.savefig(output_path)
      self.snapshot_files.append(output_path)
    except Exception as e:
      print(f"Error saving snapshot: {e}")
    plt.close()

  
  def update(self, data, epoch):
    self.step_count += 1
    # self.model = model
    # self.y = y_e

    # we dont need the loss data for this plotter
    target_step = int(self.epochs *((math.log((self.num_of_snapshots - self.snapshot_count) + 1) / math.log(self.num_of_snapshots + 1))))

    # reverse log frequency

    if self.step_count < (self.epochs - target_step):
      return
    self.snapshot_count += 1
    

    # evaluate the model
    new_eval = self.evaluate_model()
    # add the new evaluation to the old one
    self.evaluations = torch.cat((self.evaluations, new_eval), dim=0)
    # plot the snapshot
    if self.plot_3d:
      self.plot_snapshot_3d(new_eval, epoch)
    else:
      self.plot_snapshot(new_eval, epoch)
    # delete 

  
  def finish(self):
    # combine the snapshots into a gif
    images = []
    output_path = f"{self.output_folder}/{self.snapshot_folder_name}.gif"
    for cur_file in self.snapshot_files:
      try:
        images.append(PIL.Image.open(cur_file))
      except Exception as e:
        print(f"Error opening snapshot file: {e}")
    # make sure theres a pause before looping by repeating the last frame
    for i in range(50):
      images.append(images[-1])
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=30, loop=0)
    if self.remove_snapshots:
      shutil.rmtree(self.snapshot_folder)
    print(f"Snapshot gif saved to {output_path}")



class LossSDFPlotter():

  def __init__(self, sdf_plotter, loss_plotter):
    self.sdf_plotter = sdf_plotter
    self.loss_plotter = loss_plotter

  def update(self, loss_data, epoch):
    self.sdf_plotter.update(loss_data, epoch)
    self.loss_plotter.update(loss_data, epoch)
  
  def finish(self):
    self.sdf_plotter.finish()
    self.loss_plotter.finish()



  


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

  print(f"Loading data from {tank_samples}")
  tank_pt_tensor = torch.load(tank_samples)
  bunny_pt_tensor = torch.load(bunny_samples)

  # get just the first frames
  tank_pt_tensor_1 = tank_pt_tensor[0:100, :, :]
  # tank_pt_tensor_2 = tank_pt_tensor[99:100, :, :]

  bunny_pt_tensor = bunny_pt_tensor[0:1, :, :]

  # combine the two datasets
  pt_tensor = tank_pt_tensor_1
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
  epochs = 100
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
  num_models = pt_tensor.shape[0]
  y_e = y_e.reshape((num_models, num_samples, 1)).detach()
  print(X_encoded.shape)
  X_e = X_encoded[:, :3].reshape((num_models, num_samples, 3)).detach()


  # remove the code from the input

  eval_pt_tensor = torch.cat((X_e, y_e), dim=2)
  eval_pt_tensor_path = f"{output_model_folder}/{output_model_name}_eval.pt"
  torch.save(eval_pt_tensor, eval_pt_tensor_path)
  print(f"Model evaluation saved to {eval_pt_tensor_path}")



# def resample_autodecoder():

#   model_file = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/modeling/models/autodecoder_tank_2d_motion.pt"
#   model = AutoDecoder(1, 1, None)
#   model.load(model_file)

def main():
  # train_deep_sdf()
  train_deep_sdf_auto_decoder()




if __name__ == "__main__":
  main()