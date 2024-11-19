import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import PIL
import shutil
import math




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

def plot_embeddings(embeddings):

  # plot the variance bar plot
  embeddings = embeddings.detach()

  variance  = torch.var(embeddings, dim=0)
  plt.figure(figsize=(10, 10))
  plt.bar([i for i in range(variance.shape[0])], variance)
  plt.xlabel("Embedding Dimension")
  plt.ylabel("Variance")

  # plot the 2 most variant dimensions over 
  # we want a 2d plot, where we have a have a start and end
  # and we plot the trajectory 
  # one axis is the most variant dimension, the other is the second most variant dimension
  #there should be a line , connecting fi to fi+1

  reduced_embeddings = embeddings[:, variance.topk(2).indices]
  plt.figure(figsize=(10, 10))
  plt.plot(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
  plt.xlabel("Most Variant Dimension")
  plt.ylabel("Second Most Variant Dimension")
  plt.title("Embedding Trajectory")

  # now as time plots, where we plot the trajectory of the embedding over time
  # each embeding dim should get its own color
  # the x axis is time, and the y axis is the value of the embedding

  plt.figure(figsize=(10, 10))
  for i in range(int(embeddings.shape[1] * 0.1)):
    plt.plot(embeddings[:, i], label=f"Dim {i}")
  plt.xlabel("Frame")
  plt.ylabel("Value")
  plt.title("Embedding Trajectory Over Time")





  # show

  plt.show()