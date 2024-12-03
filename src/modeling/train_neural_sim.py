import sys


def prepare_data():
  pass


def train_neural_sim():

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
  # load configuration

  # load data config into configuration

  # initialize models

  # initialize latent vector buffer
  # shape: (num_seq, num_frames_per_seq, latent_dim)

  # initialize learning rate scheduler

  # initialize 

  # initialize optimizer

  # initialize ode solver

  # enter epoch loop

  # update schedules for learning rate + lambda + segment length

  # prepare dataset for epoch

  # initialize dataloader

  #==================================TRAINING LOOP===================================================================

  # enter iteration loop (batch loop)

  # unload data 
  # send the data to the device
  

  #==================================ODE SOLVER===================================================================
  # if batch_n % ode_update == 0:
  # set gradients to zero

  # prepare date for ode solver

  # run ode solver

  # calculate ode loss

  #==================================SDF DECODER===================================================================

  # get the latent vectors for the ode samples

  # create the gaussian latent vectors

  # pass the batch through the sdf decode with the samples, and latent vectors

  # ==================================BACKPROPAGATION Error===================================================================

  # calculate the sdf loss

  # calculate the joint loss

  # backpropogate the joint loss

  # step the latent buffer optimizer and the ode solver

  # if batch_n % ode_update == 0:
  # step the ode optimizer

  # log the batch losses

  # plot loss and save to fig

  # end iteration loop

  # log learning rate, lambda, segment length

  # end epoch loop

  #==================================END OF TRAINING===================================================================


  # save the model

  # save all configuration info, and generated training info (losses, lr, lambda, segment length, etc)

  # evaluate the experiment







def evaluate_experiment():
  # plot the pca of the latent vectors
  # plot the latent vector trajectory for each sequence for sdf evaluation
  # plot the latent vector trajectory for each sequence for ode + sdf evaluation
  # plot the latent vecotr variation 
  # plot the latent vector variation over time while training


  pass


def main():
  experiment_config = sys.argv[1]

if __name__ == "__main__":
  main()