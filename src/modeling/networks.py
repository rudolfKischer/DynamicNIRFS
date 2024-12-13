import torch
import torch.nn as nn
import datetime
from typing import *
from functools import wraps
from dataclasses import dataclass
import torchode

activation_functions = {
  "relu": nn.ReLU(),
  "tanh": nn.Tanh(),
  "identity": nn.Identity(),
}

def create_linear_layers(dimensions: List[int], activation: str="relu", output_activation: str="tanh"):
  layers = []
  input_dim = dimensions[0]
  for dim in dimensions[1:]:
    layers.append(nn.Linear(input_dim, dim))
    layers.append(activation_functions[activation])
    input_dim = dim
  layers.append(nn.Linear(input_dim, 1))
  layers.append(activation_functions[output_activation])
  return nn.Sequential(*layers)

def save_model(config, state_dict, path):
  torch.save({
    "config": config,
    "state_dict": state_dict,
    "timestamp": datetime.datetime.now()
  }, path)


@dataclass
class AutoDecoderConfig:
  """
  DecoderConfig is a data class that holds the configuration for a decoder network.

  Attributes:
    embedding_length (int): The length of the embedding vector.
    hidden_dimensions (List[int]): A list of integers representing the dimensions of the hidden layers.
    h_activation (str): The activation function to use for the hidden layers. Default is "relu".
    o_activation (str): The activation function to use for the output layer. Default is "tanh".
    name (str): The name of the decoder. Default is "auto_decoder".
  """
  embedding_length: int
  hidden_dimensions: List[int]
  h_activation: str = "relu"
  o_activation: str = "tanh"
  name: str = "auto_decoder"

class AutoDecoder(nn.Module):
  """
  Deep Sdf Auto Decoder
  ([x,y,z], latent) -> distance to surface 
  """

  def __init__(self, 
              config: AutoDecoderConfig
              ):
    """
    

    Args:
      config (DecoderConfig): The configuration for the decoder network.
    """
    super(AutoDecoder, self).__init__()
    # type check config
    assert isinstance(config, AutoDecoderConfig)
    self.config = config
    # unpack config into class attributes
    for key, value in config.__dict__.items():
      setattr(self, key, value)
    input_dim = 3 + self.embedding_length
    output_dim = 1
    dims = [input_dim] + self.hidden_dimensions + [output_dim]

    self.layers = create_linear_layers(dims, self.h_activation, self.o_activation)
  
  def forward(self, x: torch.Tensor, latent: torch.Tensor):

    # concatenate the expanded latent vector with x
    # normalize latent
    # latent = latent / latent.norm(dim=-1, keepdim=True)
    # set latent to 0 for now
    # new_latent = torch.zeros_like(latent)

  

    x = torch.cat([x, latent], dim=-1)
    # print(f'decoder input shape: {x.shape}')

    return self.layers(x)
  
  def save(self, path: str):
    save_model(self.config, self.state_dict(), path)

  
  @staticmethod
  def load(path: str):
    checkpoint = torch.load(path)
    config = checkpoint["config"]
    model = AutoDecoder(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model



@dataclass
class LatentODEConfig:
  embedding_length: int
  hidden_dimensions: List[int]
  h_activation: str = "relu"
  o_activation: str = "tanh"
  name: str = "latent_ode"
  alpha: float = 1.0
  gamma: float = 1.0
  order: int = 2

class LatentODE(nn.Module):
  """
  The Latent ODE has an Energy Network.
  The energy network is used to model the potential energy of the latent space given a state.
  It takes in the state, and its higher order derivatives,
  and outputs the potential energy of the latent space as a scalar.
  """



  def __init__(self, config: LatentODEConfig):
    super(LatentODE, self).__init__()
    # type check config
    assert isinstance(config, LatentODEConfig)
    self.config = config
    # unpack config into class attributes
    for key, value in config.__dict__.items():
      setattr(self, key, value)
    input_dim = self.embedding_length #* self.order
    output_dim = 1
    self.alpha_M = torch.full((self.embedding_length,), self.alpha)
    self.gamma_M = torch.full((self.embedding_length,), self.gamma)

    # disable make alpha nd gamma non trainable
    self.alpha_M.requires_grad_(False)
    self.gamma_M.requires_grad_(False)


    dims = [input_dim] + self.hidden_dimensions + [output_dim]
    self.layers = create_linear_layers(dims, self.h_activation, self.o_activation)

    # initialize the weights of the energy network
    # for layer in self.layers:
    #   if isinstance(layer, nn.Linear):
    #     nn.init.xavier_normal_(layer.weight)
    #     nn.init.zeros_(layer.bias)
  
  def forward_2(self, t: torch.Tensor, x: torch.Tensor):
    """
    q: latent state
    p: latent velocity
    V: potential energy
    """
    # x.shape = (batch_size, 2 * embedding_length)
    q = x[:, :self.embedding_length]
    p = x[:, self.embedding_length:]
    # normalize q
    q = q / q.norm(dim=-1, keepdim=True)
    V = self.layers(q)

    # calculate derivatives
    dV_dq = torch.autograd.grad(
      V, # output
      q, # input
      grad_outputs=torch.ones_like(V), # gradient of the output
      allow_unused=False, # Make an error if any of the input is not used
      create_graph=True # Create a graph to compute the gradient of the gradient
    )[0]

    dq = self.alpha_M.square() * p
    dp = -dV_dq - self.gamma_M.square() * dq
    return torch.cat([dq, dp], dim=-1)

  def forward(self, t: torch.Tensor, x: torch.Tensor):
    with torch.set_grad_enabled(True):
      x.requires_grad_(True)
      if self.order == 2:
        result = self.forward_2(t, x)
      else:
        raise NotImplementedError(f"Order {self.order} is not implemented.")
    return result
  
  def save(self, path: str):
    save_model(self.config, self.state_dict(), path)
  
  @staticmethod
  def load(path: str):
    checkpoint = torch.load(path)
    config = checkpoint["config"]
    model = LatentODE(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model


# Params
# atol
# rtol

integration_methods = {
  "tsit5": torchode.Tsit5
}

def initialize_ode_solver(
    ode: nn.Module,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    method: str = "tsit5",
):
  ode_term = torchode.ODETerm(ode)
  stepper = integration_methods[method](term=ode_term)
  step_controller = torchode.IntegralController(atol=atol, rtol=rtol, term=ode_term)
  ode_solver = torchode.AutoDiffAdjoint(stepper, step_controller)
  # still needs to be sent to device
  return ode_solver




  





