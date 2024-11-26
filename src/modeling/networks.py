import torch
import torch.nn as nn
import datetime

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

  def __init__(self, num_codes, code_dim, mlp, embedding_w=None):
    super(AutoDecoder, self).__init__()
    self.num_codes = num_codes
    self.code_dim = code_dim
    self.mlp = mlp
    self.codes = nn.Embedding(num_codes, code_dim)
    if embedding_w is not None:
      self.codes.weight = nn.Parameter(embedding_w)
    else:
      nn.init.normal_(self.codes.weight, mean=0, std=0.1)
  
  def load(self, model_name):
    # self.mlp.load(model_name)
    self.load_state_dict(torch.load(model_name)) 

  def save(self, model_name):
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

  
class DHNODE(nn.Module):
  """
  Damped Hamiltonian Neural ODE

  We want to train a model to evaluate the energy of the system at a given time step
  Then we will use an integrator like backward euler or something more accurate like RK4 to integrate the system over time

  The input to the system is the latent space of the current frame
  
  the output is the system energy
  We then use this energy , to calculate the the current impulse, 
  we then can use that to calculate the new position, based on the velocity, and the previous position

  we then use this to calculate the loss of the model, by comparing the predicted position with the actual position in latent space

  Then we use the error to propogate the loss back to the model

  We can also propogate the loss back to the 

  z(t) = (q(t), p(t))

  mlp = V(q)

  dq/dt = m^-1 * p
  dp/dt = -gamma * m^-1 * p - grad V(q)
  """
  def __init__(self, mlp, gamma=None, m=None, model_folder="models", time_stamp_saved_model=True):
    super(DHNODE, self).__init__()
    self.mlp = mlp
    self.gamma = gamma
    self.m = m
    self.latent_dim = mlp.input_dim
    if not gamma:
      # multiply by 0.1
      self.gamma = 0.1
    self.gamma = torch.eye(self.latent_dim) * self.gamma
    if not m:
      self.m = 1.0
    self.m = torch.eye(self.latent_dim) * self.m
    
    self.m_inv = torch.inverse(self.m)
    self.model_folder = model_folder
    self.time_stamp = time_stamp_saved_model

  def forward(self, x):
    """
    x = z
    z = (q, p) -> shape: (2 * latent_dim,)
    q: position -> shape: (latent_dim,)
    p: velocity -> shape: (latent_dim,)
    m: mass matrix -> shape: (latent_dim, latent_dim), default: identity matrix
    V: potential energy -> scalar

    dq = m^-1 * p 
      p: [1, latent_dim]
      m_inv: [latent_dim, latent_dim]
      dq: [1, latent_dim]
      m^-1 * p: [latent_dim, latent_dim] * [1, latent_dim] = [latent_dim, 1] , so we need to transpose the result
    dp = -gamma * m^-1 * p - grad V(q)
      p: [1, latent_dim]
      m_inv: [latent_dim, latent_dim]
      gamma: [latent_dim, latent_dim]
      grad V(q): [1, latent_dim]
      dp: [1, latent_dim]
    """
    z = x
    z.requires_grad_(True)
    q, p = z[:, :self.latent_dim], z[:, self.latent_dim:]
    # print(f'q_shape: {q.shape}')
    # print(f'p_shape: {p.shape}')
    # print(f'z_shape: {z.shape}')
    V = self.mlp(q)
    dV_dq = torch.autograd.grad(V, q, torch.ones_like(V), create_graph=True, allow_unused=True)[0]
    # print(f'dV_dq_shape: {dV_dq.shape}')
    #flip p so it can multiply with m_inv
    dq_dt = torch.matmul(self.m_inv, p.reshape(-1, 1)).T
    dp_dt = -torch.matmul(torch.matmul(self.gamma, self.m_inv), p.reshape(-1, 1)).T - dV_dq
    # print(f'dp_dt_shape: {dp_dt.shape}')
    # print(f'dq_dt_shape: {dq_dt.shape}')

    dz_dt = torch.cat((dq_dt, dp_dt), dim=1)
    # print(f'dz_dt_shape: dhnode forward :{dz_dt.shape}')
    return dz_dt

class DHNODEIntegrator(nn.Module):
  """
  The dhnode integrator is used to train the DHNODE model
  X = (q, p , num_steps) , where q is the position, p is the velocity, and num_steps is the number of rollout steps into the future
  Z = (Q_h, P_h) , where q_h is the position, and p_h is the velocity at timestep t + num_steps

  Q_h = [q_0, q_1, q_2, ..., q_h]
  P_h = [p_0, p_1, p_2, ..., p_h]

  l = |Z - Z_h|^2

  Z = [z_0, z_1, z_2, ..., z_h]

  """


  def __init__(self, model, integrator, shared_embedding):
    super(DHNODEIntegrator, self).__init__()
    self.model = model
    self.integrator = integrator
    # disable grad for embeddings
    # shared_embedding.weight.requires_grad = False

    self.embeddings = shared_embedding

  
  def get_code(self, code_idx):
    return self.embeddings(code_idx)
  
  def save(self, model_name):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = f"{self.model.model_folder}/{model_name}_dhnode_integrator_{time_str}.pt"
    if self.model.time_stamp:
      output_path = f"{self.model.model_folder}/{model_name}_dhnode_integrator_{time_str}.pt"
    torch.save(self.state_dict(), output_path)
  
  def load(self, model_name):
    self.load_state_dict(torch.load(f"{self.model.model_folder}/{model_name}.pt"))
  
  def forward(self, x):
    # x.shape = (m, 1, latent_dim * 2 + 1) . x = (q, p, num_steps)
    # print(f'x_shape_integrate: {x.shape}')
    # x may be a batch, so we will have to integrate for each item in the batch
    Z = []

    for i in range(x.shape[0]):
      num_steps = x[i,-1]
      z_0 = x[i,:-1]
      # z_0 = z_0.reshape(1, -1)
      # print(f'z_0_shape: {z_0.shape}')
      # print(f'num_steps_shape: {num_steps.shape}')
      z_t = self.integrator.integrate(z_0, num_steps=int(num_steps))
      Z.append(z_t)
    Z = torch.stack(Z, dim=0)
    return Z # [m, num_steps, latent_dim * 2]

class SDFDHNODEjoint(nn.Module):
  """
  The SDFDHNODEjoint model is used to train the DHNODE model
  X = (q, p , num_steps) , where q is the position, p is the velocity, and num_steps is the number of rollout steps into the future
  Z = (Q_h, P_h) , where q_h is the position, and p_h is the velocity at timestep t + num_steps

  Q_h = [q_0, q_1, q_2, ..., q_h]
  P_h = [p_0, p_1, p_2, ..., p_h]

  l = |Z - Z_h|^2

  Z = [z_0, z_1, z_2, ..., z_h]

  """
  def __init__(self, dhnode_integrator, 
               sdf_decoder,
               embedding, 
               model_folder="models", time_stamp_saved_model=True):
    super(SDFDHNODEjoint, self).__init__()
    self.dhnode_integrator = dhnode_integrator
    self.sdf_decoder = sdf_decoder
    self.embedding = embedding
    self.model_folder = model_folder
    self.time_stamp = time_stamp_saved_model
  
  def forward(self, x):
    # will look something like stepping with the ode , and integrator, and then getting 
    # the sdf samples, but for now this is just a placeholder
    pass
  
  def save(self, model_name):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = f"{self.model_folder}/{model_name}_sdf_dhnode_joint_{time_str}.pt"
    if self.time_stamp:
      output_path = f"{self.model_folder}/{model_name}_sdf_dhnode_joint_{time_str}.pt"
    torch.save(self.state_dict(), output_path)
  
  def load(self, model_path):
    self.load_state_dict(torch.load(model_path))
