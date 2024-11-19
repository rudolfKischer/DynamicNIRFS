import torch


class Integrator():

  def __init__(self, model):
    self.model = model



  def step(self, z_t, h = 1.):
    raise NotImplementedError

  def integrate(self, z_0, h=1., num_steps=1):
    z_t = z_0
    Z = []
    for i in range(num_steps):
      z_t = self.step(z_t, h)
      Z.append(z_t)
    return torch.stack(Z, dim=0)

class ForwardEuler(Integrator):

  def __init__(self, model):
    super(ForwardEuler, self).__init__(model)
  
  def step(self, z_t, h=1.):
    dz_dt = self.model(z_t)
    z_t1 = z_t + h * dz_dt
    return z_t1
  
class BackwardEuler(Integrator):

  def __init__(self, model):
    super(BackwardEuler, self).__init__(model)
  
  def step(self, z_t, h=1.):
    dz_dt = self.model(z_t)
    # print(f'z_t_shape: {z_t.shape}')
    # print(f'dz_dt_shape: {dz_dt.shape}')
    k1 = z_t + h * dz_dt
    dz_d_k1 = self.model(k1)
    z_t1 = z_t + h * dz_d_k1
    return z_t1

class RK2(Integrator):

  def __init__(self, model):
    super(RK2, self).__init__(model)
  
  def step(self, z_t, h=1.):
    dz_dt = self.model(z_t)
    k1 = z_t + h * dz_dt
    dz_d_k1 = self.model(k1)
    z_t1 = z_t + h * dz_d_k1
    k2 = z_t + h * dz_d_k1
    dz_d_k2 = self.model(k2)
    z_t1 = z_t + 0.5 * h * (dz_dt + dz_d_k2)
    return z_t1

class RK4(Integrator):

  def __init__(self, model):
    super(RK4, self).__init__(model)
  
  def step(self, z_t, h=1.):
    dz_dt = self.model(z_t)
    k1 = z_t + h * dz_dt
    dz_d_k1 = self.model(k1)
    k2 = z_t + h * dz_d_k1
    dz_d_k2 = self.model(k2)
    k3 = z_t + h * dz_d_k2
    dz_d_k3 = self.model(k3)
    k4 = z_t + h * dz_d_k3
    dz_d_k4 = self.model(k4)
    z_t1 = z_t + (h/6) * (dz_dt + 2 * dz_d_k2 + 2 * dz_d_k3 + dz_d_k4)
    return z_t1