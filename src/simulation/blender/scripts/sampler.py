import numpy as np
import pickle
import igl
from tqdm import tqdm
import torch
import pathlib
import sys
import trimesh


# pickle file contains list of meshes
# the meshes are dictionaires with two entries 
# 'vertices' and 'faces'
# both are numpy arrays

COLLAPSE_Y_AXIS = True
SWAP_YZ = True


def load_mesh_sequence(pickle_file_path):
  with open(pickle_file_path, "rb") as f:
    mesh_sequence = pickle.load(f)
  return mesh_sequence
  
def random_sample_unit_cube(num_samples):
    samples = np.random.rand(num_samples, 3)
    return samples

def random_sample_unit_cube_even_split(num_samples, mesh):
    # we want to sample such that half of the samples are inside the mesh
    # and half are outside
    # we do this by resapling if our amount of samples is not evenly split, and replacing the ones we have to many of
    # with the ones we have to few of
    sample = np.random.rand(int(num_samples), 3)
    if COLLAPSE_Y_AXIS:
      sample[:, 1] = 0.5
    sdf, _, _ = igl.signed_distance(sample, mesh['vertices'].astype(np.float32), mesh['faces'].astype('int32'), return_normals=False)
    return sample, sdf

def pick_half_split(extra_large_sample, large_sdf, num_samples):
    samples = np.zeros((num_samples, 3)) 
    sdf = np.zeros(num_samples)
    
    num_inside = np.sum(large_sdf < 0)
    num_outside = np.sum(large_sdf >= 0)
    large_sample_ratio = min(num_inside, num_outside) / extra_large_sample.shape[0]
    # print(f'num_inside: {num_inside}, num_outside: {num_outside}, p: {large_sample_ratio}')

    half = num_samples // 2
    num_inside = min(half, num_inside)
    num_outside = min(half, num_outside)
    minority_half = min(num_inside, num_outside)
    split = num_samples - minority_half

    neg_idx = np.where(large_sdf < 0)[0]
    pos_idx = np.where(large_sdf >= 0)[0]
    min_idx, maj_idx = (neg_idx, pos_idx) if num_inside < num_outside else (pos_idx, neg_idx)

    # print(f'samples_shape: {samples.shape}')
    # print(f'extra_large_sample_shape: {extra_large_sample.shape}')
    # print(f'split: {split}')
    samples[0:split, :] = extra_large_sample[maj_idx][0:split, :]
    samples[split:, :] = extra_large_sample[min_idx][0:minority_half, :]
    sdf[0:split] = large_sdf[maj_idx][0:split]
    sdf[split:] = large_sdf[min_idx][0:minority_half]
    
    return samples, sdf, large_sample_ratio

def squish_axis(mesh_sequence, axis):
  for mesh in mesh_sequence:
    mesh['vertices'][:, axis] = np.where(mesh['vertices'][:, axis] < 0.5, 0, 1)
  return mesh_sequence

def sample_mesh(num_samples, mesh, p):
  g = 1.0 / (p) * 1.2
  n = int(num_samples * g) # Grow sample size to account for the fact, we will only be taking a subset, such that half are inside and half are outside
  extra_large_sample, large_sdf = random_sample_unit_cube_even_split(n, mesh)
  samples, sdf, new_p = pick_half_split(extra_large_sample, large_sdf, num_samples)
  # samples = extra_large_sample[0:num_samples]
  # sdf = large_sdf[0:num_samples]
  # new_p = 0.5
  return samples, sdf, new_p, n


def sample_mesh_sequence(pickle_file_path, num_samples):
  mesh_sequence = load_mesh_sequence(pickle_file_path)
  if COLLAPSE_Y_AXIS:
    squish_axis(mesh_sequence, 1)
  frame_samples = np.zeros((len(mesh_sequence), num_samples, 3))
  frame_sdf_values = np.zeros((len(mesh_sequence), num_samples))
  p = 0.5
  with tqdm(total=len(mesh_sequence), desc="Generating Samples") as pbar:
    for i, mesh in enumerate(mesh_sequence):
      frame_samples[i], frame_sdf_values[i], p, N = sample_mesh(num_samples, mesh, p)
      negative_samples = np.sum(frame_sdf_values[i] < 0)
      positive_samples = np.sum(frame_sdf_values[i] >= 0)
      pbar.set_description(f"Generating Samples for Frame {i}, -/+: {negative_samples}/{positive_samples} p: {p:.2f} n: {num_samples} N: {N}")
      pbar.update(1)
  return frame_samples, frame_sdf_values

def swap_yz(mesh_sequence):
  for mesh in mesh_sequence:
    temp = mesh['vertices'][:, 1].copy()
    mesh['vertices'][:, 1] = mesh['vertices'][:, 2]
    mesh['vertices'][:, 2] = temp
  for mesh in mesh_sequence:
    temp = mesh['faces'][:, 0].copy()
    mesh['faces'][:, 0] = mesh['faces'][:, 2]
    mesh['faces'][:, 2] = temp
  return mesh_sequence

def to_pt_tensor(per_frame_samples, per_frame_sdf):
  # cols = 2.
  # shape is (num_frames, num_samples, 4)
  # the first 3 columns are the sample points
  # the last column is the sdf value
  num_frames = per_frame_samples.shape[0]
  num_samples = per_frame_samples.shape[1]
  pt_tensor = torch.zeros(num_frames, num_samples, 4)
  pt_tensor[:, :, 0:3] = torch.tensor(per_frame_samples)
  pt_tensor[:, :, 3] = torch.tensor(per_frame_sdf)
  return pt_tensor
  
def save_pt_tensor(pt_tensor, output_file):
  with open(output_file, "wb") as f:
    torch.save(pt_tensor, f)
   

   

def sample_and_save(pickle_file_path, num_samples):
  frame_samples, frame_sdf_values = sample_mesh_sequence(pickle_file_path, num_samples)
  pickle_path = pathlib.Path(pickle_file_path)
  pt_output_file = pickle_path.parent / (pickle_path.stem + f"_samples_{num_samples}.pt")
  pt_data = to_pt_tensor(frame_samples, frame_sdf_values)
  save_pt_tensor(pt_data, pt_output_file)
  return pt_data, pt_output_file

# Sample arbitrary unprepped single mesh from obj file
def fit_to_unit_cube(mesh):
  # get the bounding box, by get the max and min of each axis
  # get the center of the bounding box
  # translate the mesh to the origin
  # scale the mesh to demensions of 1, (uniformly scale)
  # translate to 0.5, 0.5, 0.5
  # vertices shape: (n, 3)
  bb_min = np.min(mesh['vertices'], axis=0)
  bb_max = np.max(mesh['vertices'], axis=0)
  center = (bb_max + bb_min) / 2
  mesh['vertices'] -= center
  dims = bb_max - bb_min
  scale = 1 / np.max(dims)
  mesh['vertices'] *= scale
  mesh['vertices'] += 0.5
  return mesh

def load_mesh_from_obj(obj_file):
  mesh = trimesh.load(obj_file)
  vertices = mesh.vertices
  faces = mesh.faces
  return {
    'vertices': vertices,
    'faces': faces
  }

def save_mesh_sequence(mesh_sequence, output_file):
  with open(output_file, "wb") as f:
    pickle.dump(mesh_sequence, f)

def sample_single_mesh(obj_file, num_samples):
  mesh = load_mesh_from_obj(obj_file)
  mesh = fit_to_unit_cube(mesh)
  # mesh to mesh sequence as to pickle file
  mesh_sequence = [mesh]
  if SWAP_YZ:
    mesh_sequence = swap_yz(mesh_sequence)
  output_file = pathlib.Path(obj_file).parent / (pathlib.Path(obj_file).stem + ".pkl")
  save_mesh_sequence(mesh_sequence, output_file)


  samples, sdf = random_sample_unit_cube_even_split(num_samples, mesh)
  return samples, sdf

def sample_and_save_single_mesh(obj_file, num_samples):
  samples, sdf = sample_single_mesh(obj_file, num_samples)
  obj_path = pathlib.Path(obj_file)
  pt_output_file = obj_path.parent / (obj_path.stem + f"_samples_{num_samples}.pt")
  # reshape to be a single frame
  samples = samples.reshape((1, num_samples, 3))
  sdf = sdf.reshape((1, num_samples))
  pt_data = to_pt_tensor(samples, sdf)
  save_pt_tensor(pt_data, pt_output_file)
  return pt_data, pt_output_file


def main():
  
  # # load the pickle file
  pickle_path = sys.argv[1]
  # pickle_path = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/data/tank_2d_motion_2024-11-02_21-11-47/tank_2d_motion_2024-11-02_21-11-47_simplified.pkl"
  # # pickle_path = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/meshes/tank_2d_motion_2024-11-02_13-48-16/tank_2d_motion_2024-11-02_13-48-16.pkl"
  n = int(sys.argv[2])
  # pickle_path = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/data/tank_2d_motion_2024-11-02_23-28-02/tank_2d_motion_2024-11-02_23-28-02_simplified.pkl"
  # n = 1000
  sample_and_save(pickle_path, n)

  # single mesh test
  # obj_file = '/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/stanford-bunny.obj'
  # n = 10000
  # sample_and_save_single_mesh(obj_file, n)


if __name__ == "__main__":
  main()