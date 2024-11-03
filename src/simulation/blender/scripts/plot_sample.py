import trimesh
import pyvista as pv
import numpy as np
import pickle
import torch
import pathlib
import sys


def load_mesh_sequence(pickle_file_path):
  with open(pickle_file_path, "rb") as f:
    mesh_sequence = pickle.load(f)
  return mesh_sequence

def plot_mesh_sequence(mesh_sequence, sample_points=None, sample_sdf_values=None):
    # Initialize the PyVista plotter
    plotter = pv.Plotter()
    current_frame = [0]  # Using a list to allow modification inside the event functions

    highlighted_vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0]
    ])

    plotter.add_points(highlighted_vertices, color="red", point_size=10)

    hide_mesh = False
    
    # Function to update the displayed mesh for a specific frame
    def update_mesh(frame):
        plotter.clear()  # Clear the previous frame
        frame_mesh = trimesh.Trimesh(vertices=mesh_sequence[frame]['vertices'], faces=mesh_sequence[frame]['faces'])
        pv_mesh = pv.wrap(frame_mesh)
        if sample_points is not None:
            # use the sdf values to color the points
            plotter.add_points(sample_points[frame], 
                               scalars=sample_sdf_values[frame], 
                               cmap="coolwarm",
                               clim=(-0.1, 0.01), 
                               point_size=15)
        if not hide_mesh:
          plotter.add_mesh(pv_mesh, color="lightblue", show_edges=True)
        plotter.add_points(highlighted_vertices, color="black", point_size=10)
        plotter.render()  # Refresh to display the updated mesh

    # Event handler to go to the next frame
    def next_frame():
        if current_frame[0] < len(mesh_sequence) - 1:
            current_frame[0] += 1
            update_mesh(current_frame[0])

    # Event handler to go to the previous frame
    def prev_frame():
        if current_frame[0] > 0:
            current_frame[0] -= 1
            update_mesh(current_frame[0])
    
    def toggle_mesh():
      nonlocal hide_mesh
      hide_mesh = not hide_mesh
      update_mesh(current_frame[0])

    # Bind the arrow keys to the event handlers
    plotter.add_key_event("Right", next_frame)
    plotter.add_key_event("Left", prev_frame)
    plotter.add_key_event("m", toggle_mesh)

    # Show the initial frame
    update_mesh(current_frame[0])

    # plotter.enable_parallel_projection()
    # plotter.camera_position = [(2, 2, 2), (0.5, 0.5, 0), (0, 0, 1)]
    plotter.camera.focal_point = (0.5, 0.5, 0.5)

    # change the camera to be looking down the y axis and be head on, no rotation
    # it should be located at 0.5, -1, 0.5
    plotter.camera_position = [(0.5, -3, 0.5), (0.5, 0.5, 0.5), (0, 0, 1)]

    # lock the camera rotation so it can only rotate around the z axis
    plotter.camera_set = True
    plotter.camera_set_key = "c"


    # Display the plotter window
    plotter.show()

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

def load_from_pt_tensor(input_file):
  with open(input_file, "rb") as f:
    pt_tensor = torch.load(f)
  return pt_tensor

def extract_samples(pt_tensor):
  samples = pt_tensor[:, :, 0:3]
  sdf_values = pt_tensor[:, :, 3]
  # convert to numpy
  samples = samples.numpy()
  sdf_values = sdf_values.numpy()
  return samples, sdf_values

def plot(folder_path):
  # get the name of the sim folder
  sim_folder = pathlib.Path(folder_path).stem
  sim_files = [f for f in pathlib.Path(folder_path).iterdir()]
  # get the pickle file
  pickle_file = [f for f in sim_files if f.suffix == ".pkl"][0]
  pt_file = [f for f in sim_files if f.suffix == ".pt"][0]

  mesh_sequence = load_mesh_sequence(pickle_file)
  pt_tensor = load_from_pt_tensor(pt_file)
  samples, sdf_values = extract_samples(pt_tensor)
  plot_mesh_sequence(mesh_sequence, samples, sdf_values)



def main():
  folder_path = sys.argv[1]
  plot(folder_path)



if __name__ == "__main__":
  main()


