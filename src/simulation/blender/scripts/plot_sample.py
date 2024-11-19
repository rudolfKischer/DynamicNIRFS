import trimesh
import pyvista as pv
import numpy as np
import pickle
import torch
import pathlib
import sys
import matplotlib.pyplot as plt
import io



import matplotlib.pyplot as plt
from PIL import Image


def load_mesh_sequence(pickle_file_path):
  with open(pickle_file_path, "rb") as f:
    mesh_sequence = pickle.load(f)
  return mesh_sequence

MOTION_1D = True

np.set_printoptions(precision=2)

def plot_mesh_sequence(mesh_sequence, sample_points=None, sample_sdf_values=None):
    """
    Plots a sequence of meshes with an interactive time slider and a velocity subplot.

    Parameters:
    - mesh_sequence: List of dictionaries containing mesh data. Each dictionary should have:
        - "vertices": Array of vertex coordinates.
        - "faces": Array of face indices.
        - "velocity": (Optional) Velocity vector for the mesh.
    - sample_points: (Optional) List of point arrays for sampling.
    - sample_sdf_values: (Optional) List of SDF values corresponding to sample points.
    """
    # Initialize the PyVista plotter
    plotter = pv.Plotter()

    # change the size of the plotter window
    plotter.window_size = (1920, 800)

    hide_mesh = False  # Flag to toggle mesh visibility

    # Define highlighted vertices
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

    # Extract velocities from the mesh sequence
    velocities = []
    for mesh in mesh_sequence:
        if "velocity" in mesh:
            velocities.append(mesh["velocity"])
        else:
            velocities.append([0, 0, 0])
    velocities = np.array(velocities)

    impulses = []
    for mesh in mesh_sequence:
        if "impulse" in mesh:
            impulses.append(mesh["impulse"])
        else:
            impulses.append([0, 0, 0])
    impulses = np.array(impulses)

    if impulses.ndim == 1:
        impulses = impulses.reshape(-1, 1)

    # Ensure velocities is a 2D array
    if velocities.ndim == 1:
        velocities = velocities.reshape(-1, 1)
    
    if MOTION_1D:
        velocities = velocities[:, 0]
        impulses = impulses[:, 0]

    # Option 2: Plotting individual velocity components
    fig, ax = plt.subplots(tight_layout=True)
    velocity_line_x, = ax.plot([], [], label="Velocity X")
    velocity_line_y, = ax.plot([], [], label="Velocity Y")
    velocity_line_z, = ax.plot([], [], label="Velocity Z")
    ax.set_xlim(0, len(mesh_sequence))
    ax.set_ylim(np.min(velocities) * 1.5, np.max(velocities) * 1.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Velocity (m/s)')

    # add a horizontal axis line at 0
    ax.axhline(0, color='black', lw=0.1)

    # ax.set_title('Velocity Components Over Time')
    # ax.legend()

    # Convert the Matplotlib figure to a PyVista ChartMPL
    velocity_chart = pv.ChartMPL(fig, size=(0.3, 0.45), loc=(0.65, 0.45))
    velocity_chart.background_color = (1.0, 1.0, 1.0, 0.8)
    plotter.add_chart(velocity_chart)

    # add impulse subplot
    fig2, ax2 = plt.subplots(tight_layout=True)
    impulse_line_x, = ax2.plot([], [], label="impulse X")
    impulse_line_y, = ax2.plot([], [], label="impulse Y")
    impulse_line_z, = ax2.plot([], [], label="impulse Z")
    ax2.set_xlim(0, len(mesh_sequence))
    ax2.set_ylim(np.min(impulses) * 1.5, np.max(impulses) * 1.5)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Impulse (Ns)')

    # add a horizontal axis line at 0
    ax2.axhline(0, color='black', lw=0.1)
    # ax2.set_title('impulse Components Over Time')
    # ax2.legend()

    # Convert the Matplotlib figure to a PyVista ChartMPL
    impulse_chart = pv.ChartMPL(fig2, size=(0.3, 0.45), loc=(0.65, 0.00))
    impulse_chart.background_color = (1.0, 1.0, 1.0, 0.8)
    plotter.add_chart(impulse_chart)


    # Add highlighted vertices to the plotter
    plotter.add_points(highlighted_vertices, color="red", point_size=10, name="highlighted_vertices")

    # Initialize variables to store current mesh and text actors

    mesh_actors = []
    for i, mesh in enumerate(mesh_sequence):
       frame_mesh = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"])
       pv_mesh = pv.wrap(frame_mesh)
       mesh_actor = plotter.add_mesh(pv_mesh, color="lightblue", show_edges=True, name=f"current_mesh_{i}")
       mesh_actors.append(mesh_actor)
       mesh_actor.SetVisibility(False)


    # Function to update the displayed mesh and velocity graph for a specific frame
    current_frame = 0
    prev_mesh_frame = 0
    def update_mesh(frame):
        nonlocal prev_mesh_frame, current_frame


        if frame < len(mesh_sequence):
            mesh = mesh_sequence[frame]
            vertices = mesh["vertices"]
            faces = mesh["faces"]

        #     # Create and add the mesh
            # frame_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            # pv_mesh = pv.wrap(frame_mesh)
            if not hide_mesh:
                # current_mesh_actor = plotter.add_mesh(pv_mesh, color="lightblue", show_edges=True, name="current_mesh")
                mesh_actors[frame].SetVisibility(True)
                if prev_mesh_frame != frame:
                  mesh_actors[prev_mesh_frame].SetVisibility(False)
                prev_mesh_frame = frame
                if hide_mesh:
                  mesh_actors[frame].SetVisibility(False)
                  mesh_actors[prev_mesh_frame].SetVisibility(False)

            # Add sample points if provided
            if sample_points is not None and frame < len(sample_points):
                # dont display the cmap legend
                plotter.add_points(
                    sample_points[frame],
                    scalars=sample_sdf_values[frame],
                    cmap="coolwarm",
                    clim=(-0.1, 0.01),
                    point_size=15,
                    name="sample_points",
                    # render_points_as_spheres=True,
                    show_scalar_bar=False
                )

            # Update and display velocity as text (optional)
            if "velocity" in mesh:
                velocity = velocities[frame]
                velocity_text = f"Velocity: {velocity:.2f}"
                current_text_actor = plotter.add_text(
                    velocity_text,
                    position="upper_right",
                    font_size=10,
                    color="black",
                    name="velocity_text"
                )

            # Update the velocity graph up to the current frame
            if not MOTION_1D:
              velocity_line_x.set_data(range(frame + 1), velocities[:frame + 1, 0])
              velocity_line_y.set_data(range(frame + 1), velocities[:frame + 1, 1])
              velocity_line_z.set_data(range(frame + 1), velocities[:frame + 1, 2])
            else:
              velocity_line_x.set_data(range(frame + 1), velocities[:frame + 1])

            # set x limit to be the current frame
            ax.set_xlim(0, frame)
            # disable the legend
            ax.legend().set_visible(False)
            ax.relim()
            ax.autoscale_view()
            velocity_chart.Update()

            #update the impulse graph
            if not MOTION_1D:
              impulse_line_x.set_data(range(frame + 1), impulses[:frame + 1, 0])
              impulse_line_y.set_data(range(frame + 1), impulses[:frame + 1, 1])
              impulse_line_z.set_data(range(frame + 1), impulses[:frame + 1, 2])
            else:
              impulse_line_x.set_data(range(frame + 1), impulses[:frame + 1])

            # set x limit to be the current frame
            ax2.set_xlim(0, frame)
            # disable the legend
            ax2.legend().set_visible(False)
            ax2.relim()
            ax2.autoscale_view()
            impulse_chart.Update()

        # Render the updated plotter
        plotter.render()

        current_frame = frame

    # Function to handle slider updates
    def slider_callback(value):
        frame = int(value)
        # Ensure frame is within bounds
        frame = max(0, min(frame, len(mesh_sequence) - 1))
        current_frame = frame
        update_mesh(frame)

    # Add a time slider widget without the 'format' parameter
    plotter.add_slider_widget(
        callback=slider_callback,
        rng=[0, len(mesh_sequence) - 1],
        value=0,
        title="Frame",
        pointa=(0.25, 0.1),
        pointb=(0.75, 0.1),
        style='modern',
        interaction_event='always'
        # 'format' parameter removed
    )

    # Function to toggle mesh visibility
    def toggle_mesh():
        nonlocal hide_mesh
        hide_mesh = not hide_mesh

        # go through and hide all meshes
        for i in range(len(mesh_sequence)):
          mesh_actors[i].SetVisibility(False)

        update_mesh(current_frame)

    # Bind the 'm' key to toggle mesh visibility
    plotter.add_key_event("m", toggle_mesh)

    # Add highlighted vertices again to ensure they stay on top
    plotter.add_points(highlighted_vertices, color="black", point_size=10, name="highlighted_vertices_top")

    # Set camera properties
    plotter.camera_position = [(0.5, -3, 0.5), (0.5, 0.5, 0.5), (0, 0, 1)]
    plotter.camera.focal_point = (0.5, 0.5, 0.5)
    plotter.enable_parallel_projection()
    plotter.show_bounds(grid='front', location='outer', all_edges=True)

    # Show the initial frame
    update_mesh(0)

    # Display the plotter window
    plotter.show()

# def plot_mesh_sequence(mesh_sequence, sample_points=None, sample_sdf_values=None):
#     # Initialize the PyVista plotter
#     plotter = pv.Plotter()
#     current_frame = [0]  # Using a list to allow modification inside the event functions

#     highlighted_vertices = np.array([
#         [0.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0],
#         [0.0, 1.0, 0.0],
#         [1.0, 0.0, 1.0],
#         [1.0, 1.0, 0.0],
#         [0.0, 0.0, 1.0],
#         [1.0, 0.0, 0.0],
#         [0.0, 1.0, 1.0]
#     ])

#     velocities = []
#     for mesh in mesh_sequence:
#         if "velocity" in mesh:
#             velocities.append(mesh["velocity"])
#         else:
#             velocities.append([0, 0, 0])
#     velocities = np.array(velocities)

#     # plot the velocity graph
#     velocity_graph_file = plot_velocity_graph(velocities)

#     # negative values only

#     plotter.add_points(highlighted_vertices, color="red", point_size=10)

#     hide_mesh = False

#     velocity_graph_time_window = 30
    
#     # Function to update the displayed mesh for a specific frame
#     def update_mesh(frame):
#         plotter.clear()  # Clear the previous frame
#         if frame < len(mesh_sequence):
#           mesh = mesh_sequence[frame]
#           vertices = mesh["vertices"]
#           faces = mesh["faces"]
#           if "velocity" in mesh:
#              # display the volocity as text in the corner
#               velocity = velocities[frame]
#               # first two decimal places
#               np.set_printoptions(precision=2)
#               plotter.add_text(f"Velocity: {velocity}", position="upper_left")
              
#               velocity_window = velocities[max(0, frame - velocity_graph_time_window):frame]

#               # add a subplot to the plotter in the upper left corner that plots the velocity
#               # x axis is time, y axis is velocity

              

             
#           frame_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
#           pv_mesh = pv.wrap(frame_mesh)
#         if sample_points is not None and frame < len(sample_points):
#             # use the sdf values to color the points
#             plotter.add_points(sample_points[frame], 
#                                scalars=sample_sdf_values[frame], 
#                                cmap="coolwarm",
#                                clim=(-0.1, 0.01), 
#                                point_size=15)
#         if not hide_mesh:
#           plotter.add_mesh(pv_mesh, color="lightblue", show_edges=True)


#         plotter.add_points(highlighted_vertices, color="black", point_size=10)
#         plotter.render()  # Refresh to display the updated mesh

#     # Event handler to go to the next frame
#     def next_frame():
#         if current_frame[0] < len(mesh_sequence) - 1:
#             current_frame[0] += 1
#             update_mesh(current_frame[0])

#     # Event handler to go to the previous frame
#     def prev_frame():
#         if current_frame[0] > 0:
#             current_frame[0] -= 1
#             update_mesh(current_frame[0])
    
#     def toggle_mesh():
#       nonlocal hide_mesh
#       hide_mesh = not hide_mesh
#       update_mesh(current_frame[0])

#     # Bind the arrow keys to the event handlers
#     plotter.add_key_event("Right", next_frame)
#     plotter.add_key_event("Left", prev_frame)
#     plotter.add_key_event("m", toggle_mesh)

#     # Show the initial frame
#     update_mesh(current_frame[0])

#     # plotter.enable_parallel_projection()
#     # plotter.camera_position = [(2, 2, 2), (0.5, 0.5, 0), (0, 0, 1)]
#     plotter.camera.focal_point = (0.5, 0.5, 0.5)

#     # change the camera to be looking down the y axis and be head on, no rotation
#     # it should be located at 0.5, -1, 0.5
#     plotter.camera_position = [(0.5, -3, 0.5), (0.5, 0.5, 0.5), (0, 0, 1)]

#     # lock the camera rotation so it can only rotate around the z axis
#     plotter.camera_set = True
#     plotter.camera_set_key = "c"


#     # Display the plotter window
#     plotter.show()

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

def get_pkl_and_pt_files(folder_path):
  sim_files = [f for f in pathlib.Path(folder_path).iterdir()]
  # get the pickle file
  pickle_file = [f for f in sim_files if f.suffix == ".pkl"][0]
  pt_file = [f for f in sim_files if f.suffix == ".pt"]
  if len(pt_file) == 0:
     pt_file = None
  else:
    pt_file = pt_file[0]
  return pickle_file, pt_file

def plot(pickle_file, pt_file, negative_sdf_only=True):
  mesh_sequence = load_mesh_sequence(pickle_file)
  if pt_file is None:
    plot_mesh_sequence(mesh_sequence)
    return
  pt_tensor = load_from_pt_tensor(pt_file)
  print(f'pt_tensor shape: {pt_tensor.shape}')
  samples, sdf_values = extract_samples(pt_tensor)
  # if negative_sdf_only:
     # set all positive values, put their positions at 0,0,0
    # mask = sdf_values >= 0
    # samples[mask] = 0
  plot_mesh_sequence(mesh_sequence, samples, sdf_values)
   
def main():
  # folder_path = sys.argv[1]
  # pickle_file, pt_file = get_pkl_and_pt_files(folder_path)
  # # plot(folder_path)
  # plot(pickle_file, pt_file)

  # # single_sample test
  # pickle_file = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/stanford-bunny.pkl"
  # # # pt_file = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/stanford-bunny_samples_10000.pt"
  # pt_file = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/stanford-bunny_samples_1000.pt"
  # pt_file = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/stanford-bunny_samples_1000.pt"
  # pt_file = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/modeling/models/deep_sdf_decoder_eval.pt"
  # plot(pickle_file, pt_file)


  # single td tank motion test
  # folder_path = '/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/data/tank_2d_motion_2024-11-02_23-28-02'
  # pickle_file, pt_file = get_pkl_and_pt_files(folder_path)
  # plot(pickle_file, pt_file)


  # multi shape test
  # pt_file = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/modeling/models/autodecoder_tank_2d_motion_eval.pt"
  # pickle_file = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/data/tank_2d_motion_2024-11-02_23-28-02/tank_2d_motion_2024-11-02_23-28-02_simplified.pkl"
  # plot(pickle_file, pt_file)

  # test code reconstruction
  pt_file = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/modeling/test_eval.pt"
  pickle_file = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/data/tank_2d_motion_2024-11-02_23-28-02/tank_2d_motion_2024-11-02_23-28-02_simplified.pkl"
  plot(pickle_file, pt_file)




if __name__ == "__main__":
  main()


