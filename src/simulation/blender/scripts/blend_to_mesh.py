


import bpy
import pickle
import time
import pathlib
import os
import bmesh
import sys
import subprocess
import json


print(f'Python Path: {sys.executable}')

requirements_path = '/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/requirements.txt'
def install_package_requirements(requirements_path):
   subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
install_package_requirements(requirements_path)
    
if not bpy.context.preferences.addons.get('flip_fluids_addon'):
    bpy.ops.preferences.addon_enable(module='flip_fluids_addon')

import numpy as np


# MAKE SURE TO RUN WITH `blender --background --python blend_to_mesh.py`

import bpy
import bmesh
import numpy as np
from tqdm import tqdm
import argparse

def parse_args():
  argv = sys.argv
  if "--" in argv:
    argv = argv[argv.index("--") + 1:]
  else:
    argv = []
  parser = argparse.ArgumentParser(description="Simulate fluid tank")

  # get the cwd
  cwd = os.getcwd()

  relative_path = os.path.join(cwd, "data")
  parser.add_argument("--input_folder", 
                      type=str, 
                      help="input folder with the simulation data and blender file",
                      )
  args = parser.parse_args(argv)
  return args
args = parse_args()

def get_mesh_sequence(
      config,
      use_first_frame_bounding_box=True):
    """
    Extracts and transforms the fluid mesh from each frame in Blender,
    ensuring it fits within the -1 to 1 cube uniformly across all frames
    based on the fluid's bounding box from the first frame.

    Parameters:
    - use_first_frame_bounding_box (bool): If True, use the first frame's bounding box for scaling.

    Returns:
    - mesh_sequence (list): A list of dictionaries containing transformed vertices and faces.
    """
    mesh_sequence = []
    last_frame = bpy.context.scene.frame_end

    

    # Define object names for easy reference
    # fluid_surface_name = 'fluid_surface'
    # fluid_object_name = 'Fluid'
    # container_name = 'Container'
    # fluid_domain_name = 'Domain'
    fluid_surface_name = config['fluid_surface_name']
    fluid_object_name = config['fluid_object_name']
    container_name = config['container_object_name']
    fluid_domain_name = config['domain_object_name']

    # Ensure the objects exist in the scene
    if fluid_surface_name not in bpy.data.objects:
        raise ValueError(f"Object '{fluid_surface_name}' not found in the scene.")
    if container_name not in bpy.data.objects:
        raise ValueError(f"Object '{container_name}' not found in the scene.")
    if fluid_domain_name not in bpy.data.objects:
        raise ValueError(f"Object '{fluid_domain_name}' not found in the scene.")
    if fluid_object_name not in bpy.data.objects:
        raise ValueError(f"Object '{fluid_object_name}' not found in the scene.")

    fluid_surface = bpy.data.objects[fluid_surface_name]
    container = bpy.data.objects[container_name]
    fluid_domain = bpy.data.objects[fluid_domain_name]
    fluid_object = bpy.data.objects[fluid_object_name]


    # Get the instaneous velocities of the container at each frame before performing any transformations
    fps = bpy.context.scene.render.fps
    delta_t = 1 / fps
    container_positions = []
    velocities = []
    for i in tqdm(range(1, last_frame + 1), desc="Extracting Container Velocities"):
        bpy.context.scene.frame_set(i)
        container_positions.append(np.array(container.location))
        if i == 1:
            velocities.append(np.zeros(3))
        else:
            distance = container_positions[-1] - container_positions[-2]
            velocities.append(distance / delta_t)
    
    # calculate instantaneous acceleration / impulse
    impulses = []
    for i in tqdm(range(1, len(velocities)), desc="Extracting Container Impulses"):
        impulses.append((velocities[i] - velocities[i-1]))
    # add the last acceleration as the last value
    impulses.append(impulses[-1])
       

    # Step 1: Compute Scaling Factor from the First Frame's Bounding Box
    bpy.context.scene.frame_set(1)  # Set to first frame
    bm_initial = bmesh.new()
    bm_initial.from_mesh(fluid_surface.data)
    bm_initial.verts.ensure_lookup_table()
    bm_initial.faces.ensure_lookup_table()

    # Extract all vertex coordinates in world space
    verts_world_initial = np.array([fluid_surface.matrix_world @ v.co for v in bm_initial.verts], dtype=np.float32)
    # verts_world_initial = np.array([container.matrix_world @ v.co for v in bm_initial.verts], dtype=np.float32)
    # apply 

    # Compute the axis-aligned bounding box (AABB)
    # min_coords = verts_world_initial.min(axis=0)
    # max_coords = verts_world_initial.max(axis=0)
    # bounding_box_dimensions = max_coords - min_coords

    # Determine the maximum dimension to compute scaling factor
    # max_dimension = bounding_box_dimensions.max()
    # if max_dimension <= 0:
        # raise ValueError("Invalid bounding box dimensions. The maximum dimension is zero or negative.")

    # Compute scaling factor to fit the bounding box within -1 to 1 cube (span of 2 units)
    # scale_factor = 1.0 / max_dimension
    # print(f"Computed scaling factor based on first frame: {scale_factor}")

    # in order to get the real scale factor, we need to use the absolute world dimensions of the container
    container_dimensions = container.dimensions
    domain_dimensions = fluid_domain.dimensions
    fluid_object_dimensions = fluid_object.dimensions

    # use the fluid domain vertical dimension, just for the vertical component 
    # print(f"Container dimensions: {container_dimensions}")
    # print(f"Domain dimensions: {domain_dimensions}")
    # print(f"Fluid object dimensions: {fluid_object_dimensions}")
    container_dimensions[2] = domain_dimensions[2]
    #use the fluid object x dimension for the x component
    container_dimensions[0] = fluid_object_dimensions[0]



    # we want the maximum dimension of the internal dimensions
    max_internal_dimension = max(container_dimensions)
    if max_internal_dimension <= 0:
        raise ValueError("Invalid internal container dimensions. The maximum dimension is zero or negative.")
    scale_factor = 1.0 / 12.0 #max_internal_dimension

    # Free the initial BMesh
    bm_initial.free()

    # Step 2: Retrieve Container's Center in World Coordinates

    fluid_object_translation = np.array(fluid_object.location)
    fluid_surface_translation = np.array(fluid_surface.location)


    # Step 3: Iterate Through All Frames and Transform Meshes
    for i in tqdm(range(1, last_frame + 1), desc="Extracting Meshes from Frames"):
        


        # Set the current frame
        bpy.context.scene.frame_set(i)

        container_world_matrix = container.matrix_world.copy()
        container_center_world = np.array(container_world_matrix.translation)
        # Create a new BMesh from the fluid surface's mesh data
        bm = bmesh.new()
        bm.from_mesh(fluid_surface.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        

        # Extract all vertex coordinates in world space
        verts_world = np.array([fluid_surface.matrix_world @ v.co for v in bm.verts], dtype=np.float32)



        # Translate the vertices so that the container's center aligns with the origin
        # only use the x component
        #subtract the fluid surface vertical translation
        verts_translated = verts_world
        verts_translated[:, 0] = verts_translated[:, 0] - container_center_world[0]
        verts_translated[:, 2] = verts_translated[:, 2] - fluid_surface_translation[2]


        # Scale the vertices uniformly using the precomputed scaling factor
        verts_scaled = verts_translated * scale_factor

        # scale the y component to fit the unit cube
        # we take the y dimension of the fluid object
        # divide it by our scale factor
        # then we do 1 / that value to get the scale factor
        # y_scale = 1 / (fluid_object_dimensions[1] * scale_factor)
        # verts_scaled[:, 1] = verts_scaled[:, 1] * y_scale


        # add 0.5 to x and vertical component to move to middle of unit cube
        verts_scaled[:, 0] = verts_scaled[:, 0] + 0.5
        verts_scaled[:, 1] = verts_scaled[:, 1] + 0.5

        # Swap Y and Z coordinates to correct orientation
        # This assumes that the issue is specifically a Y-Z swap; adjust if necessary
        # verts_scaled[:, [1, 2]] = verts_scaled[:, [2, 1]]

        # Convert the transformed vertices to float16 for compactness
        transformed_vertices_np = verts_scaled.astype(np.float16)


        # Extract face indices
        faces = []
        for f in bm.faces:
            face_indices = [v.index for v in f.verts]
            faces.append(face_indices)
        faces_np = np.array(faces, dtype=np.uint32)

        # Free the BMesh to free memory
        bm.free()

        # Store the transformed mesh
        mesh = {
            'vertices': transformed_vertices_np,  # Transformed vertices with Y and Z swapped
            'faces': faces_np,                    # Face indices
            'velocity': velocities[i-1],
            'impulse': impulses[i-1]
        }
        mesh_sequence.append(mesh)
    # print velocities
    # print(f"Velocities: {velocities}")
    return mesh_sequence





def write_obj_frames(mesh_sequence, output_folder):
  output_folder.mkdir(exist_ok=True)
  with tqdm(total=len(mesh_sequence), desc="Writing Frame Meshes to OBJ Files") as pbar:
    for i, mesh in enumerate(mesh_sequence):
      # save as .obj file
      obj_path = output_folder / f"{i}.obj"
      with open(obj_path, "w") as f:
        f.write(f"o {i}\n")
        for v in mesh['vertices']:
          f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in mesh['faces']:
          f.write(f"f {' '.join(str(v+1) for v in face)}\n")
        
      # simplify the mesh
      pbar.update(1)

def write_to_pickle(mesh_sequence, output_file):
  output_folder = pathlib.Path(output_file).parent
  output_folder.mkdir(exist_ok=True)
  with open(output_file, "wb") as f:
    pickle.dump(mesh_sequence, f)

def read_from_pickle(pickle_file):
  with open(pickle_file, "rb") as f:
    mesh_sequence = pickle.load(f)
  return mesh_sequence



def mesh_to_obj(mesh, output_file, name="frame"):
  with open(output_file, "w") as f:
    f.write(f"o {name}\n")
    for v in mesh['vertices']:
      f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for face in mesh['faces']:
      f.write(f"f {' '.join(str(v+1) for v in face)}\n")

def load_config_file(config_file_path):
  with open(config_file_path, "r", encoding="utf-8") as f:

    config = json.load(f)
  return config

def blend_to_mesh(blend_file_path, output_folder, config_file_path):
   
    bpy.ops.wm.open_mainfile(filepath=str(blend_file_path))

    bpy.context.scene.frame_set(0)
    bpy.ops.object.mode_set(mode='OBJECT')

    config = load_config_file(config_file_path)

    mesh_sequence = get_mesh_sequence(config)




    # create a new folder called frames in the output folder
    mesh_folder = output_folder / "frames"
    mesh_folder.mkdir(exist_ok=True)
    # create a new folder for each frame called i
    # save the mesh as an obj file in that folder
    # save the velocity and impulse as as a json file in that folder
    for i, mesh in enumerate(mesh_sequence):
        frame_folder = mesh_folder / str(i)
        frame_folder.mkdir(exist_ok=True)
        # save the mesh as an obj file
        obj_path = frame_folder / f"{i}.obj"
        mesh_to_obj(mesh, obj_path, name=f"frame_{i}")
        # save mesh as pkl
        mesh_output_file = frame_folder / f"{i}.pkl"
        with open(mesh_output_file, "wb") as f:
            pickle.dump(mesh, f)
    
    # get the velocity and impulse for each frame and save as pkl file
    velocities = [mesh['velocity'] for mesh in mesh_sequence]
    impulses = [mesh['impulse'] for mesh in mesh_sequence]
    dynamics = {
        'velocities': velocities,
        'impulses': impulses
    }
    velocities_output_file = output_folder / "dynamics.pkl"
    with open(velocities_output_file, "wb") as f:
        pickle.dump(dynamics, f)

    # save all the meshes as obj files in the output folder
    # each mesh should just have its name as the frame number
    # output_folder.mkdir(exist_ok=True)
    blend_file = pathlib.Path(blend_file_path)
    output_file = blend_file.with_name(f"mesh_sequence.pkl")
    write_to_pickle(mesh_sequence, output_file)
    # write_obj_frames(mesh_sequence, mesh_folder)
    # print(f"Saved mesh sequence to {output_file}")
    # read from pickle
    # mesh_sequence_1 = read_from_pickle(output_file)

def main():
    
    addons = {a.__name__:a for a in bpy.utils._addon_utils.modules()}
    flip_name = "flip_fluids_addon"
    if flip_name not in bpy.context.preferences.addons.keys():
      print("FLIP Fluids addon is not enabled. Install and enable the addon at: https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Addon-Installation-and-Uninstallation")
      print("Current Available addons:")
      addons.keys()
      return
    
    input_folder = pathlib.Path(args.input_folder)
    config_file_path = input_folder / "trial_config.json"
    # config_file_path = "/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/test.json"
    output_folder = input_folder

    # output_folder_name / output_folder_name.blend
    blend_file = input_folder / f"{input_folder.stem}.blend"
    blend_to_mesh(blend_file, output_folder, config_file_path)

    

          
if __name__ == "__main__":
    main()

