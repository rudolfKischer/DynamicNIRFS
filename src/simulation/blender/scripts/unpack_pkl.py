import os
import pathlib
import sys
from tqdm import tqdm
import pickle
import trimesh

def write_obj_frames(mesh_sequence, output_folder):
  output_folder = pathlib.Path(output_folder)
  output_folder.mkdir(exist_ok=True)
  file_names = []
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
      file_names.append(obj_path)
        
      # simplify the mesh
      pbar.update(1)
  return file_names

def load_mesh_sequence_from_objs(input_folder):
  obj_files = [f for f in pathlib.Path(input_folder).iterdir() if f.suffix == ".obj"]
  # FORMAT: THE LEADING UNDERSCORE SEPERATED WORD IS THE NUMBER OF THE FRAME
  # SORT BY FRAME NUMBER
  # handle the case where there is no _ in the file name
  obj_files.sort(key=lambda x: int(x.stem.split("_")[0]))

  mesh_sequence = []
  for obj_file in obj_files:
    mesh = trimesh.load(obj_file)
    vertices = mesh.vertices
    faces = mesh.faces
    mesh_sequence.append({
      'vertices': vertices,
      'faces': faces
    })
  return mesh_sequence

def read_from_pickle(pickle_file):
  with open(pickle_file, "rb") as f:
    mesh_sequence = pickle.load(f)
  return mesh_sequence

def write_to_pickle(mesh_sequence, output_file):
  output_folder = pathlib.Path(output_file).parent
  output_folder.mkdir(exist_ok=True)
  with open(output_file, "wb") as f:
    pickle.dump(mesh_sequence, f)

def unpack_pkl(pickle_file, output_folder):
  # read the mesh sequence from the pickle file
  mesh_sequence = read_from_pickle(pickle_file)
  # write the mesh sequence to obj files
  write_obj_frames(mesh_sequence, output_folder)

def pack_pkl(obj_folder, output_file):
  # load the mesh sequence from the obj files
  mesh_sequence = load_mesh_sequence_from_objs(obj_folder)
  # write the mesh sequence to a pickle file
  write_to_pickle(mesh_sequence, output_file)

def main():
  # get the folder path
  # input_file = sys.argv[1]
  input_file = '/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/meshes/tank_2d_motion_2024-11-02_13-48-16/tank_2d_motion_2024-11-02_13-48-16.pkl'
  input_file_path = pathlib.Path(sys.argv[1])
  output_folder = input_file_path.parent / "obj_frames"
  # output_folder = '/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/meshes/tank_2d_motion_2024-11-02_13-48-16/obj_frames'

  unpack_pkl(input_file, output_folder)

if __name__ == "__main__":
  main()