import pathlib
import pymeshlab
import tqdm
import pickle
import unpack_pkl


def simplify_mesh(mesh_file, output_file, volume_loss_threshold=0.1):
   # we want to simplify the mesh, while preserving the volume and topology
  ms = pymeshlab.MeshSet()
  ms.load_new_mesh(str(mesh_file))
  # ms.filters_available()
  vert_count = ms.current_mesh().vertex_number()
  face_count = ms.current_mesh().face_number()
  ms.meshing_decimation_quadric_edge_collapse(
    targetperc = 0.1,
    optimalplacement=True,
    preserveboundary=True,
    preservetopology=True,
    autoclean=True,
    planarweight=0.1,
    preservenormal=True,
    boundaryweight=float('inf'),
  )

  # print out % reduction in vertices and faces
  new_vert_count = ms.current_mesh().vertex_number()
  new_face_count = ms.current_mesh().face_number()
  # print(f"Results: V: {vert_count} -> {new_vert_count} ({100*(1-new_vert_count/vert_count):.2f}%) F: {face_count} -> {new_face_count} ({100*(1-new_face_count/face_count):.2f}%)")
  ms.save_current_mesh(str(output_file))
  report = {
    'vert_count': vert_count,
    'face_count': face_count,
    'new_vert_count': new_vert_count,
    'new_face_count': new_face_count,
  }
  return report


def write_to_pickle(mesh_sequence, output_file):
  output_folder = pathlib.Path(output_file).parent
  output_folder.mkdir(exist_ok=True)
  with open(output_file, "wb") as f:
    pickle.dump(mesh_sequence, f)



def read_from_pickle(pickle_file):
  with open(pickle_file, "rb") as f:
    mesh_sequence = pickle.load(f)
  return mesh_sequence

def simplify_meshes(input_folder, output_folder):
  obj_files = [f for f in pathlib.Path(input_folder).iterdir() if f.suffix == ".obj"]
  output_obj_files = []
  if not output_folder.exists():
    output_folder.mkdir(parents=True)
  with tqdm.tqdm(total=len(obj_files), desc="Simplifying Meshes") as pbar:
    for obj_file in obj_files:
      output_file_name = output_folder / obj_file.name
      report = simplify_mesh(obj_file, output_file_name)
      # update the progress bar , with the number of vertices and faces
      # and the percent reduction in vertices and faces in the following format:
      # V: 1000 -> 900 (10.00%) F: 2000 -> 1800 (10.00%)
      v_ratio = 100*(1-report['new_vert_count']/report['vert_count'])
      f_ratio = 100*(1-report['new_face_count']/report['face_count'])
      pbar.set_postfix_str(f"V: {report['vert_count']} -> {report['new_vert_count']} ({v_ratio:.2f}%) F: {report['face_count']} -> {report['new_face_count']} ({f_ratio:.2f}%)")
      pbar.update(1)
      output_obj_files.append(output_file_name)
  return output_obj_files

def unpack_and_simplify(input_sim_folder):
  # given an input
  # we need: input pkl file, obj file to unpack to, folder name to put simplified meshes, and then the output pkl file
  input_folder = pathlib.Path(input_sim_folder)
  folder_files = [f for f in input_folder.iterdir()]
  # pkl_file = [f for f in folder_files if f.suffix == ".pkl"][0]
  pkl_file = input_folder / (input_folder.stem + ".pkl")
  temp_obj_folder = input_folder / "obj_frames"
  temp_simplified_folder = input_folder / "obj_frames_simplified"
  output_pkl_file = input_folder / (input_folder.stem + "_simplified.pkl")

  # print the input and out file
  print(f"Input: {pkl_file}")
  print(f"Output: {output_pkl_file}")

  # if the temp obj folders already exist, clear them out, otherwise create them
  if temp_obj_folder.exists():
    for f in temp_obj_folder.iterdir():
      f.unlink()
  else:
    temp_obj_folder.mkdir()
  
  if temp_simplified_folder.exists():
    for f in temp_simplified_folder.iterdir():
      f.unlink()
  else:
    temp_simplified_folder.mkdir()
  
  # unpack the pkl file to obj files
  unpack_pkl.unpack_pkl(pkl_file, temp_obj_folder)
  # simplify the obj files
  simplified_obj_files = simplify_meshes(temp_obj_folder, temp_simplified_folder)
  # pack the simplified obj files into a pkl file
  unpack_pkl.pack_pkl(temp_simplified_folder, output_pkl_file)

  # remove the temp obj folders
  for f in temp_obj_folder.iterdir():
    f.unlink()
  temp_obj_folder.rmdir()
  for f in temp_simplified_folder.iterdir():
    f.unlink()
  temp_simplified_folder.rmdir()

  




def main():
  folder_path = '/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/meshes/tank_2d_motion_2024-11-02_19-31-25'
  unpack_and_simplify(folder_path)

if __name__ == "__main__":
  main()