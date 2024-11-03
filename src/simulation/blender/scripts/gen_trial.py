import sys
# import tank_2d_motion
# import blend_to_mesh
import simplify_meshes
import sampler
import pathlib
import sys
import subprocess
from datetime import datetime
import time
import zipfile
from tqdm import tqdm
import shutil


# run with blender command 
# blender_command = 'blender --background --python [PYTHON_SCRIPT]'
tank_2d_motion = 'tank_2d_motion.py'
blend_to_mesh = 'blend_to_mesh.py'

REDUCE_MESH = False
ZIP_RESULTS = True

def gen_trial():

  data_folder = pathlib.Path("data")
  # create if it does not exist
  data_folder.mkdir(exist_ok=True)

  trial_name = f"tank_2d_motion_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
  # trial_name = 'tank_2d_motion_2024-11-02_21-27-14'

  # get the cwd 
  cwd = pathlib.Path.cwd()
  output_folder = cwd / data_folder / trial_name
  

  # # Create and bake blender Scene, run from command line

  # bake_command = f'blender --background --python {tank_2d_motion} -- --output_folder "{output_folder}"'
  # subprocess.run(bake_command, shell=True)


  # # Convert the blender scene to a mesh sequence and store as a pickle file
  # save_command = f'blender --background --python {blend_to_mesh} -- "{output_folder}"'
  # subprocess.run(save_command, shell=True)

  # First command to bake the Blender scene
  bake_command = [
      "blender",
      "--background",
      "--python", tank_2d_motion,
      "--",
      "--output_folder", output_folder
  ]
  subprocess.run(bake_command, shell=False)

  # Second command to convert the Blender scene to a mesh sequence
  save_command = [
      "blender",
      "--background",
      "--python", blend_to_mesh,
      "--",
      output_folder  # Pass the output folder as a positional argument
  ]
  subprocess.run(save_command, shell=False)
  
  pickle_file_path = output_folder / f"{trial_name}.pkl"

  if REDUCE_MESH:
    # Reduce the number of vertices in the mesh sequence
    simplify_meshes.unpack_and_simplify(output_folder)
    # remove the original pickle file that is not called _simplified.pkl
    for f in output_folder.iterdir():
      if f.suffix == ".pkl" and not "_simplified" in f.stem:
        f.unlink()
    print(f"Removed the original pickle file: {output_folder}")
    pickle_file_path = output_folder / f"{trial_name}_simplified.pkl"
    # Gather uniform random sdf samples from the mesh sequence
  

  # NOTE: TEMPORARY HACK
  # there seems to be some conflict that causes a segmentation fault when running the sampler as a module
  # the segmentation fault also gives a warning about 
  # "UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown warnings.warn('resource_tracker: There appear to be %d '"
  # while having simplify_meshes as a module
  # The failure occurs when torch is used in sampler.py
  # and simplify_meshes.py has pymeshlab imported as a dependency
  # my guess is that its a conflict between something in pymeshlab and torch, maybe numpy
  sample_command = [
      "python",
      "sampler.py",
      str(pickle_file_path),
      "1000"
  ]

  subprocess.run(sample_command, shell=False)



  if ZIP_RESULTS:
      # zip everything up such that , the contents of the zip file
      # go into a zip file with the same name as the folder
      # and then that zip file goes into the same output folder
      zip_file = output_folder.with_suffix(".zip")
      # tqdm progress bar
      # recursive zip for folder tree
      with zipfile.ZipFile(zip_file, 'w') as z:
        # tqdm progress bar
        for f in output_folder.rglob("*"):
          z.write(f, f.relative_to(output_folder))
      

      # recursive force remove the output folder
      shutil.rmtree(output_folder)




def main():
  start_time = time.time()
  gen_trial()
  elapsed_time = time.time() - start_time
  # convert to hours, minutes, seconds
  h = elapsed_time // 3600
  m = (elapsed_time % 3600) // 60
  s = elapsed_time % 60
  print(f"Elapsed Time: {h:.0f}h {m:.0f}m {s:.2f}s")

if __name__ == "__main__":
  main()