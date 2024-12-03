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
import json
import random


# run with blender command 
# blender_command = 'blender --background --python [PYTHON_SCRIPT]'
tank_2d_motion = 'tank_2d_motion.py'
blend_to_mesh = 'blend_to_mesh.py'


ZIP_RESULTS = True

def write_to_json(data, file_path):
  with open(file_path, "w") as f:
    json.dump(data, f)

def load_json(json_file):
  with open(json_file, "r") as f:
    data = json.load(f)
  return data

def gen_trial(trial_name, phases, data_folder, config):

  
  # create if it does not exist
  data_folder.mkdir(exist_ok=True)

  
  # trial_name = 'tank_2d_motion_2024-11-02_21-27-14'

  # get the cwd 
  cwd = pathlib.Path.cwd()
  output_folder = cwd / data_folder / trial_name
  
  # create the trial folder if it does not exist
  output_folder.mkdir(exist_ok=True)

  trial_folder = output_folder

  # write the config to a json file called trial_config.json
  config_file = trial_folder / "trial_config.json"
  write_to_json(config, config_file)

  

  # # Create and bake blender Scene, run from command line

  # bake_command = f'blender --background --python {tank_2d_motion} -- --output_folder "{output_folder}"'
  # subprocess.run(bake_command, shell=True)


  # # Convert the blender scene to a mesh sequence and store as a pickle file
  # save_command = f'blender --background --python {blend_to_mesh} -- "{output_folder}"'
  # subprocess.run(save_command, shell=True)

  REDUCE_MESH = config['mesh_simplification']

  # First command to bake the Blender scene
  if 0 in phases:
    bake_command = [
        "blender",
        "--background",
        "--python", tank_2d_motion,
        "--",
        "--output_folder", trial_folder,
        "--config_file", config_file
    ]
    subprocess.run(bake_command, shell=False)

  # Second command to convert the Blender scene to a mesh sequence
  if 1 in phases:
    save_command = [
        "blender",
        "--background",
        "--python", blend_to_mesh,
        "--",
        "--input_folder", trial_folder,
    ]
    subprocess.run(save_command, shell=False)

  if REDUCE_MESH and 2 in phases:
    # Reduce the number of vertices in the mesh sequence
    simplify_command = [
        "python",
        "simplify_meshes.py",
        str(trial_folder)
    ]
    subprocess.run(simplify_command, shell=False)
  

  # NOTE: TEMPORARY HACK
  # there seems to be some conflict that causes a segmentation fault when running the sampler as a module
  # the segmentation fault also gives a warning about 
  # "UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown warnings.warn('resource_tracker: There appear to be %d '"
  # while having simplify_meshes as a module
  # The failure occurs when torch is used in sampler.py
  # and simplify_meshes.py has pymeshlab imported as a dependency
  # my guess is that its a conflict between something in pymeshlab and torch, maybe numpy
  if 3 in phases:
    sample_command = [
        "python",
        "sampler.py",
        str(trial_folder)
    ]

    subprocess.run(sample_command, shell=False)


  if ZIP_RESULTS and 4 in phases:
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

def gen_impulses(config_template_path):

  # we want to generate a set of random impulses that will be used to perturb the fluid
  # the container, fluid, and fluid domain all have certain dimensions
  # the container can move throughout the trial based on the impulses
  # but we need to make sure the container never gets moved outside the fluid domain
  # this means we need to generate a that will keep the container within the fluid domain
  config_template = load_json(config_template_path)
  # for now we will start with a single initial impulse in the x direction
  fps = config_template['fps']
  num_frames = config_template['num_frames']
  domain_dimensions = config_template['domain_dimensions']
  fluid_dimensions = config_template['fluid_dimensions']
  # container dimensions = fluid_dimensions but height is domain height
  container_dimensions = fluid_dimensions.copy()
  container_dimensions[2] = domain_dimensions[2]

  total_time = num_frames / fps
  # the container is centered at the origin
  # and the domain is centered at the origin
  # the max distance we can travel, in one direction
  # (domain_dimensions / 2) - (container_dimensions / 2)
  abs_x_max_container_pos = (domain_dimensions[0] / 2) - (container_dimensions[0] / 2)

  max_x_init_velocity = abs_x_max_container_pos / total_time

  # we will generate a random impulse in the x direction between -max_init_velocity and max_init_velocity

  mean_impulse_magnitude = config_template['mean_impulse_magnitude']
  std_dev_impulse_magnitude = config_template['std_dev_impulse_magnitude']
  impulse = random.gauss(mean_impulse_magnitude, std_dev_impulse_magnitude)
  impulse = min(impulse, max_x_init_velocity)
  impulse = max(impulse, -max_x_init_velocity)

  init_impulse = [impulse, 0, 0]

  frame_impulses = [
    [0, init_impulse],
    [int(num_frames // 2), [-impulse, 0, 0]] # stopping impulse
  ]

  return frame_impulses







def gen_trials(config_template, 
               trials_folder,
               phases):

  # load the config template
  config = load_json(config_template)
  num_trials = config['num_trials']
  data_folder = pathlib.Path(trials_folder)
  # create log file
  log_file = data_folder / "log.txt"
  # create the log file if it does not exist
  if not log_file.exists():
    log_file.touch()
  with open(log_file, "w") as f:
    f.write(f"Generating {num_trials} Trials\n")
    f.write(f"Phases: {phases}\n")
    f.write(f"Config: {config}\n")
    f.write(f"Config Template: {config_template}\n")
    f.write(f"Trials Folder: {trials_folder}\n")



  for i in tqdm(range(num_trials), desc="Generating Trials"):
    trial_impulses = gen_impulses(config_template)
    config['frame_impulses'] = trial_impulses
    trial_name = f"trial_{i}"
    # redirect stdout to log file for the whole trial generation

    gen_trial(trial_name, phases, data_folder, config)
    # redirect stdout back to console
  
  # save the config template to the trials folder
  config_file = data_folder / "trials_config.json"
  write_to_json(config, config_file)
  



# def main():
#   start_time = time.time()
#   phases = [
#     0, 
#     1, 
#     2,
#     3,
#     # 4,
#   ]
#   # trial_name = f"tank_2d_motion_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
#   trial_name = 'tank_2d_motion_2024-11-11_11-23-36'
#   data_folder = pathlib.Path("trials")
#   trial_config_template_path = pathlib.Path("/Users/rudolfkischer/MCGILL/FALL2024/Comp 400/repositories/DynamicNIRFS/src/simulation/blender/scripts/trials/trial_config_1.json")
#   trial_config = load_json(trial_config_template_path)
#   gen_trial(trial_name, phases, data_folder, trial_config)
#   elapsed_time = time.time() - start_time
#   # convert to hours, minutes, seconds
#   h = elapsed_time // 3600
#   m = (elapsed_time % 3600) // 60
#   s = elapsed_time % 60
#   print(f"Elapsed Time: {h:.0f}h {m:.0f}m {s:.2f}s")

def main():

  start_time = time.time()
  phases = [
    0, 
    1, 
    2,
    3,
    # 4,
  ]
  
  trials_folder = pathlib.Path("trials")
  cur_trials_folder = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
  trials_folder = trials_folder / cur_trials_folder
  # create the trials folder
  trials_folder.mkdir(exist_ok=True)

  trial_config_template_path = "trials/trial_config_1.json"
  gen_trials(trial_config_template_path, trials_folder, phases)

  elapsed_time = time.time() - start_time
  # convert to hours, minutes, seconds
  h = elapsed_time // 3600
  m = (elapsed_time % 3600) // 60
  s = elapsed_time % 60
  print(f"Elapsed Time: {h:.0f}h {m:.0f}m {s:.2f}s")



if __name__ == "__main__":
  main()