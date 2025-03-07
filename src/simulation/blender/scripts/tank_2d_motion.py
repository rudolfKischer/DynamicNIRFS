import os
import sys
package_dir = os.path.dirname(os.path.realpath(__file__))
if package_dir not in sys.path:
    sys.path.append(package_dir)
import io
import random
import mathutils
import bpy
import argparse
import re
import threading
import time
import json
from datetime import datetime

import bmesh
import pathlib

bpy.app.debug = False
bpy.app.debug_wm = False
bpy.app.debug_events = False




from logs import logger, BlenderLogInterCeptor

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
  parser.add_argument("--output_folder", 
                      type=str, 
                      help="Output folder for simulation data",
                      default=relative_path,
                      )
  parser.add_argument("--config_file",
                      type=str,
                      help="Path to the configuration file"
                      )
  args = parser.parse_args(argv)
  return args
args = parse_args()
config_file = args.config_file
# check if the config file exists
if not os.path.exists(config_file):
  logger.error(f"Config file does not exist: {config_file}")
  raise FileNotFoundError(f"Config file does not exist: {config_file}")

with open(config_file, "r") as f:
  # may be invalid json
  config = json.load(f)



def str_to_version(version: str):
  return list(map(int, version.split(".")))
def version_to_str(version: tuple):
  return ".".join(map(str, version))

def initialize_blender():
  # Blender version
  logger.info("Blender Version: {}".format(bpy.app.version_string))
  min_blend_version = config["blender_version"][0]
  max_blend_version = config["blender_version"][1]
  if str_to_version(bpy.app.version_string) < min_blend_version:
    logger.warning(f"Blender version is less than the minimum version required: {config['blender_version'][0]}. The script may not work as expected.")
  if str_to_version(bpy.app.version_string) > max_blend_version:
    logger.warning(f"Blender version is greater than the maximum version required: {config['blender_version'][1]}. The script may not work as expected.")

  # FLIP fluids addon
  flip_name = config["flip_fluid_blender_addon_name"]
  addons = {a.__name__:a for a in bpy.utils._addon_utils.modules()}
  if flip_name not in bpy.context.preferences.addons.keys():
    logger.warning("FLIP Fluids addon is not enabled. Install and enable the addon at: https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Addon-Installation-and-Uninstallation")
    logger.info("Current Available addons:")
    addons.keys()
  flip_version = addons[flip_name].bl_info.get('version',None)
  if flip_version != config["flip_fluid_version"]:
    logger.warning(f"FLIP Fluids version is not the expected version: {config['flip_fluid_version']}. The script may not work as expected.")
  logger.info(f"FLIP Fluids Version: {version_to_str(flip_version)}")

  logger.info(f"Output folder: {args.output_folder}")
  
def clear_scene():
  bpy.ops.object.select_all(action='SELECT')
  bpy.ops.object.delete()
  bpy.ops.outliner.orphans_purge(do_recursive=True)

def initialize_fluid_domain():

  # create a Domain
  bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,0), scale=config["domain_dimensions"])
  domain_object = bpy.context.active_object
  domain_object.name = config["domain_object_name"]
  bpy.ops.flip_fluid_operators.flip_fluid_add()
  domain_object.flip_fluid.object_type = 'TYPE_DOMAIN'
  domain_object.flip_fluid.domain.simulation.resolution = config["domain_resolution"]
  domain_object.flip_fluid.domain.simulation_method = 'FLIP'
  domain_object.flip_fluid.domain.advanced.min_max_time_steps_per_frame.value_min = config["domain_min_substeps"]
  logger.info(f"Domain created. Parameters: scale={config['domain_dimensions']}, resolution={config['domain_resolution']}, frame_end={config['num_frames']}, simulation_method={config['simulation_method']}") 


def initialize_fluid():
  # create a fluid object
  bpy.ops.mesh.primitive_cube_add(size=1, location=config["fluid_position"], scale=config["fluid_dimensions"])
  fluid_object = bpy.context.active_object
  fluid_object.name = config["fluid_object_name"]
  bpy.ops.flip_fluid_operators.flip_fluid_add()
  fluid_object.flip_fluid.object_type = 'TYPE_FLUID'
  fluid_object.flip_fluid.fluid.enable = True
  logger.info(f"Fluid object created. Parameters: scale={config['fluid_dimensions']}")

def remove_container_lid():
  # get the container object
  container_object = bpy.data.objects["Container"]
  bpy.ops.object.mode_set(mode='EDIT')
  bpy.ops.mesh.select_all(action='DESELECT')

  bm = bmesh.from_edit_mesh(container_object.data)

  for f in bm.faces:
    f.select = False
    if f.normal.z == 1:
      f.select = True
  bmesh.update_edit_mesh(container_object.data)
  bpy.ops.mesh.delete(type='FACE')
  bpy.ops.object.mode_set(mode='OBJECT')

def initialize_fluid_container():
  # add a rectangular fluid container
  # make it the same height as the fluid domain but the width of the fluid dimenstions
  # set its viewport display to wireframe
  scale = config["fluid_dimensions"].copy()
  scale[2] = config["domain_dimensions"][2]
  bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,0), scale=scale)
  # bpy.ops.transform.resize(value=(0.5,1,1))
  container_object = bpy.context.active_object
  bpy.ops.flip_fluid_operators.flip_fluid_add()
  container_object.flip_fluid.object_type = 'TYPE_OBSTACLE'
  container_object.name = config["container_object_name"]
  container_object.display_type = 'WIRE'
  container_object.flip_fluid.obstacle.is_enabled = True
  container_object.flip_fluid.obstacle.is_inverse = True

  bpy.data.objects["Container"].flip_fluid.obstacle.is_inversed = True
  # add it to the fluid sim



def add_container_motion():

  frame_impulses = config["frame_impulses"]
  # frame_impules = [ [f_no, [x, y, z]], ...]
  # the frame impulses gives us the induced velocity at a certain frame
  # the frame impulses are only a subset of the frames

  # we want to calculate position keyframes for the container
  # to do this, we need to know the position and velocity of the container at the previous keyframe
  
  # we will make our keyframes the same frames as the frame impulses
  # but we will also add keyframes at the start and end of the simulation, if they are not already in the frame impulses
  frame_impulses = sorted(frame_impulses, key=lambda x: x[0])
  # get the start and end frames
  first_frame = frame_impulses[0][0]
  last_frame = frame_impulses[-1][0]

  if first_frame != 1:
    frame_impulses.insert(0, (1, [0,0,0]))
  if last_frame != config["num_frames"]:
    frame_impulses.append((config["num_frames"], [0,0,0]))
  
  init_pos = [0,0,0]
  init_vel = frame_impulses[0][1]
  position_keyframes = [(1, init_pos)]
  frame_t_delta = 1 / bpy.context.scene.render.fps

  prev_pos = init_pos
  prev_vel = init_vel
  prev_frame = frame_impulses[0][0]
  for i in range(1, len(frame_impulses)):
    frame, vel = frame_impulses[i]
    f_delta = frame - prev_frame
    t_delta = f_delta * frame_t_delta
    pos = [prev_pos[j] + prev_vel[j] * t_delta for j in range(3)]
    position_keyframes.append((frame, pos))
    prev_pos = pos
    prev_vel = [prev_vel[j] + vel[j] for j in range(3)]
    prev_frame = frame
  
  # set the keyframes
  container_object = bpy.data.objects["Container"]
  # log the key frames
  for frame, pos in position_keyframes:
    container_object.location = pos
    container_object.keyframe_insert(data_path="location", frame=frame)
    # set interpolation to linear
    for fcurve in container_object.animation_data.action.fcurves:
      for kf in fcurve.keyframe_points:
        kf.interpolation = 'LINEAR'
  
  logger.info(f"Container motion keyframes set: {position_keyframes}")





def initialize_scene():

  # set frame rate
  bpy.context.scene.render.fps = int(config["fps"])
  bpy.context.scene.render.fps_base = 1

  # set the blender scene end frame
  bpy.context.scene.frame_end = config["num_frames"]
  initialize_fluid_domain()
  initialize_fluid()
  initialize_fluid_container()
  add_container_motion()

  for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
      space = area.spaces.active
      region_3d = space.region_3d


      rotation = (1.5708, 0, 0)
      # convert to quaternion
      rotation = mathutils.Euler(rotation).to_quaternion()
      region_3d.view_rotation = rotation


def suppress_console_output():
  sys.stdout.flush()
  sys.stderr.flush()
  devnull = os.open(os.devnull, os.O_WRONLY)
  original_stdout_fd = os.dup2(devnull, 1)
  original_stdout_fd = os.dup2(devnull, 2)
  return original_stdout_fd, original_stdout_fd

def restore_console_output(original_stdout_fd, original_stderr_fd):
    # Flush any pending output
    sys.stdout.flush()
    sys.stderr.flush()

    # Restore stdout and stderr to their original file descriptors
    os.dup2(original_stdout_fd, 1)
    os.dup2(original_stderr_fd, 2)

    # Close the duplicated file descriptors
    os.close(original_stdout_fd)
    os.close(original_stderr_fd)



def bake_simulation(output_file_path):
  #suppres console output
  original_stdout_fd, original_stderr_fd = suppress_console_output()

  domain_object = bpy.data.objects[config["domain_object_name"]]

  # start another thread that monitors the cache directory for the bake
  number_of_frames = config["num_frames"]


  def monitor_bake_status():
    start_time = datetime.now()
    while True:
      out_file_name = os.path.basename(output_file_path).split(".")[0]
      cache_dir = os.path.join(os.path.dirname(output_file_path), f'{out_file_name}_flip_fluid_cache')
      if os.path.exists(f'{cache_dir}/bakefiles'):
        files = os.listdir(f'{cache_dir}/bakefiles')
        baked_frames = len([f for f in files if re.match(r"\d+.bobj", f)])
        logger.info(f"Baked frames: {baked_frames}/{number_of_frames} : {baked_frames/number_of_frames*100:.2f}%")

        # write to status.txt
        # get the cwd
        cwd = os.getcwd()
        with open(os.path.join(cwd, "status.txt"), "w") as f:
          out_str = f"File: {output_file_path}\n"
          out_str += f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
          out_str += f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
          out_str += f"Baked frames: {baked_frames}/{number_of_frames} : {baked_frames/number_of_frames*100:.2f}%\n"
          f.write(out_str)


        if baked_frames >= number_of_frames - 1:
          break
      time.sleep(0.01)
    logger.info(f"Simulation baked")
  
  # start the monitor thread
  monitor_thread = threading.Thread(target=monitor_bake_status)
  monitor_thread.start()

  bpy.context.scene.frame_start = 1
  bpy.context.scene.frame_end = config["num_frames"]
  # deselect everything and select the domain
  bpy.ops.object.select_all(action='DESELECT')
  bpy.ops.object.select_all(action='SELECT')
  # selevt the domain
  bpy.context.view_layer.objects.active = bpy.data.objects[config["domain_object_name"]]
  cache_dir = os.path.join(os.path.dirname(output_file_path), "cache")
  bpy.data.objects[config["domain_object_name"]].flip_fluid.domain.cache_directory = bpy.path.relpath(cache_dir)
  # bake the simulation
  logger.info(f"Baking simulation")
  # reset the baked fluid simulation before baking
  bpy.ops.flip_fluid_operators.reset_bake()
  bpy.ops.flip_fluid_operators.bake_fluid_simulation_cmd()
  logger.info(f"Simulation baked")

  monitor_thread.join()

  # restore console output
  restore_console_output(original_stdout_fd, original_stderr_fd)




def save_scene(output_file_path):

  # if the output folder does not exist, create it
  if not os.path.exists(args.output_folder):
    try:
      os.makedirs(args.output_folder)
    except OSError as e:
      logger.error(f"Error creating output folder: {args.output_folder}")
      raise e
  # save the scene
  # include the date and time in the output file name

  # if the output file exists, delete it
  if os.path.exists(output_file_path):
    os.remove(output_file_path)
  

  bpy.ops.wm.save_as_mainfile(filepath=output_file_path)
  # log the output
  logger.info(f"Scene saved to: {os.path.join(args.output_folder, 'tank_2d_motion.blend')}")



def create_new_trial(output_folder, trial_name=None):
  # use the folder name as the trial name
  if trial_name is None:
    trial_name = os.path.basename(output_folder)
  # output_folder / stamp / stamp.blend
  trial_folder = output_folder
  output_file_path = os.path.join(trial_folder, f"{trial_name}.blend")
  if not os.path.exists(trial_folder):
    try:
      os.makedirs(trial_folder)
    except OSError as e:
      logger.error(f"Error creating output folder: {trial_folder}")
      raise e

  log_capture = BlenderLogInterCeptor()
  initialize_blender()
  clear_scene()
  initialize_scene()
  save_scene(output_file_path)
  config_file_path = os.path.join(trial_folder, "trial_config.json")
  # save to json, make sure utf-8 encoding is used
  with open(config_file_path, "w", encoding="utf-8") as f:
    json.dump(config, f)

  bake_simulation(output_file_path)

  log_capture.close()

  return trial_folder

def main():


  create_new_trial(args.output_folder)
  pass

if __name__ == "__main__":
  main()

