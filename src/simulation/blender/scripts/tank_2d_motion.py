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
from datetime import datetime

import bmesh

bpy.app.debug = False
bpy.app.debug_wm = False
bpy.app.debug_events = False




from logs import logger, BlenderLogInterCeptor

config = {
  "flip_fluid_blender_addon_name": "flip_fluids_addon",
  "flip_fluid_version": (1,8,1),
  "blender_version": ((3,1),(4,2)),

  # flip fluid parameters
  "domain_dimensions": (22,7,12),
  "domain_object_name": "Domain",
  "domain_resolution": 65,
  "fluid_dimensions": (10,5,3.0),
  "fluid_position": (0,0,-2.0),
  "fluid_object_name": "Fluid",
  "num_frames": 100,
  "simulation_method": "FLIP",
}

def parse_args():
  argv = sys.argv
  if "--" in argv:
    argv = argv[argv.index("--") + 1:]
  else:
    argv = []
  parser = argparse.ArgumentParser(description="Simulate fluid tank")

  # get the cwd
  cwd = os.getcwd()

  relative_path = os.path.join(cwd, "scenes")
  parser.add_argument("--output_folder", 
                      type=str, 
                      help="Output folder for simulation data",
                      default=relative_path,
                      )
  args = parser.parse_args(argv)
  return args
args = parse_args()



def str_to_version(version: str):
  return tuple(map(int, version.split(".")))
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
  # make it the same height as the fluid domain but half the width on axis
  # set its viewport display to wireframe
  # remove the top face
  # add a modifier giving it a solidify modifier
  bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,0), scale=config["domain_dimensions"])
  bpy.ops.transform.resize(value=(0.5,1,1))
  container_object = bpy.context.active_object
  bpy.ops.flip_fluid_operators.flip_fluid_add()
  container_object.flip_fluid.object_type = 'TYPE_OBSTACLE'
  container_object.name = "Container"
  container_object.display_type = 'WIRE'

  # remove the top face
  remove_container_lid()

  bpy.ops.object.modifier_add(type='SOLIDIFY')
  bpy.context.object.modifiers["Solidify"].thickness = -4.0

  # add it to the fluid sim

  logger.info(f"Container object created. Parameters: scale={config['fluid_dimensions']}")




def add_container_motion():

  # we want to move only along the x axis
  # the first key frame should be at frame 1, and should be at its original position
  # we will work with percentages of the total number of frames
  container_velocity = 6.0
  # fps is 24
  # we want to pick random positions on the domain
  # the domain is the length , well the size of the fluid domain / 2 because we scaled the container
  
  p_k = 0.05 # percentage of frames we want to be a keyframe position
  key_frames = [(1, 0)]
  x_domain = config["domain_dimensions"][0] // 4
  time_per_frame = 1/24

  down_time = (5, 30)

  i = 2
  while i < config["num_frames"]:
    if random.random() < p_k:
      # get a random position
      frame_delta = i - key_frames[-1][0]
      delta_time = frame_delta * time_per_frame
      max_distance = container_velocity * delta_time
      prev_x = key_frames[-1][1]
      x_min = max(prev_x - max_distance, -x_domain)
      x_max = min(prev_x + max_distance, x_domain)
      x = random.uniform(x_min, x_max)
      # get the frame
      frame = int(i)
      key_frames.append((frame, x))
      cur_down_time = random.randint(down_time[0], down_time[1])
      i += cur_down_time
      key_frames.append((int(i), x))
    else:
      i += 1

  
  # log the key frames
  logger.info(f"Key frames: {key_frames}")
  # set the key frames
  container_object = bpy.data.objects["Container"]
  for frame, x in key_frames:
    container_object.location.x = x
    container_object.keyframe_insert(data_path="location", frame=frame)
    # set interpolation to linear
    for fcurve in container_object.animation_data.action.fcurves:
      for kf in fcurve.keyframe_points:
        kf.interpolation = 'LINEAR'
    
  logger.info(f"Container motion keyframes set: {key_frames}")

def initialize_scene():

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
  

  bpy.ops.wm.save_as_mainfile(filepath=output_file_path)
  # log the output
  logger.info(f"Scene saved to: {os.path.join(args.output_folder, 'tank_2d_motion.blend')}")

def main():

  output_file_name = f"tank_2d_motion_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.blend"
  output_file_path = os.path.join(args.output_folder, output_file_name)

  log_capture = BlenderLogInterCeptor()
  initialize_blender()
  clear_scene()
  initialize_scene()
  save_scene(output_file_path)
  bake_simulation(output_file_path)


  log_capture.close()
  pass

if __name__ == "__main__":
  main()

