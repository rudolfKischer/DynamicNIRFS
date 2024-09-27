# DynamicNIRFS
Dynamic External Forces for Neural Implicit Reduced Fluid Simulations



## Setup

- Install blender
- Download and install the flip fluids blender add on
- Create a virtual environment
    - `virtualenv .venv`
- Activate the virtual environment
    - `source .venv/bin/activate`
- Install the requirements
    - `pip install -r requirements.txt`

# Running simulation

- change working directory to the scripts folder
    - `cd src/simulation/blender/scripts/`
- execute the simulation script with blender
    - `blender --background --python tank_2d_motion.py` 
- the results end up in `src/simulation/blender/scripts/scenes/` as `.blend` files
- Note: Blender stores the baked simulation in the blend cache folder, so the blend file and the cache dir need to remain together
