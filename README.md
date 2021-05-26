# Thermal Grasp: NVIDIA summer 2018 internship

# Setup
- Install miniconda3
- `conda env create -f environment.yml`
- Activate the environment `thermal_grasp`
- Install [Open3D](http://www.open3d.org/docs/getting_started.html#ubuntu)
through a terminal where the conda environment has been activated

# GraspIt!
- It expects world files and geometry in units of mm, but `graspit_commander`'s
`getRobot()` function returns the robot's position in units of m.
- Edit `src/EGPlanner/searchStateImpl.cpp`'s
`PositionStateApproach::createVariables()` function and set `wrist1` and
`wrist2` to `(-pi/12, pi/12, 0, pi/24)`
- Because the `HumanHand20DOF` hand model is larger than life, all graspit
grasp planning is done with object models scaled up by a factor of 1.15