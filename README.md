# [ContactGrasp: Functional Multi-finger Grasp Synthesis from Contact](https://contactdb.cc.gatech.edu/contactgrasp.html)

Utilities for ContactGrasp, including GraspIt! sampler for initializing ContactGrasp.

## Citation

"[ContactGrasp: Functional Multi-finger Grasp Synthesis from Contact](https://arxiv.org/abs/1904.03754)" -

[Samarth Brahmbhatt](https://samarth-robo.github.io),
[Ankur Handa](https://ankurhanda.github.io/),
[James Hays](https://www.cc.gatech.edu/~hays/), and
[Dieter Fox](https://research.nvidia.com/node/2945). IROS 2019.

```
@INPROCEEDINGS{brahmbhatt2019contactgrasp,
  title={{ContactGrasp: Functional Multi-finger Grasp Synthesis from Contact}},
  author={Brahmbhatt, Samarth and Handa, Ankur and Hays, James and Fox, Dieter},
  booktitle={2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2019}
}
```

## Companion Repositories:

- [contactgrasp/dart](https://github.com/contactgrasp/dart): Main entry point.
- [contactgrasp/ros_thermal_grasp](https://github.com/contactgrasp/ros_thermal_grasp): ROS code for executing ContactGrasp grasps with MoveIt!.

## Documentation

Unfortunately, the code is not very well documented right now. Please create issues for questions.

[`dart/src/grasp_analyzer_main.cpp`](https://github.com/contactgrasp/dart/blob/master/src/grasp_analyzer.cpp) is the ContactGrasp starting point. It is a GUI which allows you to load an object mesh, attractive/repulsive points, and a hand model (with init parameters). It also allows you to tune various hyperparameters (like the strength of attraction/repulsion), to optimize the hand the ContactGrasp cost function, and to save the optimized parameters of the hand.

This repository has some useful scripts, like

- [`scripts/generate_contact_info.py`](scripts/generate_contact_info.py) generates the attractive and repulsive points from ContactDB data, in the format expected by [`dart/src/grasp_analyzer_main.cpp`](https://github.com/contactgrasp/dart/blob/master/src/grasp_analyzer.cpp).
- [`dataset_tools/collect_graspit_grasps.py`](dataset_tools/collect_graspit_grasps.py): samples the stable grasps from GraspIt, and saves them in a format expected as initialization by [`dart/src/grasp_analyzer_main.cpp`](https://github.com/contactgrasp/dart/blob/master/src/grasp_analyzer.cpp).
- [`utils/graspit2dart.py`](utils/graspit2dart.py): convert GraspIt models to DART models.

