Introdction
===========

This code provides a reference implementation of method presented in the paper:

"Estimating Motion Uncertainty With Bayesian ICP"
F. Afzal Maken, F. Ramos, L. Ott
submitted to IEEE International Conference on Robotics Automation, 2020.

and 

"Speeding up Iterative Closest Point Using Stochastic Gradient Descent"
F. Afzal Maken, F. Ramos, L. Ott
ICRA 2019.

Requirements
============

The following libraries are needed to compile and run the code, the is
known to run on Ubuntu 16.04.

- boost
- eigen3
- pcl


Building
========

You will need CMake (at least version 3.1) to build the code.

```
cd sgd_icp
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release
make
```

Running
=======

Compiling the code will produce three executables:
- align_pcd
- pcl_align_pcd
- transform_cloud


align_pcd
---------

This program uses the Bayesian ICP implementation to estimate the transform and distribution
between two point clouds, the command line syntax is as follows:

`align_pcd <source_cloud> <target_cloud> <cofig_file>`

The configuration file, an example of which is provided in config.json,
sets the various parameters of the algorithm, including the initial
estimate of the transform.


pcl_align_pcd
-------------

This is a simple program which aligns the two provided points clouds
using PCL with parameters similar to the default values used by
align_pcd.

`pcl_align_pcd <source_cloud> <target_cloud>`


transform_cloud
---------------

This program applies a user specified transform onto a given cloud.

`transform_cloud --input <source_cloud> --output <output_cloud> --pose <6D vector>`
