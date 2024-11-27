ml-pinn folder was first attempt and a lot of the code was not setup correctly.
Deciding to use a more modern example shown here that also considers higher-order derivatives.

https://github.com/kochlisGit/Physics-Informed-Neural-Network-PINN-Tensorflow/tree/main

Also look at this website for DeepBSDE method..
https://github.com/janblechschmidt/PDEsByNNs/tree/main

Notes:
4_strong-4th_rayleigh_v2 uses direct potential energy but then NN overfit and 
boundary term can get very neagtive + you are undersampled with high curvature function.