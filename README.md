# README #

This version of `EightBitTransit` is an MIT-licensed python code redesigned to make use of GPU parallelization with Numba:
1. Can calculate the light curve of any pixelated image transiting a star;
2. Can invert a light curve to recover the "shadow image" that produced it.

# Installation #

To install EightBitTransit, download this directory, navigate to it, and run:

`pip install .`

# Dependencies #
* Numpy
* Scipy
* Numba
* imageio
* itertools
* matplotlib
  
Optionally (for the example notebook):
* jupyterlab

# Examples #

See `./examples/examples-gpu.ipynb` for examples of both the forward and inverse shadow imaging problem, including for dip 5 of Boyajian's Star. This code reproduces figure 12 of Sandford & Kipping 2018 (https://arxiv.org/abs/1812.01618).

# Note on memory #

Repeated TransitingImage() calls can cause python to run out of memory in certain cases (thanks to textadactyl for pointing this out!)--when you are done with a TransitingImage object, deallocate it explicitly, i.e.:

`ti = TransitingImage(...)`

*some operations on ti, etc.*

`ti = None`
