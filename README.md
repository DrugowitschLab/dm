dm
==

Diffusion model first passage time distribution library in C, with Python and MATLAB interface.

*No guarantee is provided for the correctness of the implementation.*

The code is licensed under the New BSD License.

Content
-------

The library consists of functions written in ANSI C 90 that compute the first-passage time distribution of diffusion models. It provides various functions with different degrees of optimisation. Depending on the function, they support leaky/weighted integration, time-varying drift rates, and time-varying boundaries.

In addition to the C implementation, a Python and a MATLAB interface is provided. In both cases, the provided interface calls the various different library function depending on the parameters provided.

For constant drift and bounds, the functions compute the first-passage time distribution by methods described in

Cox DR and Miller HD (1965). *The Theory of Stochastic Processes*. John Wiley & Sons, Inc.

and

Navarro DJ and Fuss IG (2009). Fast and accurate calculations for first-passage times in Wiener diffusion models. *Journal of Mathematical Psychology*, 53, 222-230.

For all other cases, the implementation described in

Smith PL (2000). Stochastic Dynamic Models of Response Time and Accuracy: A Foundational Primer. *Journal of Mathematical Psychology*, 44 (3). 408-463.

is used.

For details of the available C functions, see ``ddm_fpt_lib.h`` in the ``ddm_fpt_lib`` directory.

The provided MATLAB MEX functions in the ``matlab`` directory are

``ddm_fpt``: function for symmetric bounds. See ``ddm_fpt.m`` for usage information.

``ddm_fpt_full``: function for arbitrary bounds. See ``ddm_fpt_full.m`` for usage information.

The Python ``ddm`` module is in the ``python`` directory.

Usage
-----

To use the C library, include ``ddm_fpt_lib.h`` and link to compiled ``ddm_fpt_lib.c``

The MATLAB MEX functions need to be compiled before use. To do so, run

    mex ddm_fpt.c ../ddm_fpt_lib/ddm_fpt_lib.c
    mex ddm_fpt_full.c ../ddm_fpt_lib/ddm_fpt_lib.c

at the command line. The location of the ``mex`` executable might be OS-dependent.

The Python module can be built by calling

    python setup.py build

and installed by

    python setup.py install

at the command line. The Python module should be compatible with both Python 2 and 3. See the header comment in ``ddmmodule.c`` for how to call the different functions provided by the module.
