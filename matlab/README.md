The provided MATLAB MEX functions are

`ddm_fpt`: computed the first-passage time density for symmetric bounds. See [ddm_fpt.m](ddm_fpt.m) for usage information.

`ddm_fpt_full`: computes the first-passage time density for arbitrary bounds. See [ddm_fpt_full.m](ddm_fpt.m) for usage information.

`ddm_rand_sym`: samples first-passage time and boundary from diffusion models with symmetric bounds. See [ddm_rand_sym.m](ddm_rand_sym.m) for usage information.

`ddm_rand_asym`: samples first-passage time and boundary from diffusion models with asymmetric bounds. See [ddm_rand_asym.m](ddm_rand_asym.m) for usage information.

Usage
-----

The MATLAB MEX functions need to be compiled before use. To do so, run

```Shell
mex CXXFLAGS='$CXXFLAGS -std=c++11' ddm_fpt.cpp ../src/ddm_fpt_lib.cpp
mex CXXFLAGS='$CXXFLAGS -std=c++11' ddm_fpt_full.cpp ../src/ddm_fpt_lib.cpp
mex CXXFLAGS='$CXXFLAGS -std=c++11' ddm_rand_sym.cpp ../src/ddm_fpt_lib.cpp
mex CXXFLAGS='$CXXFLAGS -std=c++11' ddm_rand_asym.cpp ../src/ddm_fpt_lib.cpp
```

at the MATLAB prompt. Alternatively, the files can be compiled at the shell. Do do so, use

```Shell
mex CXXFLAGS="$CXXFLAGS -std=c++11" ddm_fpt.cpp ../src/ddm_fpt_lib.cpp
mex CXXFLAGS="$CXXFLAGS -std=c++11" ddm_fpt_full.cpp ../src/ddm_fpt_lib.cpp
mex CXXFLAGS="$CXXFLAGS -std=c++11" ddm_rand_sym.cpp ../src/ddm_fpt_lib.cpp
mex CXXFLAGS="$CXXFLAGS -std=c++11" ddm_rand_asym.cpp ../src/ddm_fpt_lib.cpp
```

at the Windows Command Prompt (using double quotes, (")), or

```Shell
mex CXXFLAGS='$CXXFLAGS -std=c++11' ddm_fpt.cpp ../src/ddm_fpt_lib.cpp
mex CXXFLAGS='$CXXFLAGS -std=c++11' ddm_fpt_full.cpp ../src/ddm_fpt_lib.cpp
mex CXXFLAGS='$CXXFLAGS -std=c++11' ddm_rand_sym.cpp ../src/ddm_fpt_lib.cpp
mex CXXFLAGS='$CXXFLAGS -std=c++11' ddm_rand_asym.cpp ../src/ddm_fpt_lib.cpp
```

at the Mac and Linux shell command line (using single quotes (')). Note that, for older versions of MATLAB, all occurences of `CXXFLAGS` might need to be replaced with `CFLAGS`.

The location of the `mex` executable and exact syntax for specifying the compiler flags might be depend on OS and installation details.
