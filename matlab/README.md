The provided MATLAB MEX functions are

`ddm_fpt`: function for symmetric bounds. See [ddm_fpt.m](ddm_fpt.m) for usage information.

`ddm_fpt_full`: function for arbitrary bounds. See [ddm_fpt_full.m](ddm_fpt.m) for usage information.

Usage
-----

The MATLAB MEX functions need to be compiled before use. To do so, run

```Shell
mex CFLAGS='$CFLAGS -std=c++11' ddm_fpt.cpp ../src/ddm_fpt_lib.cpp
mex CFLAGS='$CFLAGS -std=c++11' ddm_fpt_full.cpp ../src/ddm_fpt_lib.cpp
```

at the command line. The location of the `mex` executable and exact syntax for specifying the compiler flags might be OS-dependent.
