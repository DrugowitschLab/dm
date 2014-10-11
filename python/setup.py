from numpy.distutils.core import setup, Extension

module1 = Extension('ddm',
                    sources = ['ddmmodule.cpp', '../ddm_fpt_lib/ddm_fpt_lib.cpp'],
                    extra_compile_args = ['-Wno-write-strings'])

setup (name = 'DDM',
       version = '1.0',
       description = 'Methods to compute the DDM first passage-time distributions',
       ext_modules = [module1])