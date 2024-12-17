from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# scenred 디렉토리가 없다면 생성
if not os.path.exists('scenred'):
    os.makedirs('scenred')

setup(
    name='scenred',  # scen_red에서 scenred로 수정
    packages=['scenred'],
    ext_modules=cythonize("scenario_reduction_c.pyx"),
    include_dirs=[numpy.get_include()]
)