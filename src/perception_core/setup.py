"""Optional C++ geometry extension; the Python implementation remains available."""
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

extensions = []
if os.environ.get("PERCEPTION_CORE_NO_NATIVE") != "1":
    extensions.append(Pybind11Extension(
        "perception_core._geometry", ["cpp/geometry.cpp"], cxx_std=14, optional=True,
    ))

setup(ext_modules=extensions, cmdclass={"build_ext": build_ext})
