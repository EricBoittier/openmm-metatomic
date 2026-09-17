from setuptools import setup, Extension, find_packages
import os
import platform

openmm_dir = "@OPENMM_DIR@"
plugin_header_dir = "@PLUGIN_HEADER_DIR@"
plugin_library_dir = "@PLUGIN_LIBRARY_DIR@"
wrap_file = "@WRAP_FILE@"

extra_compile_args = ["-std=c++17"]
extra_link_args = []
runtime_library_dirs = [
    os.path.join(openmm_dir, "lib"),
    plugin_library_dir,
]
if platform.system() == "Darwin":
    extra_compile_args += ["-stdlib=libc++"]
    extra_link_args += ["-stdlib=libc++"]
    runtime_library_dirs = None

extension = Extension(
    name="openmmmetatomic._openmmmetatomic",
    sources=[wrap_file],
    libraries=["OpenMM", "OpenMMMetatomic"],
    include_dirs=[os.path.join(openmm_dir, "include"), plugin_header_dir],
    library_dirs=[os.path.join(openmm_dir, "lib"), plugin_library_dir],
    runtime_library_dirs=runtime_library_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="openmmmetatomic",
    version="0.0.0",
    packages=find_packages(),
    ext_modules=[extension],
    install_requires=["openmm"],
)
