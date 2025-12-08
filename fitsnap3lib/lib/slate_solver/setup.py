from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys
import subprocess
import shutil
import glob

# -----------------------------------------------------------------------------
# 1. Robust mpi4py Detection (Handles Build Isolation on Clusters)
# -----------------------------------------------------------------------------
mpi4py_include = []
try:
    import mpi4py
    mpi4py_include = [mpi4py.get_include()]
except ImportError:
    # We are likely in an isolated build environment where module site-packages are hidden.
    # On Trillium/EasyBuild, we can find mpi4py via the EBROOTMPI4PY env var.
    eb_root = os.environ.get("EBROOTMPI4PY")
    if eb_root:
        # Search for the include directory inside the module root
        # Typically: $EBROOTMPI4PY/lib/pythonX.Y/site-packages/mpi4py/include
        found_includes = glob.glob(os.path.join(eb_root, "lib", "python*", "site-packages", "mpi4py", "include"))
        if found_includes:
            print(f"Found mpi4py headers via EBROOTMPI4PY: {found_includes[0]}")
            mpi4py_include = [found_includes[0]]
        else:
            print("Warning: EBROOTMPI4PY set but include path not found.")
    else:
        print("Warning: Could not find mpi4py headers. Cython cimport mpi4py may fail.")

# -----------------------------------------------------------------------------
# 2. MPI Compiler Wrapper Detection (Your original logic)
# -----------------------------------------------------------------------------
try:
    mpicc_path = shutil.which("mpicc")
    if mpicc_path is None:
        raise RuntimeError("mpicc not found")
    
    mpi_compile_flags = subprocess.check_output(
        ["mpicc", "--showme:compile"]
    ).decode().strip().split()
    
    mpi_link_flags = subprocess.check_output(
        ["mpicc", "--showme:link"]
    ).decode().strip().split()
    print(f"Found MPI compiler: {mpicc_path}")
except Exception as e:
    print(f"Warning: MPI detection failed ({e}). Using defaults.")
    mpi_compile_flags = []
    mpi_link_flags = ["-lmpi"]

mpi_include_dirs = [f[2:] for f in mpi_compile_flags if f.startswith("-I")]

# -----------------------------------------------------------------------------
# 3. SLATE Detection
# -----------------------------------------------------------------------------
def find_slate_dir():
    candidates = [
        os.environ.get("SLATE_DIR", ""),
        os.path.expanduser("~/.local"),
        "/usr/local",
        "/opt/slate",
    ]
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "include", "slate")):
            print(f"Found SLATE in: {path}")
            return path
    return os.environ.get("SLATE_DIR", "/usr/local")

slate_dir = find_slate_dir()

# -----------------------------------------------------------------------------
# 4. Final Configuration
# -----------------------------------------------------------------------------
include_dirs = [
    np.get_include(),
    os.path.join(slate_dir, "include"),
] + mpi4py_include + mpi_include_dirs

library_dirs = [
    os.path.join(slate_dir, "lib"),
    os.path.join(slate_dir, "lib64"),
]

libraries = ["slate", "blas", "lapack"]

# OpenMP & Standard Flags
extra_compile_args = ["-std=c++17", "-O3"] + [
    f for f in mpi_compile_flags if not f.startswith("-I")
]
extra_link_args = [f for f in mpi_link_flags if not f.startswith("-l")]

# Force OpenMP on Linux/Cluster
extra_compile_args += ["-fopenmp"]
extra_link_args += ["-fopenmp"]

print(f"Include Dirs: {include_dirs}")

ext = Extension(
    "slate_wrapper",
    sources=["slate_wrapper.pyx"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="slate_solver",
    ext_modules=cythonize(
        [ext],
        compiler_directives={'language_level': "3"}
    ),
    zip_safe=False,
)
