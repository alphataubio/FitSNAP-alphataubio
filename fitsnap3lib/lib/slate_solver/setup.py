import sys
import os
import subprocess
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

# -----------------------------------------------------------------------------
# HELPER: Find OpenMP on macOS
# -----------------------------------------------------------------------------
def find_libomp():
    if sys.platform != "darwin":
        return [], [], []
    candidates = ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
    for c in candidates:
        if os.path.exists(c):
            return [f"{c}/include"], [f"{c}/lib"], ["-lomp"]
    try:
        p = subprocess.check_output(["brew", "--prefix", "libomp"]).decode().strip()
        return [f"{p}/include"], [f"{p}/lib"], ["-lomp"]
    except:
        return [], [], ["-lomp"]

# -----------------------------------------------------------------------------
# HELPER: Find MPI Flags (Robust for Linux/Mac)
# -----------------------------------------------------------------------------
def find_mpi_flags():
    mpi_includes = []
    try:
        # Try to get flags from mpicc (common on Linux/Cluster)
        cflags = subprocess.check_output(["mpicc", "--showme:compile"], stderr=subprocess.DEVNULL).decode().strip()
        for flag in cflags.split():
            if flag.startswith("-I"):
                mpi_includes.append(flag[2:])
    except:
        # Fallback for standard locations if mpicc missing/fails
        if sys.platform == "linux":
            mpi_includes.extend([
                "/usr/lib/x86_64-linux-gnu/openmpi/include",
                "/usr/include/openmpi",
                "/usr/lib/openmpi/include"
            ])
    return mpi_includes

# Get paths
omp_includes, omp_libdirs, omp_libs = find_libomp()
mpi_includes = find_mpi_flags()

# Safe import for mpi4py
try:
    import mpi4py
    # FIX: Wrap in list so we can always use * unpacking
    mpi_site_include = [mpi4py.get_include()]
except ImportError:
    # FIX: Empty list for clean unpacking
    mpi_site_include = []
    print("WARNING: mpi4py not found in build env. relying on system paths.")

# -----------------------------------------------------------------------------
# BUILD CONFIGURATION
# -----------------------------------------------------------------------------

is_macos = (sys.platform == "darwin")

cxx_flags = ["-std=c++17", "-O3", "-fPIC", "-DNDEBUG"]
link_flags = []

if is_macos:
    cxx_flags += ["-Xpreprocessor", "-fopenmp"]
    link_flags += omp_libs 
else:
    cxx_flags += ["-fopenmp"]
    link_flags += ["-fopenmp"]

extensions = [
    Extension(
        "slate_wrapper",
        sources=[
            "slate_wrapper.pyx",
            "slate.cpp"
        ],
        language="c++",
        include_dirs=[
            ".", 
            np.get_include(),
            *mpi_site_include, # FIX: Correctly unpacks [path] or []
            *mpi_includes, 
            *omp_includes, 
            os.path.join(os.environ.get("HOME", ""), ".local/include"),
            "/usr/local/include",
            "/opt/homebrew/include",
            "/usr/include" 
        ],
        library_dirs=[
            *omp_libdirs,
            os.path.join(os.environ.get("HOME", ""), ".local/lib"),
            os.path.join(os.environ.get("HOME", ""), ".local/lib64"),
            "/usr/local/lib",
            "/opt/homebrew/lib",
            "/usr/lib/x86_64-linux-gnu"
        ],
        libraries=["slate", "lapack", "blas", "mpi"], 
        extra_compile_args=cxx_flags,
        extra_link_args=link_flags,
    ),
]

setup(
    name="slate_solver",
    version="0.1.0",
    ext_modules=cythonize(
        extensions, 
        compiler_directives={'language_level': "3"}
    ),
)
