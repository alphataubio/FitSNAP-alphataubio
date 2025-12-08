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
    """
    On macOS, OpenMP (libomp) is keg-only and not in default paths.
    """
    if sys.platform != "darwin":
        return [], [], []

    candidates = ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
    for c in candidates:
        if os.path.exists(c):
            return [f"{c}/include"], [f"{c}/lib"], ["-lomp"]

    try:
        prefix = subprocess.check_output(["brew", "--prefix", "libomp"]).decode().strip()
        return [f"{prefix}/include"], [f"{prefix}/lib"], ["-lomp"]
    except (OSError, subprocess.CalledProcessError):
        return [], [], ["-lomp"]

# -----------------------------------------------------------------------------
# HELPER: Find MPI Flags (Robust for Linux/Mac)
# -----------------------------------------------------------------------------
def find_mpi_flags():
    """
    Use mpicc to find the correct include/link flags.
    Crucial for Linux where mpi.h might be in /usr/lib/.../openmpi/include
    """
    mpi_includes = []
    mpi_link_args = []
    
    # Try finding mpicc in path
    mpicc = "mpicc"
    # Check if MPICH or OpenMPI specific env vars are set, though mpicc usually suffices
    
    try:
        # Get compile flags (includes)
        cflags = subprocess.check_output([mpicc, "--showme:compile"], stderr=subprocess.DEVNULL).decode().strip()
        for flag in cflags.split():
            if flag.startswith("-I"):
                mpi_includes.append(flag[2:])
                
        # Get link flags (if needed, though we usually just link -lmpi)
        # ldflags = subprocess.check_output([mpicc, "--showme:link"], stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        # Fallback for MPICH or if --showme isn't supported (try -compile_info)
        try:
            cflags = subprocess.check_output([mpicc, "-compile_info"], stderr=subprocess.DEVNULL).decode().strip()
            for flag in cflags.split():
                if flag.startswith("-I"):
                    mpi_includes.append(flag[2:])
        except:
            # Last resort: common locations
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
    mpi_site_include = mpi4py.get_include()
except ImportError:
    mpi_site_include = []

# -----------------------------------------------------------------------------
# BUILD CONFIGURATION
# -----------------------------------------------------------------------------

# Detect OS for compiler flags
is_macos = (sys.platform == "darwin")

# Compiler flags
cxx_flags = ["-std=c++17", "-O3", "-fPIC", "-DNDEBUG"]
link_flags = []

if is_macos:
    cxx_flags += ["-Xpreprocessor", "-fopenmp"]
    link_flags += omp_libs # adds -lomp
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
            ".",  # Local directory
            np.get_include(),
            mpi_site_include,
            *mpi_includes, # <--- Added system MPI headers
            *omp_includes, 
            os.path.join(os.environ.get("HOME", ""), ".local/include"),
            "/usr/local/include",
            "/opt/homebrew/include",
            "/usr/include" # Standard Linux include
        ],
        library_dirs=[
            *omp_libdirs,
            os.path.join(os.environ.get("HOME", ""), ".local/lib"),
            os.path.join(os.environ.get("HOME", ""), ".local/lib64"),
            "/usr/local/lib",
            "/opt/homebrew/lib",
            "/usr/lib/x86_64-linux-gnu" # Standard Linux libs
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
