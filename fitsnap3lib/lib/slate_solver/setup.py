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
    We must find it to get the 'include' directory for omp.h.
    """
    if sys.platform != "darwin":
        return [], [], []

    # Try standard Homebrew location first (fastest)
    candidates = [
        "/opt/homebrew/opt/libomp",
        "/usr/local/opt/libomp"
    ]
    
    # If not found, ask brew (slower but robust)
    for c in candidates:
        if os.path.exists(c):
            return [f"{c}/include"], [f"{c}/lib"], ["-lomp"]

    try:
        prefix = subprocess.check_output(["brew", "--prefix", "libomp"]).decode().strip()
        return [f"{prefix}/include"], [f"{prefix}/lib"], ["-lomp"]
    except (OSError, subprocess.CalledProcessError):
        # Fallback: Hope it's in a standard path or user supplied it
        print("WARNING: Could not find libomp via Homebrew. Build might fail if omp.h is missing.")
        return [], [], ["-lomp"]

# Get OpenMP paths
omp_includes, omp_libdirs, omp_libs = find_libomp()

# Safe import for mpi4py
try:
    import mpi4py
    mpi_include = mpi4py.get_include()
except ImportError:
    mpi_include = []

# -----------------------------------------------------------------------------
# BUILD CONFIGURATION
# -----------------------------------------------------------------------------

# Detect OS for compiler flags
is_macos = (sys.platform == "darwin")

# Compiler flags
cxx_flags = ["-std=c++17", "-O3", "-fPIC", "-DNDEBUG"]
link_flags = []

if is_macos:
    # Apple Clang specific OpenMP flags
    cxx_flags += ["-Xpreprocessor", "-fopenmp"]
    link_flags += omp_libs # adds -lomp
else:
    # Linux (GCC)
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
            mpi_include,
            # Add dynamically found OpenMP include path
            *omp_includes, 
            # Fallbacks
            os.path.join(os.environ.get("HOME", ""), ".local/include"),
            "/usr/local/include",
            "/opt/homebrew/include"
        ],
        library_dirs=[
            # Add dynamically found OpenMP lib path
            *omp_libdirs,
            os.path.join(os.environ.get("HOME", ""), ".local/lib"),
            os.path.join(os.environ.get("HOME", ""), ".local/lib64"),
            "/usr/local/lib",
            "/opt/homebrew/lib"
        ],
        # 'omp' usually implied by -fopenmp on Linux, but needed explicit on Mac
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
