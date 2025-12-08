from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys
import subprocess
import shutil
import glob

# -----------------------------------------------------------------------------
# 1. Robust mpi4py Detection
# -----------------------------------------------------------------------------
mpi4py_paths = []
try:
    import mpi4py
    mpi4py_inc = mpi4py.get_include()
    mpi4py_paths.append(mpi4py_inc)
    # Add parent (site-packages) for 'cimport mpi4py'
    mpi4py_paths.append(os.path.dirname(os.path.dirname(mpi4py_inc)))
except ImportError:
    eb_root = os.environ.get("EBROOTMPI4PY")
    if eb_root:
        found_includes = glob.glob(os.path.join(eb_root, "lib", "python*", "site-packages", "mpi4py", "include"))
        if found_includes:
            found_inc = found_includes[0]
            print(f"Found mpi4py headers: {found_inc}")
            mpi4py_paths.append(found_inc)
            # Add site-packages parent
            site_pkg = os.path.dirname(os.path.dirname(found_inc))
            print(f"Adding site-packages to Cython path: {site_pkg}")
            mpi4py_paths.append(site_pkg)
        else:
            print("Warning: EBROOTMPI4PY set but include path not found.")

# -----------------------------------------------------------------------------
# 2. MPI Compiler Wrapper Detection
# -----------------------------------------------------------------------------
try:
    mpicc_path = shutil.which("mpicc")
    if mpicc_path is None:
        raise RuntimeError("mpicc not found")
    
    mpi_compile_flags = subprocess.check_output(["mpicc", "--showme:compile"]).decode().strip().split()
    mpi_link_flags = subprocess.check_output(["mpicc", "--showme:link"]).decode().strip().split()
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
] + mpi4py_paths + mpi_include_dirs

library_dirs = [
    os.path.join(slate_dir, "lib"),
    os.path.join(slate_dir, "lib64"),
]

# --- CRITICAL FIX HERE ---
# Changed 'blas'/'lapack' to 'blaspp'/'lapackpp' to match your install logs.
libraries = ["slate", "blaspp", "lapackpp"]

extra_compile_args = ["-std=c++17", "-O3", "-fopenmp"] + [
    f for f in mpi_compile_flags if not f.startswith("-I")
]
extra_link_args = ["-fopenmp"] + [
    f for f in mpi_link_flags if not f.startswith("-l")
]

ext = Extension(
    "slate_wrapper",
    sources=["slate_wrapper.pyx", "slate.cpp"],
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
        include_path=include_dirs,
        compiler_directives={'language_level': "3"}
    ),
    zip_safe=False,
)
