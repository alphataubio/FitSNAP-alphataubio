FAIRChem Datasets via ADIOS2
============================

This guide describes the recommended workflow for using FAIRChem datasets (OMAT24, OC20, OC22, MPtrj, ODAC25, etc.) with FitSNAP through ADIOS2 binary pack (.bp) files. This approach provides efficient element filtering, parallel I/O, and optimized performance for large-scale materials datasets.

Overview
--------

The ADIOS2 workflow for FAIRChem datasets involves two steps:

1. **Conversion**: Use ``fairchem_to_adios2.py`` to convert FAIRChem LMDB datasets into element-filtered ADIOS2 .bp files
2. **Training**: Use the ADIOS2 scraper in FitSNAP to read the .bp file for potential fitting

This workflow is superior to direct LMDB access because:

- **Element filtering happens once**: The conversion tool filters configurations by allowed elements and saves only relevant data, avoiding repeated filtering during fitting
- **Optimized I/O**: ADIOS2 provides efficient parallel I/O with MPI-aware data distribution
- **Reduced memory footprint**: Filtered datasets are typically 10-100x smaller than full LMDB datasets
- **Portable data format**: .bp files are self-contained and can be easily shared or moved between systems
- **Preserves train/val splits**: FAIRChem's native train/validation splits are maintained in the .bp file

Supported FAIRChem Datasets
---------------------------

The ADIOS2 workflow supports all FAIRChem datasets that use LMDB format:

**Materials Datasets**

**Open Materials 2024 (OMAT24)**
   - **Description**: 110M+ structures across diverse inorganic material classes
   - **Systems**: Bulk crystals, surfaces, non-equilibrium structures from rattled perturbations and AIMD trajectories
   - **Download**: https://fair-chem.github.io/core/datasets/omat24.html  
   - **Paper**: https://arxiv.org/abs/2410.12771
   - **Subsets**: rattled-300, rattled-500, rattled-1000, rattled-relax, aimd-from-PBE-1000-npt, aimd-from-PBE-1000-nvt, aimd-from-PBE-3000-npt, aimd-from-PBE-3000-nvt

**Materials Project Trajectories (MPtrj)**
   - **Description**: 1.5M+ structures from Materials Project DFT relaxations
   - **Systems**: Inorganic crystals with forces, energies, and magnetic moments
   - **Download**: https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842
   - **Paper**: https://doi.org/10.1038/s42256-023-00716-3

**sAlex (Subsampled Alexandria)**
   - **Description**: Matbench-Discovery compliant subset of Alexandria dataset  
   - **Systems**: Materials trajectory data filtered for redundancy
   - **Download**: Available with OMAT24 at https://huggingface.co/datasets/facebook/OMAT24
   - **Paper**: https://arxiv.org/abs/2410.12771

**Catalysis Datasets**

**Open Catalyst 2020 (OC20)**
   - **Description**: 1.3M+ DFT calculations for catalyst discovery
   - **Systems**: Adsorbates on catalyst surfaces
   - **Download**: https://fair-chem.github.io/catalysts/datasets/oc20.html
   - **Paper**: https://arxiv.org/abs/2010.09990

**Open Catalyst 2022 (OC22)**
   - **Description**: 62k+ DFT calculations with enhanced adsorbate coverage  
   - **Systems**: Expanded adsorbate-catalyst combinations
   - **Download**: https://fair-chem.github.io/catalysts/datasets/oc22.html
   - **Paper**: https://arxiv.org/abs/2206.08917

**Direct Air Capture Datasets**

**Open Direct Air Capture 2025 (ODAC25)**
   - **Description**: 70M+ DFT calculations for CO₂, H₂O, N₂, O₂ adsorption
   - **Systems**: 15,000+ metal-organic frameworks with diverse functionalization
   - **Download**: https://fair-chem.github.io/dac/datasets/odac25.html
   - **Paper**: Available on FAIRChem site

Installation Requirements
-------------------------

The ADIOS2 workflow requires two sets of dependencies:

**For the conversion tool** (``fairchem_to_adios2.py``):

.. code-block:: bash

   # Install ADIOS2 (platform-specific, see below)
   # Install fairchem-core for reading LMDB datasets
   pip install fairchem-core

**For FitSNAP ADIOS2 scraper**:

.. code-block:: bash

   # ADIOS2 must be installed (same as above)
   # No additional Python packages required

ADIOS2 Installation (Required for Both Steps)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ADIOS2 requires both C++ libraries and Python bindings. Installation method depends on your platform:

**macOS (local development)**:

.. code-block:: bash

   # Install C++ libraries via Homebrew
   brew install adios2
   
   # Install Python bindings
   pip uninstall adios2  # Remove any existing installation
   pip install adios2

**Linux HPC clusters**:

.. code-block:: bash

   # Load ADIOS2 module (check your cluster's module system)
   module load adios2
   # or
   module load adios2/2.10.1
   
   # Install Python bindings
   pip install adios2

**Conda/Mamba (cross-platform)**:

.. code-block:: bash

   conda install -c conda-forge adios2 adios2-python
   # or
   mamba install -c conda-forge adios2 adios2-python

.. warning::
   On macOS, running ``pip install adios2`` alone **will not work**. You must first install the C++ libraries via Homebrew or Conda before installing the Python bindings.

Complete Workflow: OMAT24 Example
----------------------------------

This section demonstrates the complete workflow using the OMAT24 dataset to create an Al-Ni binary alloy potential.

Step 1: Download OMAT24 Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OMat24 provides separate train and validation archives. Download the subsets you need:

.. code-block:: bash

   # Set base URL for OMAT24 dataset
   OMAT=https://dl.fbaipublicfiles.com/opencatalystproject/data/omat
   
   # Download training data (note: different release dates for train vs val)
   wget -c -P train $OMAT/241018/omat/train/{rattled-{1000,500,300},aimd-from-PBE-{1000,3000}-{npt,nvt},rattled-relax}.tar.gz
   
   # Download validation data
   wget -c -P val $OMAT/241220/omat/val/{rattled-{1000,500,300},aimd-from-PBE-{1000,3000}-{npt,nvt},rattled-relax}.tar.gz

This downloads the following subsets for both train and validation:

- ``rattled-300``: Structures with 300K thermal perturbations
- ``rattled-500``: Structures with 500K thermal perturbations  
- ``rattled-1000``: Structures with 1000K thermal perturbations
- ``rattled-relax``: Relaxed structures from rattled configurations
- ``aimd-from-PBE-1000-npt``: AIMD trajectories at 1000K in NPT ensemble
- ``aimd-from-PBE-1000-nvt``: AIMD trajectories at 1000K in NVT ensemble
- ``aimd-from-PBE-3000-npt``: AIMD trajectories at 3000K in NPT ensemble
- ``aimd-from-PBE-3000-nvt``: AIMD trajectories at 3000K in NVT ensemble

Each archive is typically 1-10 GB compressed. The ``-c`` flag enables continuing interrupted downloads.

Step 2: Extract Archives
~~~~~~~~~~~~~~~~~~~~~~~~

Extract all downloaded archives into their respective train and val directories. Each archive creates a subdirectory containing LMDB database files:

.. code-block:: bash

   # Extract training archives
   cd train
   for file in *.tar.gz; do
       echo "Extracting $file..."
       tar -xzf "$file"
   done
   cd ..
   
   # Extract validation archives  
   cd val
   for file in *.tar.gz; do
       echo "Extracting $file..."
       tar -xzf "$file"
   done
   cd ..

After extraction, your directory structure should look like:

.. code-block:: text

   omat24/
   ├── train/
   │   ├── rattled-300/
   │   │   ├── data.mdb
   │   │   └── lock.mdb
   │   ├── rattled-500/
   │   │   ├── data.mdb
   │   │   └── lock.mdb
   │   ├── rattled-1000/
   │   │   ├── data.mdb  
   │   │   └── lock.mdb
   │   ├── rattled-relax/
   │   │   ├── data.mdb
   │   │   └── lock.mdb
   │   ├── aimd-from-PBE-1000-npt/
   │   │   ├── data.mdb
   │   │   └── lock.mdb
   │   ├── aimd-from-PBE-1000-nvt/
   │   │   ├── data.mdb
   │   │   └── lock.mdb
   │   ├── aimd-from-PBE-3000-npt/
   │   │   ├── data.mdb
   │   │   └── lock.mdb
   │   └── aimd-from-PBE-3000-nvt/
   │       ├── data.mdb
   │       └── lock.mdb
   └── val/
       ├── rattled-300/
       │   ├── data.mdb
       │   └── lock.mdb
       ├── rattled-500/
       │   ├── data.mdb
       │   └── lock.mdb
       └── ... (same structure as train/)

Each LMDB database contains thousands to millions of atomic configurations with energies, forces, and structural information.

Step 3: Convert to ADIOS2 Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``fairchem_to_adios2.py`` conversion tool to create an element-filtered .bp file. This tool:

- Reads all LMDB databases in train/ and val/ subdirectories
- Filters configurations to only those containing specified elements
- Preserves the train/validation split from FAIRChem
- Uses multiprocessing for efficient parallel conversion
- Writes a compressed, portable ADIOS2 .bp file

Basic usage for Al-Ni system:

.. code-block:: bash

   python tools/fairchem_to_adios2.py \
       --dataset omat24 \
       --elements Al Ni \
       --output omat24_AlNi.bp

Command-line options:

- ``--dataset``: Path to dataset root directory (contains train/ and val/ subdirectories)
- ``--elements``: Space-separated list of allowed element symbols (e.g., ``Al Ni`` or ``C H O N``)
- ``--output``: Output ADIOS2 .bp filename
- ``--workers``: Number of parallel worker processes (default: all available CPUs)

Example with explicit worker count:

.. code-block:: bash

   python tools/fairchem_to_adios2.py \
       --dataset omat24 \
       --elements Al Ni \
       --output omat24_AlNi.bp \
       --workers 32

The conversion process displays progress for each LMDB database:

.. code-block:: text

   Allowed elements: Al, Ni
   Using 32 parallel workers (all CPUs)
   
   Processing train subset with 8 LMDB databases
     rattled-300: 100%|██████████| 150000/150000 [00:45<00:00, 3333 config/s]
       rattled-300: kept 12543, filtered 137457
     rattled-500: 100%|██████████| 150000/150000 [00:42<00:00, 3571 config/s]
       rattled-500: kept 11234, filtered 138766
     ...
   
   Collected 95432 training configurations
   Collected 10234 validation configurations
   
   Total configurations: 105666
   
   Writing 105666 configurations to omat24_AlNi.bp
     Flattening arrays...
     Concatenating arrays...
     Total atoms across all configs: 45678912
     Has forces: True
     Has stress: True
     Writing to ADIOS2...
   
   Successfully wrote omat24_AlNi.bp
     File contains 105666 configurations
     Elements: Al, Ni

The conversion typically reduces file size by 10-100x compared to the full LMDB dataset, as only relevant elements are retained.

Step 4: Configure FitSNAP Input File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a FitSNAP input file that uses the ADIOS2 scraper to read the .bp file:

.. code-block:: ini

   [SCRAPER]
   scraper = ADIOS2
   
   [PATH]
   dataPath = omat24_AlNi.bp
   
   [GROUPS]
   group_sections = name eweight fweight vweight
   group_types = str float float float
   smartweights = 1
   rattled-300             = 1.0  100.0  10.0
   rattled-500             = 1.0  100.0  10.0
   rattled-1000            = 1.0  100.0  10.0
   rattled-relax           = 1.0  100.0  10.0
   aimd-from-PBE-1000-npt  = 1.0  100.0  10.0
   aimd-from-PBE-1000-nvt  = 1.0  100.0  10.0  
   aimd-from-PBE-3000-npt  = 1.0  100.0  10.0
   aimd-from-PBE-3000-nvt  = 1.0  100.0  10.0

Key configuration details:

**[SCRAPER] section**:
   - ``scraper = ADIOS2``: Use the ADIOS2 scraper (not FAIRCHEM)
   - ``max_configs_per_rank``: Optional limit on configurations per MPI rank for testing (omit for production)

**[PATH] section**:
   - ``dataPath``: Path to the .bp file created in Step 3

**[GROUPS] section**:
   - Group names **must exactly match** the LMDB subdirectory names from the original dataset
   - These names are stored in the .bp file's ``unique_group_names`` attribute
   - Each group can have different energy, force, and virial weights
   - ``smartweights = 1`` enables automatic weight scaling based on group size
   - **Important**: Do NOT specify ``training_size`` or ``testing_size`` - the train/val split is already encoded in the .bp file via ``test_bool`` flags

Step 5: Run FitSNAP
~~~~~~~~~~~~~~~~~~~

Run FitSNAP with MPI parallelization:

.. code-block:: bash

   mpirun -np 192 python -m fitsnap3 AlNi_PYACE_SLATE_ARD.in --overwrite

The ADIOS2 scraper will:

1. Read metadata from the .bp file (element map, group names, configuration counts)
2. Distribute configurations across MPI ranks based on ``test_bool`` flags
3. Load positions, forces, energies, and lattice parameters for each configuration  
4. Assign group-specific weights from the [GROUPS] section
5. Pass configurations to the descriptor calculator

Example output:

.. code-block:: text

   ADIOS2 Scraper: Reading omat24_AlNi.bp
   ADIOS2 Scraper: Found 105666 configurations
   ADIOS2 Scraper: Elements: Al, Ni
   ADIOS2 Scraper: Groups: rattled-300, rattled-500, rattled-1000, ...
   ADIOS2 Scraper: 95432 training, 10234 validation configurations
   ADIOS2 Scraper: Distributing configs across 192 MPI ranks
   ...

ADIOS2 .bp File Structure
--------------------------

Understanding the .bp file structure helps with debugging and advanced usage.

Metadata (Attributes)
~~~~~~~~~~~~~~~~~~~~~

The .bp file stores metadata as ADIOS2 attributes:

- ``nconfigs`` (int): Total number of configurations
- ``element_map`` (string): Comma-separated list of allowed elements (e.g., "Al,Ni")
- ``has_forces`` (int): 1 if force data is available, 0 otherwise
- ``has_stress`` (int): 1 if stress tensor data is available, 0 otherwise  
- ``unique_group_names`` (string): Pipe-delimited list of group names (e.g., "rattled-300|rattled-500|...")

Per-Configuration Arrays
~~~~~~~~~~~~~~~~~~~~~~~~

Fixed-size arrays with length ``nconfigs``:

- ``NumAtoms[nconfigs]`` (int32): Number of atoms in each configuration
- ``Energy[nconfigs]`` (float64): Total energy (eV)
- ``test_bool[nconfigs]`` (int32): 0 for training data, 1 for validation data
- ``GroupIndices[nconfigs]`` (int32): Index into ``unique_group_names`` for each configuration
- ``Lattice[nconfigs, 3, 3]`` (float64): Lattice vectors (Å) as 3x3 matrices
- ``Stress[nconfigs, 3, 3]`` (float64): Stress tensors (eV/Å³) as 3x3 matrices (if ``has_stress=1``)

Variable-Length Flattened Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Atomic data is stored in flattened arrays with offset indexing:

- ``PositionOffsets[nconfigs]`` (int64): Starting index in ``PositionsFlat`` for each configuration
- ``PositionsFlat[total_atoms, 3]`` (float64): Flattened atomic positions (Å)
- ``AtomTypesFlat[total_atoms]`` (int32): Flattened atom type indices (map to ``element_map``)
- ``ForcesFlat[total_atoms, 3]`` (float64): Flattened atomic forces (eV/Å) (if ``has_forces=1``)

Where ``total_atoms = sum(NumAtoms)`` across all configurations.

Example: To extract positions for configuration ``i``:

.. code-block:: python

   start = PositionOffsets[i]
   end = PositionOffsets[i] + NumAtoms[i]
   positions = PositionsFlat[start:end, :].reshape(NumAtoms[i], 3)

Configuration Examples
----------------------

Basic PYACE Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Minimal configuration for Al-Ni PYACE potential:

.. code-block:: ini

   [PYACE]
   elements = Al Ni
   # ... (PYACE potential parameters)
   
   [CALCULATOR]
   calculator = LAMMPSPYACE
   energy = 1
   force = 1
   stress = 0
   
   [SOLVER]
   solver = SLATE
   compute_testerrs = 1
   
   [SCRAPER]
   scraper = ADIOS2
   
   [PATH]
   dataPath = omat24_AlNi.bp
   
   [GROUPS]
   group_sections = name eweight fweight vweight
   group_types = str float float float
   smartweights = 1
   rattled-300 = 1.0 100.0 10.0
   rattled-500 = 1.0 100.0 10.0

Testing with Limited Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For testing or development, limit the number of configurations per MPI rank:

.. code-block:: ini

   [SCRAPER]
   scraper = ADIOS2
   max_configs_per_rank = 100  # Load only 100 configs per rank for testing
   
   [PATH]
   dataPath = omat24_AlNi.bp

This is useful for:

- Quick validation of input file syntax
- Testing descriptor calculations on small subsets
- Debugging without full dataset overhead
- Memory profiling with controlled data size

**Important**: Remove or comment out ``max_configs_per_rank`` for production runs to use all available data.

Multi-Element Systems
~~~~~~~~~~~~~~~~~~~~~

For ternary or higher-order systems, specify all elements during conversion:

.. code-block:: bash

   python tools/fairchem_to_adios2.py \
       --dataset omat24 \
       --elements Al Ni Cu \
       --output omat24_AlNiCu.bp

Then use the same elements in your FitSNAP input:

.. code-block:: ini

   [PYACE]
   elements = Al Ni Cu
   
   [SCRAPER]
   scraper = ADIOS2
   
   [PATH]
   dataPath = omat24_AlNiCu.bp

Performance Considerations
--------------------------

Conversion Performance
~~~~~~~~~~~~~~~~~~~~~~

The ``fairchem_to_adios2.py`` tool uses multiprocessing for efficient parallel conversion:

**Worker Processes**:
   - Default: Uses all available CPU cores (``multiprocessing.cpu_count()``)
   - Each worker process maintains its own LMDB connection
   - Work is distributed in chunks across workers for load balancing
   - Progress bars show real-time throughput (configs/second)

**Typical Performance**:
   - Single worker: 500-1500 configs/second (depending on config size)
   - 32 workers: 10,000-30,000 configs/second aggregate
   - Conversion time for OMAT24 Al-Ni (~100k configs): 5-15 minutes on modern workstation
   - Memory usage: ~2-4 GB per worker process

**Optimization Tips**:
   - Use ``--workers`` flag to control parallelism (e.g., ``--workers 16`` for 16 cores)
   - On shared systems, limit workers to avoid overwhelming the system
   - Conversion is I/O-bound for small configs, CPU-bound for large configs
   - LMDB databases benefit from being on local storage (avoid NFS during conversion)

FitSNAP Performance
~~~~~~~~~~~~~~~~~~~

**ADIOS2 I/O**:
   - ADIOS2 provides MPI-aware parallel I/O
   - Each MPI rank reads only its assigned subset of configurations
   - Binary pack (.bp) format enables zero-copy data transfer
   - Significantly faster than text-based formats (JSON, XYZ)

**Memory Management**:
   - ``max_configs_per_rank`` limits configurations loaded per rank
   - Without this option, FitSNAP distributes all configs across available ranks
   - For production: omit ``max_configs_per_rank`` to use full dataset
   - For testing: use ``max_configs_per_rank = 100`` to quickly validate setup

**MPI Scaling**:
   - ADIOS2 scraper scales efficiently to hundreds of MPI ranks
   - Example: 105k configs on 192 ranks ≈ 550 configs/rank
   - Load balancing is automatic based on configuration distribution
   - Train/validation split preserved across all ranks

**File Size Reduction**:
   - Element filtering reduces dataset size by 10-100x
   - Example: OMAT24 full dataset ~500 GB LMDB → Al-Ni subset ~5 GB .bp
   - Smaller files enable faster I/O, easier data management, and portability

Training/Validation Split Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ADIOS2 workflow preserves FAIRChem's native train/val split:

**How it works**:
   1. FAIRChem datasets provide separate train/ and val/ archives
   2. ``fairchem_to_adios2.py`` reads both and sets ``test_bool`` flag:
      - ``test_bool = 0`` for configs from train/ directories
      - ``test_bool = 1`` for configs from val/ directories
   3. FitSNAP ADIOS2 scraper uses ``test_bool`` to assign training/testing status
   4. No random splitting - exact same split as FAIRChem publications

**Key differences from other scrapers**:
   - Legacy scrapers (JSON, XYZ): user specifies ``training_size`` and ``testing_size`` percentages
   - ADIOS2 scraper: ``training_size`` and ``testing_size`` are **ignored** - split is predetermined
   - This ensures reproducibility and consistency with FAIRChem benchmarks

**Benefits**:
   - Reproducible results across different research groups
   - Fair comparison with published FAIRChem baselines
   - No randomness in data splits
   - Validation set represents out-of-sample performance accurately

Stress Tensor Support
---------------------

.. warning::
   Stress tensor fitting is **experimental** and currently under active development. Use with caution.

The ADIOS2 workflow supports stress tensor data:

**Conversion Tool**:
   - Extracts stress tensors from LMDB if available
   - Converts from Voigt notation (6-component) to 3x3 symmetric tensor
   - Stores as ``Stress[nconfigs, 3, 3]`` array in .bp file
   - Sets ``has_stress`` attribute to indicate availability

**FitSNAP Scraper**:
   - Reads stress tensors if ``has_stress = 1`` in .bp file
   - Passes stress data to descriptor calculator
   - Honors ``vweight`` (virial weight) in [GROUPS] section

**Current Limitations**:
   - Stress fitting not fully validated for all descriptor types
   - May require careful weight tuning (``vweight``) to balance with forces/energies
   - Not recommended for production potentials until further testing
   - Set ``stress = 0`` in [CALCULATOR] section to disable stress fitting



Error Handling and Debugging
----------------------------

Conversion Tool Errors
~~~~~~~~~~~~~~~~~~~~~~

**Missing Dependencies**:

If ADIOS2 is not installed:

.. code-block:: text

   ================================================================================
   ERROR: Failed to import ADIOS2
   ================================================================================
   
   Import error: No module named 'adios2'
   
   ADIOS2 requires both the C++ libraries AND Python bindings.
   ...

Solution: Follow ADIOS2 installation instructions above.

If fairchem-core is not installed:

.. code-block:: text

   ================================================================================  
   ERROR: Failed to import fairchem
   ================================================================================
   
   Import error: No module named 'fairchem'
   
   Install: pip install fairchem-core

Solution: ``pip install fairchem-core``

**Dataset Path Errors**:

.. code-block:: text

   ERROR: Dataset path does not exist: /path/to/omat24

Solution: Verify the dataset path and ensure train/ and val/ subdirectories exist.

**Empty Results**:

.. code-block:: text

   Warning: No LMDB databases found in /path/to/omat24/train
   ERROR: No configurations collected!

Solution:
   - Ensure archives are extracted (Step 2 above)
   - Check that extracted directories contain data.mdb files
   - Verify element filtering is not too restrictive

FitSNAP ADIOS2 Scraper Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Group Name Mismatch**:

If group names in FitSNAP input don't match .bp file:

.. code-block:: text

   Warning: Group 'rattled_300' not found in .bp file
   Available groups: rattled-300, rattled-500, ...

Solution: Use exact group names from conversion (with hyphens, not underscores).

**Element Mismatch**:

If FitSNAP requests elements not in .bp file:

.. code-block:: text

   ERROR: Element 'Cu' not found in .bp file
   Available elements: Al, Ni

Solution: Recreate .bp file with correct elements or adjust FitSNAP input.

**Memory Issues**:

If MPI ranks run out of memory:

.. code-block:: text

   MemoryError: Unable to allocate array

Solution: Add ``max_configs_per_rank`` to [SCRAPER] section or increase MPI rank count.

Troubleshooting Guide
---------------------

Conversion is Very Slow
~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Conversion takes hours for moderate datasets

**Possible causes**:
   1. LMDB on network filesystem (NFS, Lustre)
   2. Insufficient worker processes
   3. Very large configurations (>1000 atoms)

**Solutions**:
   - Copy LMDB databases to local storage before conversion
   - Increase ``--workers`` to match CPU count
   - Monitor CPU usage to identify bottlenecks (I/O vs compute)

Converted File is Too Large
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: .bp file is unexpectedly large (>100 GB)

**Possible causes**:
   1. Too many elements selected (less filtering)
   2. Including elements with many configs
   3. Not using element filtering effectively

**Solutions**:
   - Review element selection - use only needed elements
   - Check filtered counts in conversion output
   - Consider creating separate .bp files for different element combinations

FitSNAP Can't Find Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: FitSNAP reports 0 configurations loaded

**Debugging steps**:
   1. Verify .bp file exists at path in [PATH] section
   2. Check group names exactly match (case-sensitive, hyphens vs underscores)
   3. Ensure elements in [PYACE]/[BISPECTRUM] match .bp file
   4. Use ``bpls`` tool to inspect .bp file contents:

.. code-block:: bash

   # List all variables in .bp file
   bpls omat24_AlNi.bp
   
   # Show attributes
   bpls -a omat24_AlNi.bp

Inconsistent Train/Val Split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Training/testing sizes don't match expectations

**Remember**:
   - ADIOS2 workflow uses **fixed** train/val split from FAIRChem
   - ``training_size`` and ``testing_size`` in input file are **ignored**
   - Split is determined during conversion by train/ vs val/ directories
   - Check conversion output for actual train/val counts

MPI Hangs or Deadlocks
~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: FitSNAP hangs during ADIOS2 reading

**Possible causes**:
   1. ADIOS2 version mismatch between conversion and FitSNAP
   2. Filesystem locking issues
   3. Inconsistent MPI configurations

**Solutions**:
   - Use same ADIOS2 version for conversion and FitSNAP
   - Ensure .bp file is on parallel filesystem (Lustre, GPFS) for multinode runs
   - Test with single rank first: ``mpirun -np 1 python -m fitsnap3 ...``

Advanced Usage
--------------

Inspecting .bp Files
~~~~~~~~~~~~~~~~~~~~

Use ADIOS2's ``bpls`` command-line tool to inspect .bp files:

.. code-block:: bash

   # Install bpls (comes with ADIOS2)
   # List all variables and attributes
   bpls -la omat24_AlNi.bp
   
   # Show specific variable details
   bpls -l omat24_AlNi.bp -d Energy -n 10
   
   # Dump attribute values
   bpls -a omat24_AlNi.bp

Programmatic Access
~~~~~~~~~~~~~~~~~~~

Read .bp files directly in Python for custom analysis:

.. code-block:: python

   from adios2 import Stream
   import numpy as np
   
   with Stream('omat24_AlNi.bp', 'r') as s:
       # Read metadata
       nconfigs = s.read_attribute('nconfigs')
       element_map = s.read_attribute('element_map').split(',')
       
       # Read arrays
       for step in s:
           num_atoms = step.read('NumAtoms')
           energies = step.read('Energy')
           positions_flat = step.read('PositionsFlat')
           # ...

Creating Subsets
~~~~~~~~~~~~~~~~

For debugging or specialized workflows, create smaller .bp files:

.. code-block:: bash

   # Option 1: Limit during conversion
   python tools/fairchem_to_adios2.py \
       --dataset omat24/train/rattled-300 \
       --elements Al Ni \
       --output small_test.bp
   
   # Then use max_configs_per_rank in FitSNAP
   [SCRAPER]
   max_configs_per_rank = 50

Merging Multiple Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~

To combine multiple FAIRChem datasets (e.g., OMAT24 + OC20):

1. Convert each dataset separately:

.. code-block:: bash

   python tools/fairchem_to_adios2.py \
       --dataset omat24 --elements Al Ni O --output omat24_AlNiO.bp
   
   python tools/fairchem_to_adios2.py \
       --dataset oc20 --elements Al Ni O --output oc20_AlNiO.bp

2. Currently, FitSNAP processes one .bp file per run. To use multiple datasets, combine them during conversion by placing all desired LMDB databases in a single dataset directory structure before running the conversion tool.

Related Documentation
---------------------

- **FAIRChem Project**: https://github.com/FAIR-Chem/fairchem
- **OMAT24 Dataset**: https://fair-chem.github.io/core/datasets/omat24.html
- **OMAT24 on HuggingFace**: https://huggingface.co/datasets/facebook/OMAT24
- **ADIOS2 Documentation**: https://adios2.readthedocs.io/
- **LMDB Documentation**: https://lmdb.readthedocs.io/
- **FitSNAP Scrapers**: :doc:`/Lib/scraper`
- **FitSNAP MPI Usage**: :doc:`/Run/parallel`
- **PYACE Calculator**: :doc:`/Lib/calculators`

Complete Working Example
------------------------

This example demonstrates the full workflow from download to training for an Al-Ni PYACE potential using OMAT24 data.

**1. Download and extract data** (one-time setup):

.. code-block:: bash

   # Create working directory
   mkdir -p scratch/omat24
   cd scratch/omat24
   
   # Download OMAT24 subsets
   OMAT=https://dl.fbaipublicfiles.com/opencatalystproject/data/omat
   wget -c -P train $OMAT/241018/omat/train/{rattled-{1000,500,300},aimd-from-PBE-{1000,3000}-{npt,nvt},rattled-relax}.tar.gz
   wget -c -P val $OMAT/241220/omat/val/{rattled-{1000,500,300},aimd-from-PBE-{1000,3000}-{npt,nvt},rattled-relax}.tar.gz
   
   # Extract archives
   cd train && for f in *.tar.gz; do tar -xzf "$f"; done && cd ..
   cd val && for f in *.tar.gz; do tar -xzf "$f"; done && cd ..
   cd ../..

**2. Convert to ADIOS2** (one-time per element combination):

.. code-block:: bash

   python tools/fairchem_to_adios2.py \
       --dataset scratch/omat24 \
       --elements Al Ni \
       --output omat24_AlNi.bp \
       --workers 32

**3. Create FitSNAP input file** (``AlNi_PYACE.in``):

.. code-block:: ini

   [PYACE]
   elements = Al Ni
   bzeroflag = 0
   embeddings = {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1,1,1,0.5], "ndensity": 2}}
   bonds = {
       "AlAl": {"radbase": "ChebExpCos", "radparameters": [0.342], "rcut": 6.832, "dcut": 0.01},
       "AlNi": {"radbase": "ChebExpCos", "radparameters": [0.318], "rcut": 6.356, "dcut": 0.01},
       "NiAl": {"radbase": "ChebExpCos", "radparameters": [0.318], "rcut": 6.356, "dcut": 0.01},
       "NiNi": {"radbase": "ChebExpCos", "radparameters": [0.281], "rcut": 5.610, "dcut": 0.01}
   }
   functions = {"BINARY": {"lmax_by_orders": [0,1,2], "nradmax_by_orders": [8,4,2]}}
   
   [CALCULATOR]
   calculator = LAMMPSPYACE
   energy = 1
   force = 1
   stress = 0
   
   [SOLVER]
   solver = SLATE
   compute_testerrs = 1
   
   [SCRAPER]
   scraper = ADIOS2
   
   [PATH]
   dataPath = omat24_AlNi.bp
   
   [GROUPS]
   group_sections = name eweight fweight vweight
   group_types = str float float float
   smartweights = 1
   rattled-300             = 1.0  100.0  10.0
   rattled-500             = 1.0  100.0  10.0
   rattled-1000            = 1.0  100.0  10.0
   rattled-relax           = 1.0  100.0  10.0
   aimd-from-PBE-1000-npt  = 1.0  100.0  10.0
   aimd-from-PBE-1000-nvt  = 1.0  100.0  10.0
   aimd-from-PBE-3000-npt  = 1.0  100.0  10.0
   aimd-from-PBE-3000-nvt  = 1.0  100.0  10.0
   
   [OUTFILE]
   output_style = PACE
   metrics = AlNi_metrics.md
   potential = AlNi_potential
   validation = 1

**4. Run FitSNAP**:

.. code-block:: bash

   # Test run with limited configs
   mpirun -np 4 python -m fitsnap3 AlNi_PYACE.in --overwrite
   
   # Production run on HPC cluster
   mpirun -np 192 python -m fitsnap3 AlNi_PYACE.in --overwrite

Summary
-------

The ADIOS2 workflow for FAIRChem datasets provides:

1. **Efficiency**: One-time element filtering reduces dataset size by 10-100x
2. **Performance**: Parallel conversion and MPI-aware I/O enable large-scale fitting
3. **Reproducibility**: Preserves FAIRChem's train/val splits for consistent benchmarking
4. **Portability**: Self-contained .bp files are easily shared and archived
5. **Flexibility**: Supports all FAIRChem datasets (OMAT24, OC20, OC22, MPtrj, ODAC25)

**Recommended workflow**:
   - Use ADIOS2 for any FAIRChem dataset with >10k configurations
   - Create separate .bp files for different element combinations
   - Test with ``max_configs_per_rank`` before production runs
   - Archive .bp files for reproducibility and sharing

**When to use direct LMDB access** (legacy workflow):
   - Small datasets (<10k configs)
   - Rapid prototyping without conversion overhead
   - Custom filtering logic not supported by conversion tool

For questions or issues, please open an issue on the FitSNAP GitHub repository.

Citations
---------

If you use FAIRChem datasets in your research, please cite the appropriate papers:

**OMAT24:**
   Barroso-Luque, L., Shuaibi, M., Fu, X. et al. Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models. *arXiv preprint* 2024, arXiv:2410.12771.

**OC20:**
   Chanussot, L., Das, A., Goyal, S. et al. Open Catalyst 2020 (OC20) Dataset and Community Challenges for Catalysis. *ACS Catal.* 2021, 11, 6059-6072.

**OC22:**  
   Tran, R., Lan, J., Shuaibi, M. et al. The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide Electrocatalysis. *ACS Catal.* 2023, 13, 3066-3084.

**MPtrj:**
   Deng, B., Zhong, P., Jun, K. et al. CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling. *Nat. Mach. Intell.* 2023, 5, 1031-1041.

**ODAC25:**
   Sriram, A., Choi, S., Yu, X. et al. Open Direct Air Capture datasets for CO₂ capture in metal-organic frameworks. Available on FAIRChem documentation.

**Alexandria/sAlex:**
   Schmidt, J., Hoffmann, N., Wang, H.-C. et al. Machine-Learning-Assisted Determination of the Global Zero-Temperature Phase Diagram of Materials. *Adv. Mater.* 2023, 35, 2210788.

**ADIOS2:**
   Godoy, W.F., Podhorszki, N., Wang, R. et al. ADIOS 2: The Adaptable Input Output System. A framework for high-performance data management. *SoftwareX* 2020, 12, 100561.
