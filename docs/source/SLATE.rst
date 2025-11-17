SLATE Multinode Solver
======================

The SLATE solver provides distributed ridge regression and Automatic Relevance Determination (ARD) regression for multi-node HPC systems using SLATE (Software for Linear Algebra Targeting Exascale). SLATE is the successor to ScalaPACK written in C++17 for modern CPU/GPU architectures.

Overview
--------

SLATE enables FitSNAP to scale linear regression to many compute nodes by leveraging distributed linear algebra operations. The solver supports two regression methods:

- **RIDGE**: Standard ridge regression (L2 regularization) for stable fitting with all features
- **ARD**: Automatic Relevance Determination for automatic feature selection and sparsity

Both methods use SLATE's distributed matrix operations to solve weighted least squares problems across multiple nodes efficiently.

.. note::
   This solver is part of FitSNAP Pull Request #278 and requires SLATE library installation.

Mathematical Background
-----------------------

Weighted Least Squares Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given:

- Design matrix :math:`\mathbf{X} \in \mathbb{R}^{m \times n}` (descriptor matrix)
- Target vector :math:`\mathbf{y} \in \mathbb{R}^{m}` (energies, forces, stresses)
- Weight vector :math:`\mathbf{w} \in \mathbb{R}^{m}` (per-sample weights)
- Coefficient vector :math:`\boldsymbol{\beta} \in \mathbb{R}^{n}` (unknown parameters)

The weighted least squares objective is:

.. math::

   \min_{\boldsymbol{\beta}} \quad \|\mathbf{W}^{1/2}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})\|_2^2

where :math:`\mathbf{W} = \text{diag}(\mathbf{w})`.

RIDGE Regression
~~~~~~~~~~~~~~~~

Ridge regression adds L2 regularization to prevent overfitting:

.. math::

   \min_{\boldsymbol{\beta}} \quad \|\mathbf{W}^{1/2}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})\|_2^2 + \alpha \|\boldsymbol{\beta}\|_2^2

The solution is:

.. math::

   \boldsymbol{\beta} = (\mathbf{X}^T \mathbf{W} \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{y}

where :math:`\alpha > 0` is the regularization parameter.

**SLATE Implementation:**

The SLATE solver uses augmented QR decomposition for numerical stability. The augmented system is:

.. math::

   \begin{bmatrix}
   \mathbf{W}^{1/2}\mathbf{X} \\
   \sqrt{\alpha} \mathbf{I}
   \end{bmatrix}
   \boldsymbol{\beta} =
   \begin{bmatrix}
   \mathbf{W}^{1/2}\mathbf{y} \\
   \mathbf{0}
   \end{bmatrix}

This system is solved using SLATE's distributed least squares solver (``slate::least_squares_solve``), which internally performs QR factorization across multiple nodes.

ARD Regression
~~~~~~~~~~~~~~

Automatic Relevance Determination (ARD) is a Bayesian approach to automatic feature selection. ARD assigns individual precision (inverse variance) parameters to each feature, allowing irrelevant features to be automatically pruned during fitting.

The ARD objective incorporates feature-specific precisions :math:`\boldsymbol{\lambda} = (\lambda_1, \ldots, \lambda_n)`:

.. math::

   \min_{\boldsymbol{\beta}} \quad \alpha \|\mathbf{W}^{1/2}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})\|_2^2 + \sum_{i=1}^{n} \lambda_i \beta_i^2

**Iterative Algorithm:**

ARD iteratively updates four sets of parameters (following scikit-learn's ARDRegression implementation):

1. **Posterior covariance diagonal:**

   .. math::

      \sigma_{ii} = \text{diag}\left[(\alpha \mathbf{X}^T \mathbf{W} \mathbf{X} + \text{diag}(\boldsymbol{\lambda}))^{-1}\right]

2. **Coefficient estimates:**

   .. math::

      \boldsymbol{\beta} = \alpha \boldsymbol{\Sigma} \mathbf{X}^T \mathbf{W} \mathbf{y}

   where :math:`\boldsymbol{\Sigma} = \text{diag}(\boldsymbol{\sigma})`.

3. **Feature precisions (lambda):**

   .. math::

      \lambda_i = \frac{\gamma_i + 2\lambda_1}{(\beta_i)^2 + 2\lambda_2}

   where :math:`\gamma_i = 1 - \lambda_i \sigma_{ii}` is the "feature usage" (effective degrees of freedom).

4. **Noise precision (alpha):**

   .. math::

      \alpha = \frac{m_{\text{train}} - \sum_i \gamma_i + 2\alpha_1}{\text{SSE} + 2\alpha_2}

   where :math:`\text{SSE} = \|\mathbf{W}^{1/2}(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})\|_2^2`.

The hyperparameters :math:`(\alpha_1, \alpha_2, \lambda_1, \lambda_2)` are computed adaptively from data variance (see Configuration section).

**Feature Pruning:**

Features are pruned when their precision becomes too large (feature is irrelevant):

- **Lambda pruning**: Remove features where :math:`\lambda_i > \text{threshold\_lambda}`
- **Gamma pruning** (experimental): Remove features where :math:`\gamma_i < \text{threshold\_gamma}`

Installation Requirements
--------------------------

SLATE Library
~~~~~~~~~~~~~

SLATE requires installation of the C++ library and Python bindings.

**From source** (recommended for HPC):

.. code-block:: bash

   # Clone SLATE
   git clone https://github.com/icl-utk-edu/slate.git
   cd slate
   
   # Configure with CMake
   mkdir build && cd build
   cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local \
         -DCMAKE_CXX_COMPILER=mpicxx \
         -Dbuild_tests=OFF \
         ..
   
   # Build and install
   make -j8
   make install

**Module load** (on HPC clusters):

.. code-block:: bash

   module load slate  # Check your cluster's module system
   module load slate/2023.11.05  # Specific version

SLATE Solver Wrapper
~~~~~~~~~~~~~~~~~~~~

Build the FitSNAP SLATE wrapper:

.. code-block:: bash

   cd fitsnap3lib/lib/slate_solver
   pip install -e .

This compiles the Cython wrapper (``slate_wrapper.pyx``) that interfaces Python with SLATE's C++ library.

**Verification:**

.. code-block:: bash

   python -c "from slate_wrapper import slate_ridge_augmented_qr_cython, slate_ard_update_cython; print('SLATE wrapper installed successfully')"

Configuration
-------------

Basic RIDGE Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimal configuration for ridge regression:

.. code-block:: ini

   [SOLVER]
   solver = SLATE
   compute_testerrs = 1
   
   [SLATE]
   method = RIDGE
   alpha = 1e-6

**Parameters:**

- ``method``: Regression method (``RIDGE`` or ``ARD``)
- ``alpha``: Regularization parameter (default: ``1e-6``)

  - Smaller ``alpha`` → less regularization, better training fit
  - Larger ``alpha`` → more regularization, more stable coefficients
  - Typical range: ``1e-8`` to ``1e-4``

Basic ARD Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Minimal configuration for ARD regression:

.. code-block:: ini

   [SOLVER]
   solver = SLATE
   compute_testerrs = 1
   
   [SLATE]
   method = ARD
   max_iter = 100
   scap = 1e-4
   scai = 1e-4

**Parameters:**

- ``max_iter``: Maximum number of ARD iterations (default: ``10``)
- ``scap``: Scaling factor for noise precision hyperparameters :math:`\alpha_1, \alpha_2` (default: ``1e-3``)
- ``scai``: Scaling factor for feature precision hyperparameters :math:`\lambda_1, \lambda_2` (default: ``1e-3``)

ARD Advanced Options
~~~~~~~~~~~~~~~~~~~~

Full ARD configuration with all options:

.. code-block:: ini

   [SLATE]
   method = ARD
   
   # Iteration control
   max_iter = 100
   rtol = 1e-3
   atol = 1e-6
   
   # Hyperparameter mode
   directmethod = 0
   
   # Adaptive hyperparameters (directmethod = 0, recommended)
   scap = 1e-4
   scai = 1e-4
   
   # Direct hyperparameters (directmethod = 1)
   alphabig = 1.0e-12
   alphasmall = 1.0e-14
   lambdabig = 1.0e-6
   lambdasmall = 1.0e-6
   
   # Pruning control
   pruning_method = lambda
   threshold_lambda = 1e4
   threshold_gamma = 0.1
   logcut = 0.3

**Iteration Control:**

- ``max_iter``: Maximum iterations (default: ``10``)
- ``rtol``: Relative tolerance for coefficient convergence (default: ``1e-3``)
- ``atol``: Absolute tolerance for coefficient convergence (default: ``1e-6``)

**Hyperparameter Modes:**

- ``directmethod``: Hyperparameter specification mode (default: ``0``)

  - ``0`` (recommended): Adaptive scaling using ``scap`` and ``scai``
  - ``1``: Direct specification using ``alphabig``, ``alphasmall``, ``lambdabig``, ``lambdasmall``

**Adaptive Hyperparameters** (``directmethod = 0``):

The hyperparameters are computed from data variance:

.. math::

   \begin{aligned}
   a_p &= \frac{1}{\text{Var}(\mathbf{y})} \\
   \alpha_1 = \alpha_2 &= \text{scap} \times a_p \\
   \lambda_1 = \lambda_2 &= \text{scai} \times a_p
   \end{aligned}

- ``scap``: Scaling for noise precision (default: ``1e-3``)
- ``scai``: Scaling for feature precision (default: ``1e-3``)

**Direct Hyperparameters** (``directmethod = 1``):

Manually specify all hyperparameters:

- ``alphabig``: :math:`\alpha_1` (default: ``1.0e-12``)
- ``alphasmall``: :math:`\alpha_2` (default: ``1.0e-14``)
- ``lambdabig``: :math:`\lambda_1` (default: ``1.0e-6``)
- ``lambdasmall``: :math:`\lambda_2` (default: ``1.0e-6``)

**Pruning Control:**

- ``pruning_method``: Feature pruning strategy (default: ``lambda``)

  - ``lambda``: Remove features where :math:`\lambda_i >` ``threshold_lambda``
  - ``gamma``: Remove features where :math:`\gamma_i <` ``threshold_gamma`` (experimental)

- ``threshold_lambda``: Lambda threshold for pruning (default: auto-computed)

  - If ``0``, auto-computed as :math:`10^{\lfloor \log_{10}(a_p) \rfloor + \text{logcut}}`
  - Typical values: ``1e3`` to ``1e6``

- ``threshold_gamma``: Gamma threshold for pruning (default: ``0.1``)

  - Gamma represents effective feature usage in [0, 1] range
  - Features with :math:`\gamma_i < 0.1` contribute less than 10% to predictions

- ``logcut``: Offset for auto-computing ``threshold_lambda`` (default: ``0.3``)

Stopping Criteria
~~~~~~~~~~~~~~~~~

ARD iterations stop when any of these conditions is met:

1. **Maximum iterations**: ``iteration > max_iter``
2. **Relative convergence**: :math:`\|\boldsymbol{\beta}^{(k)} - \boldsymbol{\beta}^{(k-1)}\| / \|\boldsymbol{\beta}^{(k-1)}\| <` ``rtol``
3. **Absolute convergence**: :math:`\|\boldsymbol{\beta}^{(k)} - \boldsymbol{\beta}^{(k-1)}\| <` ``atol``
4. **SIGUSR1 signal**: Allows graceful early stopping on HPC systems
5. **SLURM time limit**: Stops if remaining time < 2× last iteration time

To trigger SIGUSR1 stopping:

.. code-block:: bash

   # Find process ID
   squeue -u $USER
   
   # Send signal to main FitSNAP process
   scancel --signal=SIGUSR1 <job_id>

Validation Output
-----------------

When ``validation = 1`` in the ``[OUTFILE]`` section, SLATE creates comprehensive validation reports.

Validation Notebook
~~~~~~~~~~~~~~~~~~~

SLATE automatically generates a Jupyter notebook with:

1. **Configuration summary**: Complete FitSNAP input file settings
2. **SLURM environment**: Job parameters (nodes, tasks, memory, time)
3. **Error analysis tables**: MAE, RMSE, R² for all groups

   - Separate tables for Energy, Force, Stress
   - Weighted and unweighted metrics
   - Training and validation split side-by-side
   - Groups sorted by validation RMSE (descending)

4. **Scatterplots**: Predictions vs. ground truth for each row type
5. **ARD-specific plots** (only for ``method = ARD``):

   - Gamma/Lambda heatmaps organized by basis function rank
   - Gamma/Lambda histograms per iteration

**Example notebook location:**

.. code-block:: bash

   # If metrics = AlNi_PYACE_SLATE_ARD.md
   # Notebook created: AlNi_PYACE_SLATE_ARD.ipynb
   jupyter notebook AlNi_PYACE_SLATE_ARD.ipynb

Error Analysis Tables
~~~~~~~~~~~~~~~~~~~~~

Tables display metrics for each group:

.. list-table::
   :header-rows: 2
   :widths: 24 5 11 11 11 5 11 11 11

   * - Group
     - 
     - 
     - Training
     - 
     - 
     - 
     - Validation
     - 
   * - 
     - N
     - MAE
     - RMSE
     - R²
     - N
     - MAE
     - RMSE (↓)
     - R²
   * - **ALL**
     - 12543
     - 0.001234
     - 0.002456
     - 0.998765
     - 1234
     - 0.001345
     - 0.002567
     - 0.998654
   * - rattled-300
     - 2543
     - 0.001123
     - 0.002234
     - 0.998876
     - 234
     - 0.001234
     - 0.002456
     - 0.998765

- **ALL**: Aggregated metrics across all groups
- Groups sorted by validation RMSE descending (worst first)
- Separate tables for weighted and unweighted metrics

Scatterplots
~~~~~~~~~~~~

For each row type (Energy, Force, Stress), scatterplots show:

- X-axis: Ground truth values
- Y-axis: Predicted values
- Training data: Lighter colors, circles
- Validation data: Darker colors, squares with borders
- Black dashed line: Perfect prediction (y = x)
- Color-coded by group (tab20 colormap)

ARD Gamma/Lambda Heatmaps
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Only generated for ARD method**

Heatmaps visualize feature evolution across iterations for each basis function rank:

- **Rows**: Individual basis functions (ns, ls quantum numbers)
- **Columns**: ARD iterations
- **Color**: Gamma (feature usage) or Log10(Lambda) values
- **White cells**: Pruned features (below gamma threshold or above lambda threshold)

Each rank (0, 1, 2, 3, 4) gets a separate heatmap showing:

- Left margin: Element symbols
- Second column: ns (radial order) values
- Third column: ls (angular order) values
- Main heatmap: Iteration history

ARD Gamma/Lambda Histograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Side-by-side histograms for selected iterations showing:

- **Left column**: Gamma distribution (feature usage)

  - Range and mean statistics
  - Turbo colormap colored by bin position

- **Right column**: Log10(Lambda) distribution (feature precision)

  - Log-scale range and mean
  - Turbo colormap colored by bin position

Iterations shown:

- First iteration
- Evenly spaced middle iterations (up to 8)
- Final iteration

ADIOS2 Data Storage
~~~~~~~~~~~~~~~~~~~

All validation data is stored in ADIOS2 binary pack format:

**Metadata attributes:**

- ``nconfigs``: Total configuration count
- ``sorted_group_names``: Group names list
- ``basis_ranks``: PYACE basis function ranks
- ``blist``: Basis function descriptors
- ``has_forces``, ``has_stress``: Data availability flags

**Per-iteration arrays** (ARD only):

- ``gamma[iteration, n_features]``: Feature usage evolution
- ``lambda[iteration, n_features]``: Feature precision evolution

**Prediction arrays:**

For each combination of (row_type, group_idx, training/testing):

- ``energy_0_training[n_points, 2]``: [truths, predictions] for energy, group 0, training
- ``forces_1_testing[n_points, 2]``: [truths, predictions] for forces, group 1, testing
- etc.

File location: ``<metrics_prefix>.bp``

Examples
--------

The following examples demonstrate SLATE usage for different material systems and regression methods.

AlNi RIDGE (Binary Alloy)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Ridge regression for Al-Ni system using OMAT24 data:

.. code-block:: ini

   [PYACE]
   elements = Al Ni
   bzeroflag = 0
   embeddings = {"ALL": {"npot": "FinnisSinclairShiftedScaled", "fs_parameters": [1,1,1,0.5], "ndensity": 2}}
   bonds = {
       "AlAl": {"radbase": "ChebExpCos", "radparameters": [0.342], "rcut": 6.832, "dcut": 0.01},
       "AlNi": {"radbase": "ChebExpCos", "radparameters": [0.318], "rcut": 6.356, "dcut": 0.01},
       "NiNi": {"radbase": "ChebExpCos", "radparameters": [0.281], "rcut": 5.610, "dcut": 0.01}
   }
   functions = {
       "BINARY": {"lmax_by_orders": [0,3,2,1], "nradmax_by_orders": [8,3,2,1]},
       "Al":     {"lmax_by_orders": [0,3,2,1], "nradmax_by_orders": [8,3,2,1]},
       "Ni":     {"lmax_by_orders": [0,3,2,1], "nradmax_by_orders": [8,3,2,1]}
   }
   
   [CALCULATOR]
   calculator = LAMMPSPYACE
   energy = 1
   force = 1
   stress = 0
   
   [SOLVER]
   solver = SLATE
   compute_testerrs = 1
   
   [SLATE]
   method = RIDGE
   alpha = 1e-6
   
   [SCRAPER]
   scraper = ADIOS2
   max_configs_per_rank = 10
   
   [PATH]
   dataPath = omat24_AlNi.bp
   
   [GROUPS]
   group_sections = name eweight fweight vweight
   group_types = str float float float
   smartweights = 1
   rattled-300 = 1.0 100.0 10.0
   rattled-500 = 1.0 100.0 10.0
   # ... (more groups)
   
   [OUTFILE]
   output_style = PACE
   metrics = AlNi_PYACE_SLATE_RIDGE.md
   potential = AlNi_PYACE_SLATE_RIDGE
   validation = 1

**Key features:**

- Binary alloy with element-specific basis functions
- OMAT24 dataset with fixed train/val split
- Ridge regression with moderate regularization (α = 1e-6)
- Validation enabled for comprehensive error analysis

**Location:** ``examples/AlNi_PYACE_SLATE_RIDGE/``

AlNi ARD (Binary Alloy with Feature Selection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ARD regression for automatic feature selection:

.. code-block:: ini

   [SLATE]
   method = ARD
   max_iter = 10
   scap = 1e-1
   scai = 1e-1
   pruning_method = lambda
   threshold_lambda = 1e4
   logcut = 0.3

**Key features:**

- Same Al-Ni system as RIDGE example
- Automatic feature pruning during training
- Adaptive hyperparameters with moderate scaling
- Lambda-based pruning removes high-precision features
- Validation generates gamma/lambda heatmaps

**Location:** ``examples/AlNi_PYACE_SLATE_ARD/``

InP RIDGE (Binary Compound)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ridge regression for InP semiconductor:

.. code-block:: ini

   [PYACE]
   elements = In P
   embeddings = {
       "In": {"npot": "FinnisSinclair", "fs_parameters": [1, 1], "ndensity": 1},
       "P":  {"npot": "FinnisSinclair", "fs_parameters": [1, 1], "ndensity": 1}
   }
   bonds = {
       "InIn": {"radbase": "ChebExpCos", "radparameters": [1.737], "rcut": 5.790, "dcut": 0.01},
       "PIn":  {"radbase": "ChebExpCos", "radparameters": [1.502], "rcut": 5.007, "dcut": 0.01},
       "PP":   {"radbase": "ChebExpCos", "radparameters": [1.267], "rcut": 4.224, "dcut": 0.01}
   }
   functions = {"ALL": {"lmin_by_orders": [0,0,1,1], "lmax_by_orders": [1,2,2,1], "nradmax_by_orders": [22,3,2,1]}}
   bzeroflag = 1
   
   [ESHIFT]
   In = -1.65967588701534
   P = 4.38159549501534
   
   [SOLVER]
   solver = SLATE
   
   [SLATE]
   method = RIDGE
   alpha = 1e-6
   
   [SCRAPER]
   scraper = JSON
   
   [PATH]
   dataPath = ../InP_JPCA2020/JSON

**Key features:**

- Compound semiconductor with element-specific embeddings
- Energy shifts for elemental reference states
- Higher-rank basis functions (up to rank 4)
- JSON scraper for compatibility with legacy data
- Training/testing split specified in GROUPS section

**Note:** Example directory exists but ``InP_PYACE_SLATE_RIDGE.in`` file not found. Configuration shown is inferred from InP_PYACE_SLATE_ARD example.

InP ARD (Binary Compound with Feature Selection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ARD regression for InP with automatic basis optimization:

.. code-block:: ini

   [SLATE]
   method = ARD
   max_iter = 100
   scap = 1e-4
   scai = 1e-4
   logcut = 0.3

**Key features:**

- Same InP system as RIDGE example
- More iterations for convergence (100 vs default 10)
- Fine-tuned hyperparameter scaling (scap/scai = 1e-4)
- Large basis set (22 radial functions) benefits from ARD pruning
- Validation shows which basis functions are active

**Location:** ``examples/InP_PYACE_SLATE_ARD/``

Ta RIDGE (Elemental Metal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ridge regression for tantalum:

.. code-block:: ini

   [PYACE]
   elements = Ta
   embeddings = {"Ta": {"npot": "FinnisSinclair", "fs_parameters": [1, 1], "ndensity": 1}}
   bonds = {"ALL": {"radbase": "ChebExpCos", "radparameters": [1.275], "rcut": 4.25, "dcut": 0.01}}
   functions = {"ALL": {"lmin_by_orders": [0,0,1,1], "lmax_by_orders": [0,5,2,1], "nradmax_by_orders": [22,5,3,1]}}
   bzeroflag = 0
   
   [ESHIFT]
   Ta = 0.0
   
   [SOLVER]
   solver = SLATE
   compute_testerrs = 1
   detailed_errors = 1
   
   [SLATE]
   method = RIDGE
   alpha = 1e-4
   
   [SCRAPER]
   scraper = JSON
   
   [PATH]
   dataPath = ../Ta_Linear_JCP2014/JSON
   
   [GROUPS]
   group_sections = name training_size testing_size eweight fweight vweight
   group_types = str float float float float float
   smartweights = 1
   random_sampling = 0
   Displaced_BCC = 1.0 0.0 1.0 20.0 1.E-8
   Elastic_BCC   = 1.0 0.0 1.0 10.0 1.E-8
   Liquid        = 1.0 0.0 1.0 10.0 1.E-8
   # ... (more groups)
   
   [OUTFILE]
   output_style = PACE
   validation = 0

**Key features:**

- Single element system (simplified basis)
- High angular momentum (lmax = 5 for rank-2)
- Diverse training groups (bulk, surfaces, defects, liquid)
- Higher regularization (α = 1e-4) for stability
- No validation output (faster production runs)

**Location:** ``examples/Ta_PYACE_SLATE_RIDGE/``

Ta ARD (Elemental Metal with Feature Selection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ARD regression for tantalum with automatic feature selection:

.. code-block:: ini

   [SLATE]
   method = ARD
   max_iter = 100
   directmethod = 0
   scap = 1e-4
   scai = 1e-4
   logcut = 0.3
   threshold_lambda = 0  # Auto-computed

**Key features:**

- Same Ta system as RIDGE example
- Large basis set (22 + 5 + 3 + 1 = 31 radial functions per element)
- ARD automatically selects relevant high-rank contributions
- Auto-computed threshold based on data variance
- Adaptive hyperparameters for robust convergence

**Location:** ``examples/Ta_PYACE_SLATE_ARD/``

Running Examples
~~~~~~~~~~~~~~~~

**On a workstation** (limited configs for testing):

.. code-block:: bash

   # Test with small subset
   mpirun -np 4 python -m fitsnap3 AlNi_PYACE_SLATE_RIDGE.in --overwrite

**On HPC cluster** (production scale):

.. code-block:: bash

   # SLURM submission script
   #!/bin/bash
   #SBATCH --nodes=8
   #SBATCH --ntasks-per-node=24
   #SBATCH --time=04:00:00
   
   module load slate
   module load openmpi
   
   mpirun -np 192 python -m fitsnap3 AlNi_PYACE_SLATE_ARD.in --overwrite

**With validation:**

.. code-block:: bash

   # Run fitting
   mpirun -np 192 python -m fitsnap3 Ta_PYACE_SLATE_ARD.in
   
   # View validation notebook
   jupyter notebook Ta_PYACE_SLATE_ARD.ipynb

Performance Considerations
---------------------------

Scaling Characteristics
~~~~~~~~~~~~~~~~~~~~~~~

**RIDGE:**

- Scales efficiently to 100+ nodes for large matrices (m > 100k, n > 1k)
- Performance dominated by distributed QR factorization
- Memory per node: O(m/p × n) where p = number of nodes
- Typical speedup: near-linear up to 64 nodes, then communication overhead increases

**ARD:**

- Each iteration performs distributed matrix operations similar to RIDGE
- Iteration overhead: coefficient updates, pruning checks
- Memory per node: O(m/p × n_active) where n_active decreases over iterations
- Total time: max_iter × (RIDGE solve time + update overhead)
- Convergence typically in 5-20 iterations for well-posed problems

Optimal Configuration
~~~~~~~~~~~~~~~~~~~~~

**Node/rank allocation:**

- Use full nodes: ``ntasks-per-node`` = cores per node
- SLATE performs best with 2D process grids, but FitSNAP uses 1D distribution
- Minimize cross-node communication: favor taller matrices (more samples per node)

**Tile size:**

- Auto-computed to balance memory and communication
- Target: 16 MB per tile maximum
- Current implementation: one tile per rank (not yet optimized)

**Data distribution:**

- Each node stores local portion of augmented matrix
- Regularization rows distributed across nodes
- Training/testing split handled efficiently via weight masking

Memory Requirements
~~~~~~~~~~~~~~~~~~~

**Per-node memory estimate:**

.. math::

   \text{Memory} \approx \frac{m}{p} \times n \times 8 \text{ bytes} \times 2

Factor of 2 accounts for both ``aw`` and ``bw`` arrays.

**Example:**

- 100k samples, 1000 features, 8 nodes
- Memory per node ≈ (100k / 8) × 1000 × 8 × 2 = 200 MB

**ARD memory:**

- Additional storage for gamma, lambda arrays: O(n) per iteration
- ADIOS2 validation: stores all iterations, typically <100 MB total

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

**Symptom:**

.. code-block:: text

   Warning: SLATE module import failed
   To install: cd fitsnap3lib/lib/slate_solver && pip install -e .

**Solution:**

Build the SLATE wrapper:

.. code-block:: bash

   cd fitsnap3lib/lib/slate_solver
   pip install -e .
   
   # Verify installation
   python -c "from slate_wrapper import slate_ridge_augmented_qr_cython"

ARD Not Converging
~~~~~~~~~~~~~~~~~~

**Symptoms:**

- Coefficients oscillating between iterations
- Gamma values not stabilizing
- Maximum iterations reached without convergence

**Solutions:**

1. **Increase tolerance:**

   .. code-block:: ini
   
      rtol = 1e-2  # Less strict
      atol = 1e-5

2. **Adjust hyperparameters:**

   .. code-block:: ini
   
      scap = 1e-3  # Increase for more aggressive regularization
      scai = 1e-2  # Decrease for less aggressive pruning

3. **Use directmethod:**

   .. code-block:: ini
   
      directmethod = 1
      alphabig = 1e-10
      lambdasmall = 1e-8

4. **Check data quality:**

   - Ensure features are not collinear
   - Verify weight scaling is reasonable
   - Check for outliers in training data

ARD Pruning Too Aggressively
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Most features pruned early, poor fit quality

**Solutions:**

1. **Increase lambda threshold:**

   .. code-block:: ini
   
      threshold_lambda = 1e6  # Allow more features to remain

2. **Switch to gamma pruning:**

   .. code-block:: ini
   
      pruning_method = gamma
      threshold_gamma = 0.05  # Keep features contributing >5%

3. **Decrease pruning start:**

   Currently pruning starts at iteration 3 (hardcoded). To delay, modify source or increase ``max_iter``.

MPI Deadlocks
~~~~~~~~~~~~~

**Symptom:** Job hangs during SLATE operations

**Solutions:**

1. **Check process grid:**

   - Ensure MPI ranks evenly divide into nodes
   - Current implementation requires: ``mpi_size % mpi_number_of_nodes == 0``

2. **Verify data distribution:**

   .. code-block:: bash
   
      # Enable debug mode
      [EXTRAS]
      debug = 1

3. **Test on single node:**

   .. code-block:: bash
   
      mpirun -np 24 python -m fitsnap3 input.in  # Single node test

4. **Check SLATE installation:**

   - Ensure SLATE built with same MPI as FitSNAP
   - Verify SLATE library path in ``LD_LIBRARY_PATH``

Validation Notebook Empty
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Notebook generated but plots missing

**Solutions:**

1. **Check ADIOS2 file:**

   .. code-block:: bash
   
      ls -lh *.bp
      # Should exist and have non-zero size

2. **Verify validation flag:**

   .. code-block:: ini
   
      [OUTFILE]
      validation = 1  # Must be enabled

3. **Check matplotlib backend:**

   Validation uses ``Agg`` backend (non-interactive). If issues persist, try:

   .. code-block:: bash
   
      pip install matplotlib --upgrade

4. **Manually inspect ADIOS2:**

   .. code-block:: bash
   
      python -c "import adios2; fr = adios2.FileReader('output.bp'); print(fr.available_variables())"

Related Documentation
---------------------

- **SLATE Project**: https://github.com/icl-utk-edu/slate
- **SLATE Documentation**: https://icl-utk-edu.github.io/slate/
- **scikit-learn ARDRegression**: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html
- **FitSNAP Pull Request #278**: https://github.com/FitSNAP/FitSNAP/pull/278
- **PYACE Calculator**: :doc:`/Lib/calculators`
- **ADIOS2 Documentation**: https://adios2.readthedocs.io/

Citations
---------

If you use the SLATE solver in your research, please cite:

**SLATE:**

   Gates, M., Kurzak, J., Charara, A., et al. SLATE: Design of a Modern Distributed and Accelerated Linear Algebra Library. *SC20: International Conference for High Performance Computing, Networking, Storage and Analysis*, 2020.

**ARDRegression (algorithm basis):**

   Tipping, M. E. Sparse Bayesian learning and the relevance vector machine. *Journal of Machine Learning Research*, 2001, 1, 211-244.

**FitSNAP:**

   Cusentino, M. A., Thompson, A. P., & Wood, M. A. Explicit multielement extension of the spectral neighbor analysis potential for chemically complex systems. *Journal of Physical Chemistry A*, 2020, 124(26), 5456-5464.

**PYACE:**

   Lysogorskiy, Y., Bochkarev, A., Mrovec, M., & Drautz, R. Performant implementation of the atomic cluster expansion (PACE): Application to copper and silicon. *npj Computational Materials*, 2021, 7(1), 1-12.
