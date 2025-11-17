===========================================
Polynomial Atomic Cluster Expansion (PYACE)
===========================================

Overview
========

The Polynomial Atomic Cluster Expansion (PACE) provides a **complete and systematically improvable** representation of interatomic potentials through body-ordered basis functions. FitSNAP's PYACE implementation leverages the ``pyace`` Python package (pacemaker-compatible) to construct machine learning potentials with **linear scaling** in the number of neighbors, regardless of expansion order.

PACE achieves **meV/atom accuracy** comparable to neural network potentials while maintaining **computational efficiency** suitable for large-scale molecular dynamics simulations with millions of atoms.

Mathematical Foundation
=======================

Atomic Cluster Expansion
-------------------------

The total energy is decomposed into atomic contributions:

.. math::

   E = \sum_i E_i

Each atomic energy :math:`E_i` is expanded using body-ordered basis functions constructed from the local atomic environment. The key innovation is the **density trick** that enables linear scaling.

Atomic Base and Density Trick
------------------------------

Define the atomic density projection on basis functions:

.. math::

   A_{i\mu nlm} = \sum_{j \in \text{neighbors}} \delta_{\mu\mu_j} \phi_{\mu_j\mu_i nlm}(\mathbf{r}_{ji})

where the one-particle basis is:

.. math::

   \phi_{\mu_j\mu_i nlm}(\mathbf{r}_{ji}) = R^{\mu_j\mu_i}_{nl}(r_{ji}) Y^m_l(\hat{\mathbf{r}}_{ji})

with:

- :math:`R^{\mu_j\mu_i}_{nl}(r)`: **Radial basis functions** (element-pair specific)
- :math:`Y^m_l(\hat{\mathbf{r}})`: **Spherical harmonics** for angular dependencies  
- :math:`n`: radial quantum number, :math:`l,m`: angular momentum quantum numbers
- :math:`\mu_i, \mu_j`: chemical species indices

Body-Ordered Basis Functions
-----------------------------

Many-body correlations are constructed as products of the atomic base:

.. math::

   A_{i\mathbf{\mu nlm}} = \prod_{\alpha=1}^{\nu} A_{i\mu_\alpha n_\alpha l_\alpha m_\alpha}

The body order is :math:`\nu + 1` (including the central atom). The critical insight: **all :math:`\nu`-body interactions computed in** :math:`O(N_c)` **time** where :math:`N_c` is the number of neighbors, avoiding the naive :math:`O(N_c^\nu)` scaling.

Rotationally Invariant Expansion
---------------------------------

For rotational invariance, products of spherical harmonics must couple to :math:`L=0` using Clebsch-Gordan coefficients. The atomic property expansion becomes:

.. math::

   \phi_i^{(p)} = \sum_{\mathbf{\mu nlm}} \tilde{c}^{(p)}_{\mu_i\mathbf{\mu nlm}} A_{i\mathbf{\mu nlm}}

where expansion coefficients :math:`\tilde{c}` incorporate coupling coefficient constraints.

Energy Functional Forms
-----------------------

**Linear Model** (Silicon-like covalent systems):

.. math::

   E_i = \phi_i^{(1)}

**Finnis-Sinclair Model** (Metallic systems like Copper):

.. math::

   E_i = -\sqrt{\phi_i^{(2)} + \epsilon} + \phi_i^{(1)} + \phi_i^{(3)}

where :math:`\phi_i^{(1)}` represents pair repulsion, :math:`\phi_i^{(2)}` the bond order (density), and :math:`\phi_i^{(3)}` core repulsion at short distances.

**FinnisSinclairShiftedScaled** (Binary alloys like AlNi):

.. math::

   E_i = \text{fs\_parameters}[0] \cdot \left( -\sqrt{\phi_i^{(2)} + \epsilon} + \text{fs\_parameters}[3] \right) + \phi_i^{(1)}

Provides additional flexibility for fitting cohesive energy curves in multi-component systems.

Radial Basis Functions
======================

ChebExpCos (Recommended)
------------------------

Chebyshev polynomial basis with exponential coordinate transformation and cosine envelope:

.. math::

   R_{nl}(r) = \sum_k c_{nlk} g_k(r)

where:

.. math::

   g_k(r) = \frac{1}{2}\left[1 - T_{k-1}(x)\right]\left[1 + \cos\left(\frac{\pi r}{r_{\text{cut}}}\right)\right]

.. math::

   x = 1 - 2\frac{e^{-\lambda(r/r_{\text{cut}} - 1)} - 1}{e^\lambda - 1}

**Parameters:**

- ``radparameters = [λ]``: Controls radial resolution (typical: 1.0-2.0)
- ``rcut``: Outer cutoff radius
- ``dcut``: Width of smooth cutoff (default: 0.01)

**Advantages:**

- Exponentially denser sampling at small :math:`r` (captures steep gradients)
- Smooth cutoff ensures :math:`R_{nl}(r_{\text{cut}}) = 0` and :math:`dR_{nl}/dr|_{r_{\text{cut}}} = 0`
- Orthogonal basis enables clean separation of radial channels

ChebPow
-------

Chebyshev basis with power-law coordinate transformation:

.. math::

   x = 1 - 2\left(\frac{r}{r_{\text{cut}}}\right)^p

**When to use:** Uniform radial sampling needed (set :math:`p=1`).

SBessel
-------

Spherical Bessel functions of the first kind with smooth cutoff.

**When to use:** Physics-motivated basis for systems with oscillatory potentials (e.g., metals near Friedel oscillations).

Core Repulsion
==============

Prevents unphysical atom overlap at distances shorter than training data range:

.. math::

   \phi_i^{(\text{core})} = c \sum_j \frac{e^{-\gamma r_{ji}^2}}{r_{ji}}

Activated via ``"core-repulsion": [c, γ]`` in bonds specification.

**Inner Cutoff:** The ACE expansion is smoothly disabled using:

.. math::

   f_{\text{cut}}(\phi^{(\text{core})}) = \begin{cases}
   1 & 0 \leq \phi^{(\text{core})} \leq \phi_{\text{cut}} - \Delta_{\text{cut}} \\
   \frac{1}{2}\left(1 - \cos\left(\pi\frac{\phi^{(\text{core}}} - \phi_{\text{cut}}}{\Delta_{\text{cut}}}\right)\right) & \phi_{\text{cut}} - \Delta_{\text{cut}} < \phi^{(\text{core})} < \phi_{\text{cut}} \\
   0 & \phi^{(\text{core})} \geq \phi_{\text{cut}}
   \end{cases}

Controlled by ``r_in`` and ``delta_in`` parameters.

PYACE Input File Format
========================

[PYACE] Section
---------------

Required Parameters
^^^^^^^^^^^^^^^^^^^

.. code-block:: ini

   [PYACE]
   elements = Al Ni           # Chemical species (MUST use symbols, not numbers)
   bzeroflag = 0              # 0: include per-element constants; 1: exclude (use [ESHIFT])
   
   embeddings = {JSON}        # Embedding function specification
   bonds = {JSON}             # Radial basis and cutoffs per element pair  
   functions = {JSON}         # Angular basis per element combination

Embedding Specification
^^^^^^^^^^^^^^^^^^^^^^^

Controls the functional form :math:`E_i = F(\phi_i^{(1)}, \phi_i^{(2)}, \ldots)`:

**Example 1: Metallic (Finnis-Sinclair)**

.. code-block:: python

   embeddings = {
       "ALL": {
           "npot": "FinnisSinclair",
           "fs_parameters": [1, 1],
           "ndensity": 1
       }
   }

- ``npot``: Potential type (``"FinnisSinclair"`` or ``"FinnisSinclairShiftedScaled"``)
- ``fs_parameters``: Embedding function parameters
- ``ndensity``: Number of density functions (typically 1 or 2)

**Example 2: Binary Alloy (Shifted-Scaled)**

.. code-block:: python

   embeddings = {
       "ALL": {
           "npot": "FinnisSinclairShiftedScaled",
           "fs_parameters": [1, 1, 1, 0.5],
           "ndensity": 2
       }
   }

**Parameters:**

- ``[a, b, c, d]``: :math:`E_i = a \cdot \left(-\sqrt{b \cdot \phi_i^{(2)} + \epsilon} + d\right) + c \cdot \phi_i^{(1)}`

Bonds Specification  
^^^^^^^^^^^^^^^^^^^

Defines radial basis and cutoffs **per element pair**:

.. code-block:: python

   bonds = {
       "AlAl": {
           "radbase": "ChebExpCos",
           "radparameters": [0.342],
           "rcut": 6.832,
           "dcut": 0.01,
           "core-repulsion": [0, 0.342],
           "r_in": 0.625,
           "delta_in": 0.01
       },
       "AlNi": {
           "radbase": "ChebExpCos",
           "radparameters": [0.318],
           "rcut": 6.356,
           "dcut": 0.01,
           "core-repulsion": [0, 0.318],
           "r_in": 0.650,
           "delta_in": 0.01
       },
       "NiNi": {
           "radbase": "ChebExpCos",
           "radparameters": [0.281],
           "rcut": 5.610,
           "dcut": 0.01,
           "core-repulsion": [0, 0.281],
           "r_in": 0.540,
           "delta_in": 0.01
       }
   }

**Notes:**

- ``"NiAl"`` not needed if symmetric (automatically set equal to ``"AlNi"``)
- Bond pair names: ``"{element_j}{element_i}"`` for neighbor :math:`j` → central atom :math:`i`
- Values automatically populate ``nradmax`` and ``lmax`` from ``functions`` section

Functions Specification
^^^^^^^^^^^^^^^^^^^^^^^

Defines **body order** and **angular complexity** per element combination:

**Single Element (Ta):**

.. code-block:: python

   functions = {
       "ALL": {
           "lmin_by_orders": [0, 0, 1, 1],  # Min L per rank
           "lmax_by_orders": [0, 5, 2, 1],  # Max L per rank  
           "nradmax_by_orders": [22, 5, 3, 1]  # Max radial functions per rank
       }
   }

Produces ranks 1-4 (body orders 2-5) with decreasing angular/radial complexity.

**Binary System (InP):**

.. code-block:: python

   functions = {
       "ALL": {
           "lmin_by_orders": [0, 0, 1, 1],
           "lmax_by_orders": [1, 2, 2, 1],
           "nradmax_by_orders": [22, 3, 2, 1]
       }
   }

**Binary System (AlNi) - Element-Specific:**

.. code-block:: python

   functions = {
       "BINARY": {  # Any rank containing 2+ elements
           "lmax_by_orders": [0, 1, 2, 3, 4],
           "nradmax_by_orders": [16, 8, 4, 2, 1]
       },
       "Al": {  # Ranks with only Al neighbors
           "lmax_by_orders": [0, 1, 2, 3, 4],
           "nradmax_by_orders": [16, 8, 4, 2, 1]
       },
       "Ni": {  # Ranks with only Ni neighbors  
           "lmax_by_orders": [0, 1, 2, 3, 4],
           "nradmax_by_orders": [16, 8, 4, 2, 1]
       }
   }

**Element Combination Keys:**

- ``"ALL"``: Every possible rank combination (single-element systems or uniform treatment)
- ``"BINARY"``: Any rank with 2+ distinct elements (e.g., Al-Ni-Ni, Al-Al-Ni)
- ``"TERNARY"``: Any rank with 3 distinct elements (e.g., C-O-H-O)
- ``"{Element}"``: Ranks containing only that element (e.g., ``"Al"`` → Al-Al-Al)

**Index Meaning:**

- Position in array = rank index (0 = rank 1 = pair interactions)
- ``lmin/lmax_by_orders[k]`` = angular momentum range for rank :math:`k+1`
- ``nradmax_by_orders[k]`` = number of radial basis functions for rank :math:`k+1`

[CALCULATOR] Section
--------------------

.. code-block:: ini

   [CALCULATOR]
   calculator = LAMMPSPYACE    # PyACE calculator with LAMMPS compute pace
   energy = 1                   # Fit energies (0 = exclude)
   force = 1                    # Fit forces (0 = exclude)
   stress = 0                   # Fit stresses (0 = exclude)

[ESHIFT] Section
----------------

Per-element energy shifts when ``bzeroflag = 1``:

.. code-block:: ini

   [ESHIFT]
   In = -1.65967588701534
   P = 4.38159549501534

**Purpose:** Removes reference energy offsets (e.g., isolated atom DFT energies).

**Relationship to bzeroflag:**

- ``bzeroflag = 0``: One-hot encoding automatically learns per-element constants → **no [ESHIFT] needed**
- ``bzeroflag = 1``: Per-element constants excluded → **must provide [ESHIFT]** for correct absolute energies

Complete Examples
=================

Example 1: AlNi Binary Alloy (OMAT24 Dataset)
----------------------------------------------

**System:** fcc Al-Ni alloys from Open Materials 2024 dataset

**Strategy:** FinnisSinclairShiftedScaled embedding with element-specific basis functions

.. code-block:: ini

   [PYACE]
   elements = Al Ni
   bzeroflag = 0
   
   embeddings = {
       "ALL": {
           "npot": "FinnisSinclairShiftedScaled",
           "fs_parameters": [1, 1, 1, 0.5],
           "ndensity": 2
       }
   }
   
   bonds = {
       "AlAl": {"radbase": "ChebExpCos", "radparameters": [0.342],
                "rcut": 6.832, "dcut": 0.01,
                "core-repulsion": [0, 0.342], "r_in": 0.625, "delta_in": 0.01},
       "AlNi": {"radbase": "ChebExpCos", "radparameters": [0.318],
                "rcut": 6.356, "dcut": 0.01,
                "core-repulsion": [0, 0.318], "r_in": 0.650, "delta_in": 0.01},
       "NiNi": {"radbase": "ChebExpCos", "radparameters": [0.281],
                "rcut": 5.610, "dcut": 0.01,
                "core-repulsion": [0, 0.281], "r_in": 0.540, "delta_in": 0.01}
   }
   
   functions = {
       "BINARY": {"lmax_by_orders": [0, 1, 2, 3, 4],
                  "nradmax_by_orders": [16, 8, 4, 2, 1]},
       "Al": {"lmax_by_orders": [0, 1, 2, 3, 4],
              "nradmax_by_orders": [16, 8, 4, 2, 1]},
       "Ni": {"lmax_by_orders": [0, 1, 2, 3, 4],
              "nradmax_by_orders": [16, 8, 4, 2, 1]}
   }
   
   [CALCULATOR]
   calculator = LAMMPSPYACE
   energy = 1
   force = 1
   stress = 0
   
   [SOLVER]
   solver = SLATE
   compute_testerrs = 1
   detailed_errors = 0
   
   [SLATE]
   method = ARD
   pruning_method = lambda
   threshold_lambda = 1e4
   scap = 1e-1
   scai = 1e-1
   max_iter = 10

**Location:** ``examples/AlNi_PYACE_SLATE_ARD/``

Example 2: Ta Elemental Metal
------------------------------

**System:** bcc Tantalum with complex defect structures

**Strategy:** Finnis-Sinclair with high radial resolution for pair terms

.. code-block:: ini

   [PYACE]
   elements = Ta
   
   embeddings = {
       "Ta": {"npot": "FinnisSinclair", "fs_parameters": [1, 1], "ndensity": 1}
   }
   
   bonds = {
       "ALL": {
           "radbase": "ChebExpCos",
           "radparameters": [1.275],
           "dcut": 0.01,
           "rcut": 4.25,
           "core-repulsion": [0, 1.275],
           "r_in": 1.2,
           "delta_in": 0.01
       }
   }
   
   functions = {
       "ALL": {"lmin_by_orders": [0, 0, 1, 1],
               "lmax_by_orders": [0, 5, 2, 1],
               "nradmax_by_orders": [22, 5, 3, 1]}
   }
   
   bzeroflag = 0
   
   [CALCULATOR]
   calculator = LAMMPSPYACE
   energy = 1
   force = 1
   stress = 0

**Location:** ``examples/Ta_PYACE_SLATE_ARD/``

**Key Feature:** Rank 1 (pairs) uses 22 radial functions for accurate pair potential; higher ranks use fewer radial functions but more angular terms (``lmax=[0,5,2,1]``).

Example 3: InP Covalent Semiconductor
--------------------------------------

**System:** Zinc-blende InP with tetrahedral bonding

**Strategy:** Linear model with moderate angular terms

.. code-block:: ini

   [PYACE]
   elements = In P
   
   embeddings = {
       "In": {"npot": "FinnisSinclair", "fs_parameters": [1, 1], "ndensity": 1},
       "P": {"npot": "FinnisSinclair", "fs_parameters": [1, 1], "ndensity": 1}
   }
   
   bonds = {
       "InIn": {"radbase": "ChebExpCos", "radparameters": [1.737],
                "rcut": 5.790, "dcut": 0.01,
                "core-repulsion": [0, 1.737], "r_in": 1.705, "delta_in": 0.01},
       "PIn": {"radbase": "ChebExpCos", "radparameters": [1.502],
               "rcut": 5.007, "dcut": 0.01,
               "core-repulsion": [0, 1.502], "r_in": 1.403, "delta_in": 0.01},
       "PP": {"radbase": "ChebExpCos", "radparameters": [1.267],
              "rcut": 4.224, "dcut": 0.01,
              "core-repulsion": [0, 1.267], "r_in": 1.100, "delta_in": 0.01}
   }
   
   functions = {
       "ALL": {"lmin_by_orders": [0, 0, 1, 1],
               "lmax_by_orders": [1, 2, 2, 1],
               "nradmax_by_orders": [22, 3, 2, 1]}
   }
   
   bzeroflag = 1
   
   [ESHIFT]
   In = -1.65967588701534
   P = 4.38159549501534

**Location:** ``examples/InP_PYACE_SLATE_ARD/``

**Key Feature:** Uses ``bzeroflag = 1`` with explicit [ESHIFT] values; directional bonding captured by ``lmax=[1,2,2,1]``.

Migrating from Legacy ACE to PYACE
===================================

Parameter Mapping
-----------------

================== ========================= =================================
Legacy ACE         PYACE                     Notes
================== ========================= =================================
``numTypes``       ``len(elements)``         Auto-computed from elements list
``type``           ``elements``              **Must use symbols** (e.g., "Al Ni" not "1 2")
``rcutfac``        ``bonds[...]["rcut"]``    Per-pair specification
``lambda``         ``bonds[...]["radparameters"]`` Inside list: ``[λ]``
``rcinner``        ``bonds[...]["r_in"]``    Inner cutoff radius
``drcinner``       ``bonds[...]["delta_in"]`` Inner cutoff width
``ranks``          ``len(lmax_by_orders)``   Array length defines ranks
``lmin``           ``lmin_by_orders``        Array per rank
``lmax``           ``lmax_by_orders``        Array per rank
``nmax``           ``nradmax_by_orders``     Array per rank
``nmaxbase``       Auto-set                  Computed from ``nradmax_by_orders``
``bzeroflag``      ``bzeroflag``             **Same**
``erefs``          ``[ESHIFT]``              When ``bzeroflag = 1``
================== ========================= =================================

Migration Example: InP
----------------------

**Legacy ACE Format:**

.. code-block:: ini

   [ACE]
   numTypes = 2
   rcutfac = 5.790  5.007  5.007  4.224
   lambda = 1.737  1.502  1.502  1.267
   rcinner = 1.705  1.403  1.403  1.100
   drcinner = 0.01 0.01 0.01 0.01
   ranks = 1 2 3 4
   lmax =  1 2 2 1
   nmax =  22 3 2 1
   mumax = 2
   lmin = 0 0 1 1
   erefs = 0 0
   nmaxbase = 22
   type = In P
   bzeroflag = 1

**PYACE Format:**

.. code-block:: ini

   [PYACE]
   elements = In P
   
   embeddings = {
       "In": {"npot": "FinnisSinclair", "fs_parameters": [1, 1], "ndensity": 1},
       "P": {"npot": "FinnisSinclair", "fs_parameters": [1, 1], "ndensity": 1}
   }
   
   bonds = {
       "InIn": {"radbase": "ChebExpCos", "radparameters": [1.737],
                "rcut": 5.790, "dcut": 0.01,
                "core-repulsion": [0, 1.737], "r_in": 1.705, "delta_in": 0.01},
       "PIn":  {"radbase": "ChebExpCos", "radparameters": [1.502],
                "rcut": 5.007, "dcut": 0.01,
                "core-repulsion": [0, 1.502], "r_in": 1.403, "delta_in": 0.01},
       "PP":   {"radbase": "ChebExpCos", "radparameters": [1.267],
                "rcut": 4.224, "dcut": 0.01,
                "core-repulsion": [0, 1.267], "r_in": 1.100, "delta_in": 0.01}
   }
   
   functions = {
       "ALL": {"lmin_by_orders": [0, 0, 1, 1],
               "lmax_by_orders": [1, 2, 2, 1],
               "nradmax_by_orders": [22, 3, 2, 1]}
   }
   
   bzeroflag = 1
   
   [ESHIFT]
   In = -1.65967588701534
   P = 4.38159549501534

**Key Changes:**

1. ``type = In P`` → ``elements = In P``
2. ``rcutfac`` array → per-pair ``bonds`` dictionary
3. ``ranks/lmax/nmax`` → ``functions`` dictionary with ``lmax_by_orders``
4. ``erefs`` → ``[ESHIFT]`` section
5. Added ``embeddings`` specification (required in PYACE)
6. Bond pairs explicitly named (``InIn``, ``PIn``, ``PP``)

Validation Notebooks
====================

PYACE potentials generate validation notebooks showing basis function analysis:

Basis Function Heatmaps
-----------------------

**Generated by:** ``validation = 1`` in ``[OUTFILE]`` section

**Content:**

1. **Central Atom Heatmaps:** Basis functions grouped by central element

   - Rows: Neighbor element combinations (e.g., "Al", "Ni", "Al Ni")
   - Columns: ``ns`` (radial functions), ``ls`` (angular momentums)
   - Color intensity: Gamma or Lambda basis function values per ARD iteration (white: pruned)

2. **Neighbor Heatmaps:** Basis functions grouped by neighbor combinations

   - Shows which angular terms (``ls``) are active for each radial channel (``ns``)
   - Helps identify if high-:math:`l` terms are necessary (indicator of angular selectivity)

3. **ARD Coefficient Analysis:** When using SLATE ARD solver

   - Histogram of coefficient magnitudes
   - Pruning statistics (how many functions retained/removed)
   - Identifies most important basis functions

**Interpretation:**

- **Dense low-**:math:`l` **regions:** System has weak angular dependence (metallic)
- **Sparse high-**:math:`l` **regions:** Strong directional bonding (covalent)
- **Radial channel dominance:** Check if more ``nradmax`` needed in specific ranks

Advanced Topics
===============

Embedding Function Details
---------------------------

**FinnisSinclair:**

.. math::

   E_i = \phi_i^{(1)} - \sqrt{\phi_i^{(2)} + \epsilon}

- :math:`\phi_i^{(1)}`: Pair repulsion (rank 1)
- :math:`\phi_i^{(2)}`: Bond order / electron density (ranks 2-4)
- :math:`\epsilon = 10^{-10}`: Numerical stabilizer

**FinnisSinclairShiftedScaled:**

.. math::

   E_i = a \cdot \left( -\sqrt{b \cdot \phi_i^{(2)} + \epsilon} + d \right) + c \cdot \phi_i^{(1)}

- ``fs_parameters = [a, b, c, d]``
- Extra flexibility for multi-component cohesive energy curves
- Use when simple FS fails to capture alloy energetics

**When to use which:**

- **Metals (Cu, Ta):** FinnisSinclair sufficient
- **Alloys (AlNi):** FinnisSinclairShiftedScaled for better transferability
- **Covalent (InP, Si):** Linear model (no embedding) with ``bzeroflag = 1``

Hyperparameter Selection
-------------------------

**Cutoff Radii:**

- Start with first minimum of radial distribution function
- Binary systems: Use element-specific cutoffs based on atomic radii
- Larger cutoffs → more basis functions → better accuracy but slower

**Radial Resolution (nradmax):**

- Rank 1: High (15-30) for accurate pair potential
- Rank 2-3: Moderate (3-8) for many-body corrections
- Rank 4+: Low (1-3) for high-order correlations

**Angular Complexity (lmax):**

- Metals: ``lmax = [0, 1-3, 1-2, 1]`` (weak angular)
- Covalent: ``lmax = [0-2, 2-4, 2-3, 1-2]`` (strong angular)  
- Start low, increase if errors persist

**Body Order (ranks):**

- Minimum: 3 ranks (up to 4-body)
- Typical: 4 ranks (up to 5-body)
- Diminishing returns beyond 5 ranks for most systems

Performance Considerations
--------------------------

**Computational Cost:**

- Scales as :math:`O(N_c \cdot N_{\text{basis}})` per atom
- :math:`N_{\text{basis}} \propto \sum_{\nu} (\nu + 1) \cdot n_{\text{rad}}^\nu \cdot l_{\text{max}}^\nu`
- SLATE ARD pruning reduces :math:`N_{\text{basis}}` by 50-90%

**Memory:**

- Descriptor matrix: :math:`N_{\text{configs}} \times N_{\text{basis}}`
- For 1M configs with 5k basis: ~40 GB
- Use distributed memory (MPI) for large datasets


References
==========

.. [Drautz2019] Drautz, R. "Atomic cluster expansion for accurate and transferable interatomic potentials." *Phys. Rev. B* **99**, 014104 (2019). `DOI:10.1103/PhysRevB.99.014104 <https://doi.org/10.1103/PhysRevB.99.014104>`_

.. [Lysogorskiy2021] Lysogorskiy, Y., van der Oord, C., Bochkarev, A., et al. "Performant implementation of the atomic cluster expansion (PACE) and application to copper and silicon." *npj Comput. Mater.* **7**, 97 (2021). `DOI:10.1038/s41524-021-00559-9 <https://doi.org/10.1038/s41524-021-00559-9>`_

.. [Pacemaker] Pacemaker Documentation: `https://pacemaker.readthedocs.io <https://pacemaker.readthedocs.io>`_

Example Locations
=================

All examples available in ``FitSNAP-alphataubio/examples/``:

- ``AlNi_PYACE_SLATE_ARD/``: Binary alloy with ARD pruning
- ``AlNi_PYACE_SLATE_RIDGE/``: Binary alloy with Ridge regression
- ``Ta_PYACE_SLATE_ARD/``: Elemental metal
- ``InP_PYACE_SLATE_ARD/``: Binary semiconductor
- ``InP_PACE_ARD/``: Legacy ACE format (for comparison)

See Also
========

- :doc:`SLATE solver documentation <slate>` for distributed ARD/Ridge regression
- :doc:`LAMMPS calculator <calculator>` for descriptor computation
- :doc:`ADIOS2 scraper <scraper>` for large dataset handling
