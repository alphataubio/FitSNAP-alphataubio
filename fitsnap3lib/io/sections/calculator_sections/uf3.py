
from fitsnap3lib.io.sections.sections import Section

import json, ast
import numpy as np
from datetime import datetime

from fitsnap3lib.lib.uf3.data.composition import ChemicalSystem
from fitsnap3lib.lib.uf3.representation.bspline import BSplineBasis

def _format_species(obj):
  """Format element tuples/lists for logging without Python string quotes."""
  if isinstance(obj, str):
    return obj
  if isinstance(obj, tuple):
    return "(" + ", ".join(_format_species(x) for x in obj) + ")"
  if isinstance(obj, list):
    return "[" + ", ".join(_format_species(x) for x in obj) + "]"
  return str(obj)


class Uf3(Section):
  """
  UF3 B-spline basis: builds ``BSplineBasis``, ``ncoeff``, type mapping, and (on first
  access to ``template_uf3_path``) writes ``{OUTFILE potential}.uf3`` for LAMMPS
  ``compute ... uf3``. Section order in the input file does not matter:
  ``[OUTFILE]`` / ``[REFERENCE]`` are read from the parsed config, not from
  ``Section.sections``.
  """

  def __init__(self, name, config, pt, infile, args):
    super().__init__(name, config, pt, infile, args)

    allowedkeys = [
      "type",
      "degree",
      "knot_strategy",
      "r_min", "r_min_2b", "r_min_3b",
      "r_max", "r_max_2b", "r_max_3b",
      "resolution", "resolution_2b", "resolution_3b",
      "r_min_map_json",
      "r_max_map_json",
      "resolution_map_json",
      "knots_map_json",
      "bzeroflag",
    ]

    sec = self.name.upper()
    for key in self._config[sec]:
      if key not in allowedkeys:
        raise RuntimeError(f"Unmatched variable in {sec} section of input: {key}")

    elements_str = self.get_value(sec, "type", "Al Ni")
    user_elements = elements_str.split()
    self.degree = self.get_value(sec, "degree", "2", "int")
    if self.degree not in (2, 3):
      raise RuntimeError(
        "FitSNAP UF3: degree must be 2 (2-body .uf3 only) or 3 (2- and 3-body). "
        f"Got degree = {self.degree}."
      )

    # ChemicalSystem sorts symbols (electronegativity / Z). LAMMPS ``compute uf3 E1 E2 ...`` maps
    # type 1 -> first symbol, etc., and column order follows nested type loops (1,1)..(1,nt)..(nt,nt).
    # BSpline partitions use the same sorted ``element_list``. Using the raw input order here
    # desynchronizes A-matrix columns from BSpline 1-body / 2-body layout and destroys the fit.
    self.chemical_system = ChemicalSystem(element_list=user_elements, degree=self.degree)
    self.type = list(self.chemical_system.element_list)
    self.numtypes = len(self.type)
    if user_elements != self.type:
      self.pt.single_print(
        "[UF3] Using canonical element order for LAMMPS types and BSpline partitions: "
        + f"{' '.join(self.type)} (input was: {' '.join(user_elements)})\n"
      )

    knot_strategy = self.get_value(sec, "knot_strategy", "linear")
    self.bzeroflag = self.get_value(sec, "bzeroflag", "1", "bool")
    r_min_map = r_max_map = resolution_map = knots_map = None

    interactions = self.chemical_system.interactions_map

    # r_min

    if self._config.has_option(sec, "r_min_map_json"):
      r_min_map = ast.literal_eval(self._config.get(sec, "r_min_map_json"))

    if r_min_map is None and self._config.has_option(sec, "r_min"):
      r_min = float(self._config.get(sec, "r_min"))
      self.r_min_map = {t: r_min for t in interactions[2]}

    # r_max

    if self._config.has_option(sec, "r_max_map_json"):
      r_max_map = ast.literal_eval(self._config.get(sec, "r_max_map_json"))

    if r_max_map is None:

      if self._config.has_option(sec, "r_max"):
        r_max_2b = r_max_3b = float(self._config.get(sec, "r_max"))
        
      if self._config.has_option(sec, "r_max_2b"):
        r_max_2b = float(self._config.get(sec, "r_max_2b"))
        
      if self._config.has_option(sec, "r_max_3b"):
        r_max_3b = float(self._config.get(sec, "r_max_3b"))

      self.r_max_map = {t: r_max_2b for t in interactions[2]}
      if 3 in interactions:
        self.r_max_map.update({t: [r_max_3b, r_max_3b, 2*r_max_3b] for t in interactions[3]})

    # resolution

    if self._config.has_option(sec, "resolution_map_json"):
      resolution_map = ast.literal_eval(self._config.get(sec, "resolution_map_json"))

    if self._config.has_option(sec, "knots_map_json"):
      knots_map = ast.literal_eval(self._config.get(sec, "knots_map_json"))["knots"]

    if resolution_map is None:

      if self._config.has_option(sec, "resolution"):
        resolution_2b = resolution_3b = int(self._config.get(sec, "resolution"))

      if self._config.has_option(sec, "resolution_2b"):
        resolution_2b = int(self._config.get(sec, "resolution_2b"))

      if self._config.has_option(sec, "resolution_3b"):
        resolution_3b = int(self._config.get(sec, "resolution_3b"))

      self.resolution_map = {t: resolution_2b for t in interactions[2]}
      if 3 in interactions:
        self.resolution_map.update({t: [resolution_3b, resolution_3b, resolution_3b] for t in interactions[3]})


    #self.pt.single_print(f"*** resolution_map {self.resolution_map}")

    # create basis

    self.bspline_basis = BSplineBasis(
      self.chemical_system,
      r_min_map=self.r_min_map,
      r_max_map=self.r_max_map,
      resolution_map=self.resolution_map,
      knot_strategy=knot_strategy,
      offset_1b=bool(not self.bzeroflag),
      leading_trim=0,
      trailing_trim=3,
      knots_map=knots_map,
    )

    # get_feature_partition_sizes() = [1]*numtypes (composition / 1-body) then 2-body blocks
    # (then 3-body if degree>2). LAMMPS ``compute uf3`` exposes 2-body columns, then 3-body
    # (same total count as these spline blocks when nbody=3 and the template includes 3B).
    self.feature_partition_sizes = self.bspline_basis.get_feature_partition_sizes()
    self.ncoeff = int(np.sum(self.feature_partition_sizes))
    self.type_mapping = {el: i + 1 for i, el in enumerate(self.type)}

    self.pt.single_print(
      f"----------------------------------------------------------------\n"
      f"  UF3 B-SPLINE BASIS                                            \n"
      f"                                                                \n"
      f"    Xie, S.R., Rupp, M. & Hennig, R.G.,                         \n"
      f"    Ultra-fast interpretable machine-learning potentials.       \n"
      f"    npj computational materials 9, 162 (2023).                  \n"
      f"    https://doi.org/10.1038/s41524-023-01092-7                  \n"
      f"                                                                \n"
      f"    CHEMICAL SYSTEM                                             \n"
      f"    Elements: {_format_species(self.chemical_system.element_list)}\n"
      f"    Degree: {self.chemical_system.degree}                     \n"
      f"    Singles: {_format_species(self.chemical_system.interactions_map[1])}\n"
      f"    Pairs: {_format_species(self.chemical_system.interactions_map[2])}"
    )

    if self.chemical_system.degree >= 3: self.pt.single_print(
      f"    Triplets: {_format_species(self.chemical_system.interactions_map[3])}      "
    )

    self.pt.single_print(
      f"\n    BASIS FUNCTIONS                                             "
    )

    sizes = self.bspline_basis.get_interaction_partitions()[0]
    for n in range(1, self.bspline_basis.degree + 1):
      for interaction in self.bspline_basis.interactions_map[n]:
        self.pt.single_print(
          f"    {_format_species(interaction):16s} {sizes[interaction]:4d}"
        )

    self.pt.single_print(
      f"    TOTAL            {sum(self.feature_partition_sizes):4d}     \n"
    )

    # Initialize on all ranks so the attribute always exists before create_a() runs.
    # write_uf3_lammps_pot (rank 0 only) will populate them; broadcast syncs everyone.
    self.basis_ranks = []
    self.blist = []

    @self.pt.rank_zero
    def _write():
      self.write_uf3_lammps_pot("descriptors.uf3")
    _write()
    self.pt.all_barrier()

    if not self.pt.stubs:
      self.basis_ranks = self.pt._comm.bcast(self.basis_ranks, root=0)
      self.blist = self.pt._comm.bcast(self.blist, root=0)


  @staticmethod
  def _parse_trim(val):
    if isinstance(val, str) and val.strip().startswith("{"):
      return json.loads(val.replace("'", '"'))
    return int(val)

  def write_uf3_lammps_pot(self, path, knots_spacing_type="nk"):
    """Write a LAMMPS UF3 potential file (2B/3B blocks) for ``compute uf3`` descriptors."""

    chemical_sys = self.chemical_system
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    author = "FitSNAP"
    leading_trim = 0
    trailing_trim = 3
    lammps_units = str(self.get_value("REFERENCE", "units", "metal")).lower()

    self.basis_ranks = []
    self.blist = []

    # One rank-0 name per element (``interactions_map[1]`` is symbol strings;
    # ``len("Al")`` is wrong for a per-species count).
    for interaction in chemical_sys.interactions_map[1]:
      self.basis_ranks.append(0)
      self.blist.append(f"{interaction}")

    with open(path, "w") as pot:

      for interaction in chemical_sys.interactions_map[2]:

        knots_2b = self.bspline_basis.knots_map[interaction]
        length = self.bspline_basis.get_interaction_partitions()[0][interaction]
        start_idx = self.bspline_basis.get_interaction_partitions()[1][interaction] + 1
        end_idx = start_idx + length
        self.basis_ranks.extend([1] * length)
        # One label per 2B column: middle knot of each cubic B-spline support window
        # (``get_knot_subintervals``), not a slice of ``knots_2b`` by global column
        # index (wrong for >2 elements) or by ``offset(first_pair)+1`` (wrong for 4+).
        subs = self.bspline_basis.knot_subintervals[interaction]
        if len(subs) != length:
          raise RuntimeError(
            f"UF3 2B {interaction}: knot_subintervals ({len(subs)}) != partition size ({length})."
          )
        self.blist.extend(
          [
            f"{interaction[0]} {interaction[1]}  .  {float(sub[2]):.3f}"
            for sub in subs
          ]
        )
        #self.pt.single_print(f"*** len(blist_2b) {len(self.blist)} {self.blist}")

        pot.write(f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} AUTHOR: {author} CITATION:\n")
        pot.write(f"2B {interaction[0]} {interaction[1]} {leading_trim} {trailing_trim}")
        if knots_spacing_type == "uk": pot.write(" uk\n")
        elif knots_spacing_type == "nk": pot.write(" nk\n")
        else: raise ValueError(f"Knot spacing type {knots_spacing_type} not uk or nk")
        pot.write(f"{self.bspline_basis.r_max_map[interaction]} ")
        pot.write(f"{len(self.bspline_basis.knots_map[interaction])}\n")
        pot.write(" ".join([f'{v:.17g}' for v in knots_2b]) + "\n")
        pot.write(f"{self.bspline_basis.get_interaction_partitions()[0][interaction]}\n")
        pot.write(" ".join([f'{v-1:.0f}' for v in range(start_idx, end_idx)]) + "\n")
        pot.write("#\n")


      if 3 in self.bspline_basis.interactions_map:
        for interaction in self.bspline_basis.interactions_map[3]:
          km = self.bspline_basis.knots_map[interaction]
          length = self.bspline_basis.get_interaction_partitions()[0][interaction]
          start_idx = self.bspline_basis.get_interaction_partitions()[1][interaction] + 1
          indices = list(range(start_idx, start_idx + length))
          decompressed = self.bspline_basis.decompress_3B(indices, interaction)
          d0, d1, d2 = decompressed.shape
          exp_ij, exp_ik, exp_jk = d0 + 4, d1 + 4, d2 + 4
          nk_ij, nk_ik, nk_jk = len(km[0]), len(km[1]), len(km[2])
          if (nk_ij, nk_ik, nk_jk) != (exp_ij, exp_ik, exp_jk):
            raise RuntimeError(
              f"UF3 3B {interaction}: knot lengths (ij,ik,jk)=({nk_ij},{nk_ik},{nk_jk}) "
              f"!= coefficient grid+4 ({exp_ij},{exp_ik},{exp_jk}). "
              "LAMMPS requires nknots = ncoef_per_dim + 4 for each 3B dimension."
            )

          pot.write(f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} AUTHOR: {author} CITATION:\n")
          pot.write(f"3B {interaction[0]} {interaction[1]} {interaction[2]} {leading_trim} {trailing_trim}")
          if knots_spacing_type == "uk": pot.write(" uk\n")
          elif knots_spacing_type == "nk": pot.write(" nk\n")
          else: raise ValueError(f"Knot spacing type {knots_spacing_type} not uk or nk")
          pot.write(f"{self.bspline_basis.r_max_map[interaction][2]} ")
          pot.write(f"{self.bspline_basis.r_max_map[interaction][1]} ")
          pot.write(f"{self.bspline_basis.r_max_map[interaction][0]} ")
          pot.write(f"{nk_jk} {nk_ik} {nk_ij} \n")
          pot.write(" ".join([f'{v:.17g}' for v in km[2]]) + "\n")
          pot.write(" ".join([f'{v:.17g}' for v in km[1]]) + "\n")
          pot.write(" ".join([f'{v:.17g}' for v in km[0]]) + "\n")

          pot.write(f"{decompressed.shape[0]} {decompressed.shape[1]} {decompressed.shape[2]}\n")
          for i in range(decompressed.shape[0]):
            for j in range(decompressed.shape[1]):
              pot.write(' '.join([f"{v-1:.0f}" for v in decompressed[i,j]]) + "\n")

          # One blist / rank entry per compressed 3B coefficient (same mask as
          # ``get_feature_partition_sizes``). Using ``decompressed[i,j,k] > 1`` after
          # ``decompress_3B(range(...))`` is wrong: those values are global indices,
          # so symmetry fill marks far too many cells and breaks SLATE ARD reshapes.
          tm = np.asarray(self.bspline_basis.template_mask[interaction], dtype=np.intp)
          fw = np.asarray(self.bspline_basis.flat_weights[interaction])
          for c in np.where(fw > 0)[0]:
            flat_idx = int(tm[c])
            # Grid axes follow ``decompress_3B``: (l,m,n) = ``knots_map`` (km[0],km[1],km[2]).
            # Pot file prints km[2], km[1], km[0]; label with (gk,gj,gi) not (gi,gj,gk) or
            # asymmetric resolutions (e.g. [10,10,20]) index the wrong knot length and crash.
            gi, gj, gk = np.unravel_index(flat_idx, (d0, d1, d2))
            self.basis_ranks.append(2)
            self.blist.append(
              f"{interaction[0]} {interaction[1]} {interaction[2]}  "
              f"{km[2][gk]:.3f}  {km[1][gj]:.3f}  {km[0][gi]:.3f}"
            )

          pot.write("#\n")


    if len(self.blist) != self.ncoeff or len(self.basis_ranks) != self.ncoeff:
      raise RuntimeError(
        f"UF3 blist / basis_ranks length mismatch: len(blist)={len(self.blist)}, "
        f"len(basis_ranks)={len(self.basis_ranks)}, ncoeff={self.ncoeff}"
      )
