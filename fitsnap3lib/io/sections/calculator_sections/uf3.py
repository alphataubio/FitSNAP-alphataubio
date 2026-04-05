import json
import numpy as np

from fitsnap3lib.io.sections.sections import Section
from fitsnap3lib.lib.uf3.generate_uf3_lammps_pots import write_uf3_lammps_pot_files


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
      "elements",
      "degree",
      "knot_strategy",
      "r_min",
      "r_max",
      "resolution",
      "r_min_map_json",
      "r_max_map_json",
      "resolution_map_json",
      "knots_map_json",
      "leading_trim",
      "trailing_trim",
      "fit_offsets",
      "bzeroflag",
    ]

    sec = self.name.upper()
    for key in self._config[sec]:
      if key not in allowedkeys:
        raise RuntimeError(f"Unmatched variable in {sec} section of input: {key}")

    from fitsnap3lib.lib.uf3.data.composition import ChemicalSystem
    from fitsnap3lib.lib.uf3.representation.bspline import BSplineBasis
    from fitsnap3lib.lib.uf3.util import json_io

    elements_str = self.get_value(sec, "elements", "Al Ni")
    user_elements = elements_str.split()
    self.degree = self.get_value(sec, "degree", "2", "int")
    if self.degree not in (2, 3):
      raise RuntimeError(
        "FitSNAP UF3: degree must be 2 (2-body .uf3 only) or 3 (2- and 3-body). "
        f"Got degree = {self.degree}."
      )
    # LAMMPS ``pair_style uf3`` / ``compute uf3`` nbody: 2 or 3
    self.lammps_nbody = 3 if self.degree > 2 else 2

    # ChemicalSystem sorts symbols (electronegativity / Z). LAMMPS ``compute uf3 E1 E2 ...`` maps
    # type 1 -> first symbol, etc., and column order follows nested type loops (1,1)..(1,nt)..(nt,nt).
    # BSpline partitions use the same sorted ``element_list``. Using the raw input order here
    # desynchronizes A-matrix columns from BSpline 1-body / 2-body layout and destroys the fit.
    self.chemical_system = ChemicalSystem(element_list=user_elements, degree=self.degree)
    self.elements = list(self.chemical_system.element_list)
    self.numtypes = len(self.elements)
    if user_elements != self.elements:
      self.pt.single_print(
        "[UF3] Using canonical element order for LAMMPS types and BSpline partitions: "
        + f"{' '.join(self.elements)} (input was: {' '.join(user_elements)})\n"
      )

    fit_offsets = self.get_value(sec, "fit_offsets", "0", "bool")
    knot_strategy = self.get_value(sec, "knot_strategy", "linear")

    r_min_map = r_max_map = resolution_map = knots_map = None
    if self._config.has_option(sec, "r_min_map_json"):
      r_min_map = json_io.load_interaction_map(self._config.get(sec, "r_min_map_json"))
    if self._config.has_option(sec, "r_max_map_json"):
      r_max_map = json_io.load_interaction_map(self._config.get(sec, "r_max_map_json"))
    if self._config.has_option(sec, "resolution_map_json"):
      resolution_map = json_io.load_interaction_map(self._config.get(sec, "resolution_map_json"))
    if self._config.has_option(sec, "knots_map_json"):
      knots_map = json_io.load_interaction_map(self._config.get(sec, "knots_map_json"))["knots"]

    if r_max_map is None and self._config.has_option(sec, "r_max"):
      r_max = float(self._config.get(sec, "r_max"))
      r_max_map = {t: r_max for t in self.chemical_system.interactions_map[2]}
    if r_min_map is None and self._config.has_option(sec, "r_min"):
      r_min = float(self._config.get(sec, "r_min"))
      r_min_map = {t: r_min for t in self.chemical_system.interactions_map[2]}
    if resolution_map is None and self._config.has_option(sec, "resolution"):
      res = int(self._config.get(sec, "resolution"))
      resolution_map = {t: res for t in self.chemical_system.interactions_map[2]}

    lt = tt = None
    if self._config.has_option(sec, "leading_trim"):
      lt = self._parse_trim(self._config.get(sec, "leading_trim"))
    if self._config.has_option(sec, "trailing_trim"):
      tt = self._parse_trim(self._config.get(sec, "trailing_trim"))
    if tt is None:
      tt = 3

    self.bspline_basis = BSplineBasis(
      self.chemical_system,
      r_min_map=r_min_map,
      r_max_map=r_max_map,
      resolution_map=resolution_map,
      knot_strategy=knot_strategy,
      offset_1b=bool(fit_offsets),
      leading_trim=lt,
      trailing_trim=tt,
      knots_map=knots_map,
    )

    # get_feature_partition_sizes() = [1]*numtypes (composition / 1-body) then 2-body blocks
    # (then 3-body if degree>2). LAMMPS ``compute uf3`` exposes 2-body columns, then 3-body
    # (same total count as these spline blocks when nbody=3 and the template includes 3B).
    _sizes = self.bspline_basis.get_feature_partition_sizes()
    self.nfeats = int(np.sum(_sizes))
    self.ncoeff = int(np.sum(_sizes[self.numtypes :]))
    self.fit_offsets = int(fit_offsets)
    self.bzeroflag = self.get_value(sec, "bzeroflag", "1", "bool")
    self.type_mapping = {el: i + 1 for i, el in enumerate(self.elements)}
    self.potential_element_args = " ".join(self.elements)
    self._uf3_template_abs_path = None

    self.pt.single_print(
      f"----------------------------------------------------------------\n"
      f"  UF3 B-spline basis\n    elements {' '.join(self.elements)}  degree {self.degree}  "
      f"LAMMPS nbody {self.lammps_nbody}  ncoeff {self.ncoeff} (descriptor cols); "
      f"full_features {self.nfeats}\n"
      f"----------------------------------------------------------------\n"
    )

  @staticmethod
  def _parse_trim(val):
    if isinstance(val, str) and val.strip().startswith("{"):
      return json.loads(val.replace("'", '"'))
    return int(val)

  @property
  def template_uf3_path(self):
    """Absolute path to the zero-coeff template ``.uf3`` (created lazily)."""
    self._ensure_uf3_template()
    return self._uf3_template_abs_path

  def _ensure_uf3_template(self):
    from os import path

    if self._uf3_template_abs_path is not None:
      return

    from fitsnap3lib.lib.uf3.regression.least_squares import WeightedLinearModel

    raw_pot = self.get_value("OUTFILE", "potential", None)
    if raw_pot is None or str(raw_pot).strip() == "":
      raise RuntimeError(
        "[UF3] needs [OUTFILE] potential = <basename> in the same input (any section order)."
      )

    potential_base = self.check_path(str(raw_pot))
    lammps_units = str(self.get_value("REFERENCE", "units", "metal")).lower()

    pot_dir = Section.get_outfile_directory(self)
    fname = f"{potential_base}-fit.uf3"
    out_path = path.join(pot_dir, fname) if pot_dir else fname

    model = WeightedLinearModel(bspline_config=self.bspline_basis)
    model.coefficients = np.zeros(model.n_feats)

    @self.pt.rank_zero
    def _write():
      write_uf3_lammps_pot_files(
        chemical_sys=model.bspline_config.chemical_system,
        model=model,
        knots_spacing_type="nk",
        pot_dir=pot_dir or ".",
        uf3_lammps_pot_name=fname,
        author="FitSNAP",
        lammps_units=lammps_units,
      )

    _write()
    self.pt.all_barrier()

    self._uf3_template_abs_path = path.abspath(out_path)
