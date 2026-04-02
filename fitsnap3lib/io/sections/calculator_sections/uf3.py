import json
import os
import importlib.util
import numpy as np

from fitsnap3lib.io.sections.sections import Section


def _load_write_uf3_lammps_pot_files():
  """Load write_uf3_lammps_pot_files from the uf3 package (pip or source tree)."""
  try:
    import uf3.representation.bspline as bsp
    root = os.path.dirname(os.path.dirname(bsp.__file__))
    path = os.path.abspath(os.path.join(root, "..", "lammps_plugin", "scripts", "generate_uf3_lammps_pots.py"))
    if not os.path.isfile(path):
      return None
    spec = importlib.util.spec_from_file_location("_uf3_gen_uf3pots", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.write_uf3_lammps_pot_files
  except Exception:
    return None


class Uf3(Section):
  """
  UF3 B-spline basis: builds ``BSplineBasis``, ``ncoeff``, type mapping, and writes
  ``{OUTFILE.potential_name}.uf3`` (template) for LAMMPS ``compute ... uf3/kk/host``.
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

    from uf3.data.composition import ChemicalSystem
    from uf3.representation.bspline import BSplineBasis
    from uf3.util import json_io

    elements_str = self.get_value(sec, "elements", "Al Ni")
    self.elements = elements_str.split()
    self.numtypes = len(self.elements)
    self.degree = self.get_value(sec, "degree", "2", "int")
    if self.degree != 2:
      raise RuntimeError("FitSNAP UF3 with compute uf3/kk supports degree = 2 only (2-body .uf3).")

    self.chemical_system = ChemicalSystem(element_list=self.elements, degree=self.degree)

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

    self.ncoeff = int(np.sum(self.bspline_basis.get_feature_partition_sizes()))
    self.fit_offsets = int(fit_offsets)
    self.bzeroflag = self.get_value(sec, "bzeroflag", "1", "bool")
    self.type_mapping = {el: i + 1 for i, el in enumerate(self.elements)}

    self.pt.single_print(
      f"----------------------------------------------------------------\n"
      f"  UF3 B-spline basis\n    elements {' '.join(self.elements)}  degree {self.degree}  ncoeff {self.ncoeff}\n"
      f"----------------------------------------------------------------\n"
    )

    self._write_template_potential()

  @staticmethod
  def _parse_trim(val):
    if isinstance(val, str) and val.strip().startswith("{"):
      return json.loads(val.replace("'", '"'))
    return int(val)

  def _write_template_potential(self):
    from uf3.regression.least_squares import WeightedLinearModel
    from os import path

    writer = _load_write_uf3_lammps_pot_files()
    if writer is None:
      raise RuntimeError(
        "Could not load write_uf3_lammps_pot_files from uf3 (pip install uf3 or use a full source tree)."
      )

    potential_name = Section.sections["OUTFILE"].potential_name
    if not potential_name:
      raise RuntimeError("OUTFILE potential is required for UF3 template path")

    pot_dir = Section.get_outfile_directory(self)
    fname = f"{potential_name}.uf3"
    out_path = path.join(pot_dir, fname) if pot_dir else fname

    model = WeightedLinearModel(bspline_config=self.bspline_basis)
    model.coefficients = np.zeros(model.n_feats)

    @self.pt.rank_zero
    def _write():
      writer(
        chemical_sys=model.bspline_config.chemical_system,
        model=model,
        knots_spacing_type="nk",
        pot_dir=pot_dir or ".",
        uf3_lammps_pot_name=fname,
        author="FitSNAP",
        lammps_units=Section.sections["REFERENCE"].units,
      )

    _write()
    self.pt.all_barrier()

    self.template_uf3_path = path.abspath(out_path)
    self.potential_element_args = " ".join(self.elements)
