import os
import numpy as np

from fitsnap3lib.io.outputs.outputs import Output, optional_open
from fitsnap3lib.io.sections.sections import Section


def _load_write_uf3_lammps_pot_files():
  try:
    import importlib.util
    import uf3.representation.bspline as bsp
    root = os.path.dirname(os.path.dirname(bsp.__file__))
    path = os.path.abspath(os.path.join(root, "..", "lammps_plugin", "scripts", "generate_uf3_lammps_pots.py"))
    if not os.path.isfile(path):
      return None
    spec = importlib.util.spec_from_file_location("_uf3_gen_uf3pots_out", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.write_uf3_lammps_pot_files
  except Exception:
    return None


class Uf3(Output):
  """Write UF3 ``model.json`` (via ``WeightedLinearModel.to_json``) and fitted ``.uf3``."""

  def __init__(self, name, pt, config):
    super().__init__(name, pt, config)

  def output(self, coeffs, errors):
    if self.config.sections["CALCULATOR"].nonlinear:
      self.write_nn(errors)
    else:
      self.write(coeffs, errors)

  def write(self, coeffs, errors):
    @self.pt.rank_zero
    def decorated_write():
      self.write_lammps(coeffs)
      self.write_errors(errors)
    decorated_write()

  def write_nn(self, errors):
    @self.pt.rank_zero
    def decorated_write():
      self.write_errors_nn(errors)
    decorated_write()

  def write_lammps(self, coeffs):
    from uf3.regression.least_squares import WeightedLinearModel

    calc = self.config.sections["CALCULATOR"].calculator.upper()
    if calc != "LAMMPSUF3":
      raise TypeError("UF3 output style requires calculator LAMMPSUF3")

    uf3s = Section.sections["UF3"]
    if not uf3s.bzeroflag:
      raise NotImplementedError("UF3 output currently requires [UF3] bzeroflag = 1")

    writer = _load_write_uf3_lammps_pot_files()
    if writer is None:
      raise RuntimeError("Could not load write_uf3_lammps_pot_files from the uf3 package.")

    model = WeightedLinearModel(bspline_config=uf3s.bspline_basis)
    c = np.asarray(coeffs).ravel()
    if c.size < model.n_feats:
      raise ValueError(f"Expected at least {model.n_feats} coefficients, got {c.size}")
    model.coefficients = c[: model.n_feats]

    outsec = Section.sections["OUTFILE"]
    potential_name = outsec.potential_name
    odir = Section.get_outfile_directory(outsec)
    os.makedirs(odir, exist_ok=True)

    json_path = os.path.join(odir, f"{potential_name}_model.json")
    model.to_json(json_path)

    writer(
      chemical_sys=model.bspline_config.chemical_system,
      model=model,
      knots_spacing_type="nk",
      pot_dir=odir,
      uf3_lammps_pot_name=f"{potential_name}.uf3",
      author="FitSNAP",
      lammps_units=Section.sections["REFERENCE"].units,
    )

    coeff_path = os.path.join(odir, f"{potential_name}.uf3coeff")
    with optional_open(coeff_path, "wt") as fp:
      fp.write(f"# FitSNAP UF3 linear fit coefficients (flattened, length {model.n_feats})\n")
      for i, v in enumerate(model.coefficients):
        fp.write(f"{v:.18g}\n")
