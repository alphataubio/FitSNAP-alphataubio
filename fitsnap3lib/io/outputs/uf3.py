import os
import numpy as np

from fitsnap3lib.io.outputs.outputs import Output, optional_open
from fitsnap3lib.io.sections.sections import Section
from fitsnap3lib.lib.uf3.generate_uf3_lammps_pots import write_uf3_lammps_pot_files


class Uf3(Output):
  """
  Write UF3 ``model.json`` (via ``WeightedLinearModel.to_json``) and fitted ``.uf3``.

  Supports ``[UF3] bzeroflag = 0`` (composition + 2-body columns in ``A``) and ``1``
  (2-body columns only; 1-body slots in the exported vector are set to zero).
  """

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
    from fitsnap3lib.lib.uf3.regression.least_squares import WeightedLinearModel

    calc = self.config.sections["CALCULATOR"].calculator.upper()
    if calc != "LAMMPSUF3":
      raise TypeError("UF3 output style requires calculator LAMMPSUF3")

    uf3s = Section.sections["UF3"]
    model = WeightedLinearModel(bspline_config=uf3s.bspline_basis)
    c = np.asarray(coeffs, dtype=float).ravel()
    nt = uf3s.numtypes
    n2 = uf3s.ncoeff
    ntot = uf3s.nfeats
    if ntot != nt + n2:
      raise RuntimeError(f"UF3 nfeats ({ntot}) != numtypes ({nt}) + ncoeff ({n2})")

    if uf3s.bzeroflag:
      # Design matrix has only LAMMPS spline columns (2-body; plus 3-body if degree>2); 1-body implicit.
      if c.size < n2:
        raise ValueError(
          f"UF3 bzeroflag=1: expected at least {n2} fitted spline coefficients, got {c.size}"
        )
      full = np.zeros(ntot, dtype=float)
      full[nt:] = c[:n2]
      model.coefficients = full
    else:
      # Design matrix is [composition, spline columns] — same order as BSplineBasis partition_sizes
      # (1 per element, then 2-body blocks, then 3-body if degree>2).
      if c.size < ntot:
        raise ValueError(
          f"UF3 bzeroflag=0: expected at least {ntot} coefficients "
          f"({nt} composition + {n2} 2-body), got {c.size}"
        )
      model.coefficients = c[:ntot].copy()

    outsec = Section.sections["OUTFILE"]
    potential_name = outsec.potential_name
    odir = Section.get_outfile_directory(outsec)
    os.makedirs(odir, exist_ok=True)

    json_path = os.path.join(odir, f"{potential_name}_model.json")
    model.to_json(json_path)

    write_uf3_lammps_pot_files(
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
      fp.write(
        f"# FitSNAP UF3 linear fit coefficients (flattened, length {model.n_feats}; "
        f"bzeroflag={int(uf3s.bzeroflag)})\n"
      )
      for i, v in enumerate(np.asarray(model.coefficients).ravel()):
        fp.write(f"{v:.18g}\n")
