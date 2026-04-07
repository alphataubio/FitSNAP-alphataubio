import os
import numpy as np
from datetime import datetime


from fitsnap3lib.io.outputs.outputs import Output, optional_open
from fitsnap3lib.io.sections.sections import Section


class Uf3(Output):
  """
  Write UF3 ``model.json`` (via ``WeightedLinearModel.to_json``) and fitted ``.uf3``.

  Supports ``[UF3] bzeroflag = 0`` (composition + 2-body columns in ``A``) and ``1``
  (2-body columns only; 1-body slots in the exported vector are set to zero).
  """

  def __init__(self, name, pt, config):
    super().__init__(name, pt, config)

  def output(self, coeffs, errors):
    @self.pt.rank_zero
    def decorated_write():
      self.write_lammps(coeffs)
      self.write_errors(errors)
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
    pdir = os.path.dirname(potential_name) or odir
    stem = os.path.basename(potential_name)
    os.makedirs(pdir, exist_ok=True)

    json_path = os.path.join(pdir, f"{stem}_model.json")
    model.to_json(json_path)

    self.write_uf3_lammps_pot(model.bspline_config.chemical_system, model, f"{stem}.uf3")

    uf3_section = Section.sections["UF3"]
    if uf3_section.bzeroflag: coeff_names = uf3_section.blist
    else: coeff_names = [[0]]+uf3_section.blist
    coeff_path = os.path.join(pdir, f"{stem}.uf3coeff")
    with optional_open(coeff_path, "wt") as fp:
      fp.write(
        f"# FitSNAP UF3 linear fit coefficients (flattened, length {model.n_feats}; "
        f"bzeroflag={int(uf3s.bzeroflag)})\n\n"
      )
      # np.asarray(model.coefficients).ravel()
      fp.write("\n".join(f" {coeff:<30.18} #  {bname}" for coeff, bname in zip(coeffs, coeff_names)))


  def write_uf3_lammps_pot(self, chemical_sys, model, path, knots_spacing_type="nk"):
    """Returns list

    Creates and writes UF3 lammps potential files. Takes UF3 composition object,
    UF3 model, knots_spacing_type, name of potential directory, name of uf3
    lammps pot file to be generateas, author and lammps units input. Will
    overwrite the files if files with the same exists
    """
    
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    author = "FitSNAP"
    leading_trim = 0
    trailing_trim = 3
    lammps_units = Section.sections["REFERENCE"].units

    with open(path, "w") as pot:

      for interaction in chemical_sys.interactions_map[2]:
        pot.write(f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} AUTHOR: {author} CITATION:\n")
        pot.write(f"2B {interaction[0]} {interaction[1]} {leading_trim} {trailing_trim}")
        if knots_spacing_type == "uk": pot.write(" uk\n")
        elif knots_spacing_type == "nk": pot.write(" nk\n")
        else: raise ValueError(f"Knot spacing type {knots_spacing_type} not uk or nk")
        pot.write(f"{model.bspline_config.r_max_map[interaction]} ")
        pot.write(f"{len(model.bspline_config.knots_map[interaction])}\n")
        pot.write(" ".join([f'{v:.17g}' for v in model.bspline_config.knots_map[interaction]]) + "\n")
        pot.write(f"{model.bspline_config.get_interaction_partitions()[0][interaction]}\n")
        start_index = model.bspline_config.get_interaction_partitions()[1][interaction]
        length = model.bspline_config.get_interaction_partitions()[0][interaction]
        #pot.write(" ".join([f'{v:.17g}' for v in model.coefficients[start_index:start_index + length]]) + "\n")
        pot.write(" ".join([f'1' for v in model.coefficients[start_index:start_index + length]]) + "\n")
        pot.write("#\n")

      if 3 in model.bspline_config.interactions_map:
        for interaction in model.bspline_config.interactions_map[3]:
          pot.write(f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} AUTHOR: {author} CITATION:\n")
          pot.write(f"3B {interaction[0]} {interaction[1]} {interaction[2]} {leading_trim} {trailing_trim}")
          if knots_spacing_type == "uk": pot.write(" uk\n")
          elif knots_spacing_type == "nk": pot.write(" nk\n")
          else: raise ValueError(f"Knot spacing type {knots_spacing_type} not uk or nk")
          pot.write(f"{model.bspline_config.r_max_map[interaction][2]} ")
          pot.write(f"{model.bspline_config.r_max_map[interaction][1]} ")
          pot.write(f"{model.bspline_config.r_max_map[interaction][0]} ")
          pot.write(f"{len(model.bspline_config.knots_map[interaction][2])} ")
          pot.write(f"{len(model.bspline_config.knots_map[interaction][1])} ")
          pot.write(f"{len(model.bspline_config.knots_map[interaction][0])} \n")
          pot.write(" ".join(['{v:.17g}' for v in model.bspline_config.knots_map[interaction][2]]) + "\n")
          pot.write(" ".join(['{v:.17g}' for v in model.bspline_config.knots_map[interaction][1]]) + "\n")
          pot.write(" ".join(['{v:.17g}' for v in model.bspline_config.knots_map[interaction][0]]) + "\n")

          solutions = least_squares.arrange_coefficients(model.coefficients, model.bspline_config)
          decompressed = model.bspline_config.decompress_3B( \
            solutions[(interaction[0], interaction[1],interaction[2])], \
            (interaction[0], interaction[1],interaction[2]))

          pot.write(f"{decompressed.shape[0]} {decompressed.shape[1]} {decompressed.shape[2]}\n")
          for i in range(decompressed.shape[0]):
            for j in range(decompressed.shape[1]):
              pot.write(' '.join(map(str, decompressed[i,j])) + "\n")

          pot.write("#\n")


