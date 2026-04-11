import os
import numpy as np
from datetime import datetime


from fitsnap3lib.io.outputs.outputs import Output, optional_open
from fitsnap3lib.io.sections.sections import Section


class Uf3(Output):
  """
  Write UF3 potential.

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

    calc = self.config.sections["CALCULATOR"].calculator.upper()
    if calc != "LAMMPSUF3":
      raise TypeError("UF3 output style requires calculator LAMMPSUF3")

    outfile_section = Section.sections["OUTFILE"]
    potential_name = outfile_section.potential_name
    odir = Section.get_outfile_directory(outfile_section)
    pdir = os.path.dirname(potential_name) or odir
    stem = os.path.basename(potential_name)
    os.makedirs(pdir, exist_ok=True)
    self.write_uf3_lammps_pot(f"{stem}.uf3", coeffs)

    uf3_section = Section.sections["UF3"]
    coeff_path = os.path.join(pdir, f"{stem}.uf3coeff")
    with optional_open(coeff_path, "wt") as fp:
      fp.write(
        f"# FitSNAP UF3 linear fit coefficients (flattened, length {len(coeffs)}; "
        f"bzeroflag={int(uf3_section.bzeroflag)})\n\n"
      )
      fp.write("\n".join(f" {coeff:<30.18} #  {bname}" for coeff, bname in zip(coeffs, uf3_section.blist)))


  def write_uf3_lammps_pot(self, path, coeffs, knots_spacing_type="nk"):
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
    bspline_basis = Section.sections["UF3"].bspline_basis

    with open(path, "w") as pot:

      for interaction in bspline_basis.chemical_system.interactions_map[2]:
        pot.write(f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} AUTHOR: {author} CITATION:\n")
        pot.write(f"2B {interaction[0]} {interaction[1]} {leading_trim} {trailing_trim}")
        if knots_spacing_type == "uk": pot.write(" uk\n")
        elif knots_spacing_type == "nk": pot.write(" nk\n")
        else: raise ValueError(f"Knot spacing type {knots_spacing_type} not uk or nk")
        pot.write(f"{bspline_basis.r_max_map[interaction]} ")
        pot.write(f"{len(bspline_basis.knots_map[interaction])}\n")
        pot.write(" ".join([f'{v:.17g}' for v in bspline_basis.knots_map[interaction]]) + "\n")
        pot.write(f"{bspline_basis.get_interaction_partitions()[0][interaction]}\n")
        start_idx = bspline_basis.get_interaction_partitions()[1][interaction]
        length = bspline_basis.get_interaction_partitions()[0][interaction]
        pot.write(" ".join([f'{v:.17g}' for v in coeffs[start_idx:start_idx + length]]) + "\n")
        pot.write("#\n")

      if 3 in bspline_basis.interactions_map:
        for interaction in bspline_basis.interactions_map[3]:

          length = bspline_basis.get_interaction_partitions()[0][interaction]
          start_idx = bspline_basis.get_interaction_partitions()[1][interaction]
          decompressed = bspline_basis.decompress_3B(coeffs[start_idx:start_idx + length], interaction)

          pot.write(f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} AUTHOR: {author} CITATION:\n")
          pot.write(f"3B {interaction[0]} {interaction[1]} {interaction[2]} {leading_trim} {trailing_trim}")
          if knots_spacing_type == "uk": pot.write(" uk\n")
          elif knots_spacing_type == "nk": pot.write(" nk\n")
          else: raise ValueError(f"Knot spacing type {knots_spacing_type} not uk or nk")
          pot.write(f"{bspline_basis.r_max_map[interaction][2]} ")
          pot.write(f"{bspline_basis.r_max_map[interaction][1]} ")
          pot.write(f"{bspline_basis.r_max_map[interaction][0]} ")
          pot.write(f"{len(bspline_basis.knots_map[interaction][2])} ")
          pot.write(f"{len(bspline_basis.knots_map[interaction][1])} ")
          pot.write(f"{len(bspline_basis.knots_map[interaction][0])} \n")
          pot.write(" ".join(['{v:.17g}' for v in bspline_basis.knots_map[interaction][2]]) + "\n")
          pot.write(" ".join(['{v:.17g}' for v in bspline_basis.knots_map[interaction][1]]) + "\n")
          pot.write(" ".join(['{v:.17g}' for v in bspline_basis.knots_map[interaction][0]]) + "\n")

          pot.write(f"{decompressed.shape[0]} {decompressed.shape[1]} {decompressed.shape[2]}\n")
          for i in range(decompressed.shape[0]):
            for j in range(decompressed.shape[1]):
              pot.write(' '.join(map(str, decompressed[i,j])) + "\n")

          pot.write("#\n")


