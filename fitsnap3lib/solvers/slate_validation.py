from fitsnap3lib.solvers.slate_common import SlateCommon

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as pe
from collections import Counter, defaultdict
import os, re, json, io, base64, adios2

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

CPK_COLORS = {
  "H":  "#FFFFFF", "He": "#D9FFFF",
  "Li": "#CC80FF", "Be": "#C2FF00", "B": "#FFB5B5", "C": "#000000", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "Ne": "#B3E3F5",
  "Na": "#AB5CF2", "Mg": "#8AFF00", "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F", "Ar": "#80D1E3",
  "K":  "#8F40D4", "Ca": "#3DFF00",    "Fe": "#E06633",    "Ni": "#50D050",
  "Cu": "#C88033", "Zn": "#7D80B0",    "Br": "#A62929",    "Ag": "#C0C0C0",
  "I":  "#940094", "Au": "#FFD123",
}


try:
  from slate_wrapper import slate_ridge_augmented_qr_cython, slate_ard_update_cython
except ImportError:
  try:
    import sys
    import os
    slate_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib', 'slate_solver')
    if slate_path not in sys.path:
      sys.path.insert(0, slate_path)
    from slate_wrapper import slate_ridge_augmented_qr_cython, slate_ard_update_cython
  except ImportError as e:
    print(f"Warning: Could not import SLATE ARD functions: {e}")
    slate_ard_update_cython = None

try:
  from mpi4py import MPI
except ImportError:
  MPI = None
  
class CompactJSONEncoder(json.JSONEncoder):
  def iterencode(self, o, _one_shot=False):
    # Get the default encoder
    encoder = super().iterencode(o, _one_shot=_one_shot)

    # Accumulate pieces to modify lists only
    inside_list = False
    buffer = []
    for chunk in encoder:
      if chunk.startswith('['):
        inside_list = True
        buffer.append(chunk)
        continue
      if inside_list:
        buffer.append(chunk)
        if chunk.endswith(']'):
          text = ''.join(buffer)
          # Remove newlines and spaces inside the list
          text = text.replace('\n', '').replace('  ', '').replace(' ,', ',').replace(', ', ',')
          yield text
          buffer.clear()
          inside_list = False
        continue
      yield chunk


def _notebook_config_key(k):
  """JSON object keys must be strings"""
  if isinstance(k, str):
    return k
  if isinstance(k, tuple):
    return "(" + ", ".join(str(x) for x in k) + ")"
  return str(k)


def _notebook_config_jsonable(obj, depth=0):
  """Recursively convert config values so ``json.dumps`` succeeds (no tuple dict keys)."""
  if depth > 40:
    return "<max depth>"
  if obj is None or isinstance(obj, (bool, int, float, str)):
    return obj
  if isinstance(obj, (np.integer, np.floating, np.bool_)):
    return obj.item()
  if isinstance(obj, np.ndarray):
    return _notebook_config_jsonable(obj.tolist(), depth + 1)
  if isinstance(obj, dict):
    return {
      _notebook_config_key(k): _notebook_config_jsonable(v, depth + 1)
      for k, v in obj.items()
    }
  if isinstance(obj, (list, tuple)):
    return [_notebook_config_jsonable(x, depth + 1) for x in obj]
  if isinstance(obj, set):
    return [_notebook_config_jsonable(x, depth + 1) for x in sorted(obj, key=str)]
  return str(obj)


class SlateValidation(SlateCommon):

  # --------------------------------------------------------------------------------------------

  def validation_notebook(self):
  
    if self.pt._rank != 0 or not self.config.sections["OUTFILE"].validation: return
    output_prefix = self.config.sections['OUTFILE'].metrics.replace('.md', '')

    if self.config.sections["CALCULATOR"].kokkos: notebook_file = f"{output_prefix}_kk.ipynb"
    else: notebook_file = f"{output_prefix}.ipynb"

            
    # Create Jupyter notebook structure
    notebook = {
      "cells": [],
      "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3" },
        "language_info": {"name": "python", "version": "3.9.0"}
      },
      "nbformat": 4,
      "nbformat_minor": 4
    }
    
    # -------- TITLE --------

    notebook["cells"].append({
      "cell_type": "markdown",
      "metadata": {},
      "source": [f"# SLATE {self.method} Validation Report\n{output_prefix}\n", "\n"]
    })
            
    # -------- CONFIGURATION --------
 
    notebook["cells"].append({
      "cell_type": "markdown",
      "metadata": {
        "collapsed": True,
        "jp-MarkdownHeadingCollapsed": True,
        "jupyter": {"outputs_hidden": True}
      },
      "source": ["## Configuration"]
    })

    config_dict = {}
    for section_name, section in self.config.sections.items():
      if hasattr(section, '__dict__'):
        raw = {k: v for k, v in section.__dict__.items() if not k.startswith('_')}
        config_dict[section_name] = _notebook_config_jsonable(raw)

    config_json = json.dumps(config_dict, cls=CompactJSONEncoder, indent=2, default=str)
      
    notebook["cells"].append({
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Configuration from FitSNAP run\n",
        f"config = {config_json}\n\n",
      ]
    })

    # -------- SLURM --------
      
    notebook["cells"].append({
      "cell_type": "markdown",
      "metadata": {
        "collapsed": True,
        "jp-MarkdownHeadingCollapsed": True,
        "jupyter": {"outputs_hidden": True}
      },
      "source": ["## SLURM"]
    })
      
    slurm_keys = [f"{k} = {v}" for k,v in sorted(os.environ.items()) if k.startswith("SLURM_")]
    slurm_source = "\n".join(slurm_keys) if len(slurm_keys)>0 else "n/a\n"

    notebook["cells"].append({"cell_type": "raw", "metadata": {}, "source": [slurm_source]})
      
    # -------- ERROR ANALYSIS --------
      
    notebook["cells"].append({ "cell_type": "markdown", "metadata": {}, "source": ["## Error Analysis"]})
      
    # Generate HTML tables from error_analysis DataFrame
    if hasattr(self, 'errors') and self.errors is not None and not self.errors.empty:
      # Determine which row types exist
      row_types = self.errors.index.get_level_values('Subsystem').unique()
          
      for row_type in row_types:
        if row_type in ['Energy', 'Force', 'Stress']:
          # Create a markdown cell for this subsystem
          reference_section = self.config.sections["REFERENCE"]
          if row_type == 'Energy': row_units = f" ({reference_section.error_energy_units})"
          elif row_type == 'Force': row_units = f" ({reference_section.error_force_units})"
          elif row_type == 'Stress': row_units = f" ({reference_section.error_stress_units})"
          else: row_units = ""
          subsystem_lines = [f"### {row_type}{row_units}\n\n"]
                  
          # Get data for this row type
          try:
            df_subset = self.errors.xs(row_type, level='Subsystem')
                      
            # Process weighted and unweighted separately
            for weighting in ['Unweighted', 'weighted']:
              try:
                df_weight = df_subset.xs(weighting, level='Weighting')
                # Pivot to get training and testing side by side
                df_pivot = df_weight.reset_index()
                # Create separate dataframes for training and testing
                df_train = df_pivot[df_pivot['Testing'] == 'Training'].set_index('Group')
                df_test = df_pivot[df_pivot['Testing'] == 'Testing'].set_index('Group')
                # Merge them
                df_combined = df_train[['ncount', 'mae', 'rmse', 'rsq']].join(
                  df_test[['ncount', 'mae', 'rmse', 'rsq']],
                  how='outer',
                  rsuffix='_test'
                )
                              
                # Sort by Test RMSE descending, but keep *ALL at top
                all_key = '*ALL'
                if all_key in df_combined.index:
                  df_all = df_combined.loc[[all_key]]
                  df_rest = df_combined.drop(all_key)
                else:
                  df_all = pd.DataFrame()
                  df_rest = df_combined
                              
                df_rest = df_rest.sort_values('rmse_test', ascending=False)
                df_combined = pd.concat([df_all, df_rest])
                              
                # Build HTML table in booktabs style
                subsystem_lines.append(f"**{weighting.capitalize()} Metrics**\n\n")
                html = '<table style="border-collapse: collapse; table-layout: auto; width: 100%; font-size: 14px;">\n'
                              
                # Header row with toprule
                fmt = "padding: 6px 10px; white-space: nowrap;"
                fmt_n = fmt + " text-align: center; width: 5%;"
                fmt_left = fmt + " text-align: left; width: 24%;"
                fmt_right = fmt + " text-align: right; width: 11%;"
                html += '  <thead>\n'
                html += '    <tr style="border-top: 2px solid black; border-bottom: 2px solid black;">\n'
                html += f'      <th style="{fmt_left}">Group</th>\n'
                html += f'      <th style="{fmt_n} ">N</th>\n'
                html += f'      <th style="{fmt_right}">MAE</th>\n'
                html += f'      <th style="{fmt_right}">RMSE</th>\n'
                html += f'      <th style="{fmt_right}">R²</th>\n'
                html += f'      <th style="{fmt_n}">N</th>\n'
                html += f'      <th style="{fmt_right}">MAE</th>\n'
                html += f'      <th style="{fmt_right}">RMSE (&darr;)</th>\n'
                html += f'      <th style="{fmt_right}">R²</th>\n'
                html += '    </tr>\n'
                html += '  </thead>\n'
                              
                # Body with Training/Testing labels and cmidrules
                html += '  <tbody>\n'
                html += '    <tr>\n'
                html += '      <td style="padding: 4px 12px;"></td>\n'
                html += '      <td colspan="4" style="text-align: center; padding: 4px 12px; font-style: italic; border-bottom: 1px solid #999;">Training</td>\n'
                html += '      <td colspan="4" style="text-align: center; padding: 4px 12px; font-style: italic; border-bottom: 1px solid #999;">Validation</td>\n'
                html += '    </tr>\n'
                              
                # Helper functions to format numbers
                def format_int_value(val):
                  # Handle NaN values
                  if pd.isna(val) or (isinstance(val, float) and np.isnan(val)): return "-"
                  else: return int(val)

                def format_value(val):
                  # Handle NaN values
                  if pd.isna(val) or (isinstance(val, float) and np.isnan(val)): return "-"
                  if abs(val) < 1e-6 and val != 0: return f"{val:.2e}"
                  else: return f"{val:.4f}"

                # Data rows
                for idx, row in df_combined.iterrows():
                  # Check if this is the ALL row
                  is_all = (idx == '*ALL')
                                  
                  # Clean group name and format
                  group_name = idx.replace('*', '')
                  if is_all:
                    group_display = f'<strong>{group_name}</strong>'
                    row_style = 'font-weight: bold;'
                  else:
                    group_display = f'&nbsp;&nbsp;{group_name}'
                    row_style = ''
                                  
                  fmt = "padding: 3px 5px; font-family: monospace; white-space: nowrap; "
                  fmt_n = fmt + "text-align: center; border-left: 5px solid white;"
                  fmt_left = fmt + "text-align: left; overflow: hidden; text-overflow: ellipsis;"
                  fmt_right = fmt + "text-align: right;"
                  html += f'    <tr style="{row_style}">\n'
                  html += f'      <td style="{fmt_left}">{group_display}</td>\n'
                  html += f'      <td style="{fmt_n}">{format_int_value(row["ncount"])}</td>\n'
                  html += f'      <td style="{fmt_right}">{format_value(row["mae"])}</td>\n'
                  html += f'      <td style="{fmt_right}">{format_value(row["rmse"])}</td>\n'
                  html += f'      <td style="{fmt_right}">{format_value(row["rsq"])}</td>\n'
                  html += f'      <td style="{fmt_n}">{format_int_value(row["ncount_test"])}</td>\n'
                  html += f'      <td style="{fmt_right}">{format_value(row["mae_test"])}</td>\n'
                  html += f'      <td style="{fmt_right}">{format_value(row["rmse_test"])}</td>\n'
                  html += f'      <td style="{fmt_right}">{format_value(row["rsq_test"])}</td>\n'
                  html += '    </tr>\n'

                html += '    <tr style="border-bottom: 2px solid black;"><td colspan="9"></td></tr>\n'
                html += '  </tbody>\n'
                html += '</table>\n\n'

                subsystem_lines.append(html)
                subsystem_lines.append("\n")

              except KeyError:
                # This weighting type doesn't exist
                pass

          except KeyError:
            # This row type doesn't exist
            pass
                  
          # Add this subsystem's markdown cell
          notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": subsystem_lines})

    # Create scatterplots for predictions vs truths
    output_prefix = self.config.sections['OUTFILE'].metrics.replace('.md', '')
    adios2_path = f"{output_prefix}.bp"

    try:
      with adios2.FileReader(adios2_path) as adios2_file:
        sorted_group_names = list(adios2_file.read_attribute("sorted_group_names"))
        available_vars = adios2_file.available_variables()
        unique_row_types = set()
        for var_name in available_vars.keys():
          if '_' in var_name:
            parts = var_name.split('_')
            if len(parts) >= 3 and parts[-1] in ['training', 'testing']:
              row_type = parts[0].capitalize()
              unique_row_types.add(row_type)

        for row_type in sorted(unique_row_types):
          if row_type not in ['Energy', 'Force', 'Stress']:
            continue

          reference_section = self.config.sections["REFERENCE"]
          scatter_factor = 1.0
          if row_type == 'Energy':
            if reference_section.units == "metal":
              scatter_factor = 1000.0
              units = reference_section.error_energy_units
            else:
              units = reference_section.energy_units
          elif row_type == 'Force':
            if reference_section.units == "metal":
              scatter_factor = 1000.0
              units = reference_section.error_force_units
            else:
              units = reference_section.force_units
          elif row_type == 'Stress':
            if reference_section.units == "metal":
              scatter_factor = 0.0001
              units = reference_section.error_stress_units
            else:
              units = reference_section.stress_units
          else:
            units = ""

          fig = plt.figure(figsize=(9.5, 5.9), layout='constrained')
          gs = fig.add_gridspec(1, 2, width_ratios=[62, 38])
          ax = fig.add_subplot(gs[0, 0])
          fig.add_subplot(gs[0, 1]).set_visible(False)
          tab20 = plt.cm.get_cmap('tab20')
          all_preds, all_truths = [], []

          for group_idx, group_name in enumerate(sorted_group_names):
            color_idx_light = (group_idx * 2 + 1) % 20
            color_idx_dark = (group_idx * 2) % 20

            row_type_lower = row_type.lower()
            var_name_train = f"{row_type_lower}_{group_idx}_training"
            if var_name_train in available_vars:
              data_train = adios2_file.read(var_name_train, step_selection=[0, 1])
              if len(data_train) > 0:
                truths_train = scatter_factor * data_train[:, 0]
                preds_train = scatter_factor * data_train[:, 1]
                all_preds.extend(preds_train)
                all_truths.extend(truths_train)
                if len(all_preds) > 1:
                  size = 99 / np.log10(len(all_preds))
                else:
                  size = 1
                ax.scatter(
                  truths_train, preds_train, zorder=8, c=[tab20(color_idx_light)], alpha=.8,
                  s=size, edgecolors='none', label=f"{group_name} (train)"
                )

            var_name_test = f"{row_type_lower}_{group_idx}_testing"
            if var_name_test in available_vars:
              data_test = adios2_file.read(var_name_test, step_selection=[0, 1])
              if len(data_test) > 0:
                truths_test = scatter_factor * data_test[:, 0]
                preds_test = scatter_factor * data_test[:, 1]
                all_preds.extend(preds_test)
                all_truths.extend(truths_test)
                npt = max(len(all_preds), 2)
                ax.scatter(
                  truths_test, preds_test, zorder=9, linewidths=0.5,
                  c=[tab20(color_idx_dark)], edgecolors='black', alpha=.95,
                  s=99 / np.log10(npt) * 1.618,
                  label=f"{group_name} (test)", marker='s'
                )

          if len(all_preds) == 0:
            plt.close()
            continue

          all_preds = np.array(all_preds)
          all_truths = np.array(all_truths)
          lo = min(all_truths.min(), all_preds.min())
          hi = max(all_truths.max(), all_preds.max())
          pad = 0.05 * (hi - lo)
          lims = (lo - pad, hi + pad)
          ax.plot(lims, lims, 'k--', alpha=0.618, lw=2, label='Perfect prediction')
          ax.margins(0)

          ax.set_xlabel(f'True {row_type} ({units})', fontsize=14, fontweight='bold')
          ax.set_ylabel(f'Predicted {row_type} ({units})', fontsize=14, fontweight='bold')
          ax.grid(True, alpha=0.3)
          ax.set_aspect('equal', adjustable='box')

          handles, labels = ax.get_legend_handles_labels()
          hl_dict = dict(zip(labels, handles))

          train_handles, train_labels, test_handles, test_labels = [], [], [], []
          spacer = Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible=False)

          for g_name in sorted_group_names:
            t_key = f"{g_name} (train)"
            v_key = f"{g_name} (test)"
            if t_key in hl_dict:
              train_handles.append(hl_dict[t_key])
              train_labels.append(t_key)
            else:
              train_handles.append(spacer)
              train_labels.append("")

            if v_key in hl_dict:
              test_handles.append(hl_dict[v_key])
              test_labels.append(v_key)
            else:
              test_handles.append(spacer)
              test_labels.append("")

          final_handles = train_handles + test_handles
          final_labels = train_labels + test_labels

          pp_key = 'Perfect prediction'
          if pp_key in hl_dict:
            final_handles.append(hl_dict[pp_key])
            final_labels.append(pp_key)

          # Anchor legend to the axes box (equal aspect shrinks ax vertically; a
          # full-height leg_ax would misalign the legend top with the plot top).
          fig.canvas.draw()
          leg = fig.legend(
            handles=final_handles,
            labels=final_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            bbox_transform=ax.transAxes,
            ncol=1,
            borderpad=.618,
            title="GROUPS",
            title_fontsize=12,
            prop={"size": 12}
          )
          leg.get_title().set_fontweight("bold")

          notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {
              "collapsed": False,
              "jp-MarkdownHeadingCollapsed": False,
              "jupyter": {"outputs_hidden": False}
            },
            "source": [f"### {row_type} Scatterplot"]
          })

          buf = io.BytesIO()
          img_format = 'svg' if len(all_truths) < 10000 else 'png'
          plt.savefig(buf, format=img_format, bbox_inches='tight')
          plt.close()
          buf.seek(0)
          img_bytes = buf.getvalue()
          buf.close()
          img_base64 = base64.b64encode(img_bytes).decode('utf-8')

          mime = "image/svg+xml" if img_format == "svg" else "image/png"

          notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
              f'<div align="center"><img src="data:{mime};base64,{img_base64}" '
              'style="width:90%;height:auto;"></div>'
            ]
          })

    except Exception as e:
      self.pt.single_print(f"Could not create scatterplots: {e}")

    self.validation_notebook_rdf(notebook)

    if self.method == 'ARD':
      self.validation_notebook_ard(notebook)

    with open(notebook_file, 'w') as f:
      json.dump(notebook, f, indent=2)

    self.pt.debug_single_print(f"Created validation notebook: {notebook_file}")


  def validation_notebook_rdf(self, notebook):
    output_prefix = self.config.sections['OUTFILE'].metrics.replace('.md', '')
    adios2_path = f"{output_prefix}.bp"

    try:
      with adios2.FileReader(adios2_path) as adios2_file:
        available_attrs = adios2_file.available_attributes()
        if 'rdf_gr' not in available_attrs:
          return

        elements = [
          x.strip() for x in str(adios2_file.read_attribute('rdf_elements')).split(',') if x.strip()
        ]
        n_bins = int(np.asarray(adios2_file.read_attribute('rdf_n_bins')).reshape(-1)[0])
        r_centers = np.asarray(adios2_file.read_attribute('rdf_r_centers'), dtype=np.float64).reshape(-1)
        n_elem = len(elements)
        gr = np.asarray(adios2_file.read_attribute('rdf_gr'), dtype=np.float64).reshape(
          n_elem, n_elem, n_bins
        )
        rcut_in = np.asarray(adios2_file.read_attribute('rdf_rcut_in'), dtype=np.float64).reshape(
          n_elem, n_elem
        )
        rcut = np.asarray(adios2_file.read_attribute('rdf_rcut'), dtype=np.float64).reshape(
          n_elem, n_elem
        )

      notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Radial Distribution Functions\n"]
      })

      for i, elem_i in enumerate(elements):
        fig, ax = plt.subplots(figsize=(9.0, 5.5), layout='constrained')
        ymax = 0.0
        series_rcut = []

        for j, elem_j in enumerate(elements):
          color = CPK_COLORS.get(elem_j, '#888888')
          y = gr[i, j, :]
          ymax = max(ymax, float(np.nanmax(y)) if y.size else 0.0)
          series_rcut.append(float(rcut[i, j]))

          (line,) = ax.plot(
            r_centers, y, color=color, lw=2.2, label=f"{elem_i}-{elem_j}", zorder=3
          )
          line.set_path_effects([
            pe.Stroke(linewidth=4.0, foreground='#222222'),
            pe.Normal(),
          ])

          if rcut_in[i, j] > 0.0:
            ax.axvline(
              rcut_in[i, j], color=color, ls='--', lw=1.2, alpha=0.85, zorder=2
            )
          if rcut[i, j] > 0.0:
            ax.axvline(
              rcut[i, j], color=color, ls='--', lw=1.2, alpha=0.85, zorder=2
            )

        ax.axhline(1.0, color='k', ls=':', lw=1.0, alpha=0.5, zorder=1)
        x_hi = (max(series_rcut) + 1.0) if series_rcut else r_centers[-1]
        ax.set_xlim(0.0, x_hi)
        ax.set_ylim(0.0, ymax * 1.08 if ymax > 0.0 else 1.0)
        ax.set_xlabel('r (Angstrom)', fontsize=13, fontweight='bold')
        ax.set_ylabel('g(r)', fontsize=13, fontweight='bold')
        ax.set_title(f'{elem_i} RDF', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best', fontsize=10, title='Pairs', title_fontsize=10)

        notebook["cells"].append({
          "cell_type": "markdown",
          "metadata": {},
          "source": [f"### {elem_i} RDF"]
        })

        buf = io.BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        notebook["cells"].append({
          "cell_type": "markdown",
          "metadata": {},
          "source": [
            f'<div align="center"><img src="data:image/svg+xml;base64,{img_base64}" '
            'style="width:90%;height:auto;"></div>'
          ]
        })

    except Exception as e:
      self.pt.single_print(f"Could not create RDF plots: {e}")

  def validation_notebook_ard(self, notebook):
      
    # Load data from adios2 to create plots
      
    output_prefix = self.config.sections['OUTFILE'].metrics.replace('.md', '')
    adios2_path = f"{output_prefix}.bp"
    with adios2.FileReader(adios2_path) as adios2_file:
      basis_ranks = adios2_file.read_attribute("basis_ranks")
      blist = adios2_file.read_attribute("blist")
      num_steps = adios2_file.num_steps()-1 # last step was preds/truths
      gamma_history = adios2_file.read("gamma", step_selection=[0,num_steps])
      lambda_history = adios2_file.read("lambda", step_selection=[0,num_steps])
      n_feat = gamma_history.size // num_steps
      if n_feat * num_steps != gamma_history.size:
        raise ValueError(
          f"gamma step size mismatch: size={gamma_history.size}, num_steps={num_steps}"
        )
      blist = list(blist)
      basis_ranks = np.asarray(basis_ranks, dtype=int)
      if len(blist) != n_feat or basis_ranks.size != n_feat:
        self.pt.single_print(
          f"Warning: ADIOS2 blist length {len(blist)} / basis_ranks {basis_ranks.size} "
          f"!= n_features {n_feat} from gamma; aligning to gamma."
        )
        if len(blist) > n_feat:
          blist = blist[:n_feat]
        elif len(blist) < n_feat:
          blist.extend([f"<basis {i}>" for i in range(len(blist), n_feat)])
        if basis_ranks.size > n_feat:
          basis_ranks = basis_ranks[:n_feat]
        elif basis_ranks.size < n_feat:
          pad = int(basis_ranks[-1]) if basis_ranks.size else 0
          basis_ranks = np.concatenate(
            [basis_ranks, np.full(n_feat - basis_ranks.size, pad, dtype=int)]
          )
      gamma_array = gamma_history.reshape((num_steps, n_feat))
      lam = lambda_history.reshape((num_steps, n_feat))
      lambda_array = np.full_like(lam, np.nan, dtype=float)
      pos = (lam > 0) & np.isfinite(lam)
      lambda_array[pos] = np.log10(lam[pos])
      
    blist_rank = defaultdict(list)
    rank_indices = defaultdict(list)

    split_pattern = r" ns \[|\] ls \[|\]| \[0\]$"

    for i, (r, f) in enumerate(zip(basis_ranks, blist)):
      blist_rank[r].append(re.split(split_pattern, f))
      rank_indices[r].append(i)

    gamma_env_html = _build_gamma_env_table_html(gamma_array[-1], blist, basis_ranks)
    if gamma_env_html is not None:
      notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Gamma by Atomic Environment\n\n", gamma_env_html]
      })

    # Summary statistics with side-by-side gamma and lambda distributions
      
    n_iterations, n_features = gamma_array.shape
    # Create summary plots - (n_iterations) x 2 grid
    # Select representative iterations to show
    if n_iterations <= 10:
      iter_indices = list(range(n_iterations))
    else:
      # Show first, last, and evenly spaced middle iterations
      iter_indices = list(range(10)) + list(np.linspace(10, n_iterations-2, min(5, n_iterations-2), dtype=int)) + [n_iterations-1]

    n_rows = len(iter_indices)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5*n_rows))

    cmap_gamma = plt.cm.turbo
    cmap_lambda = plt.cm.turbo.reversed()
      
    # If only one row, axes won't be 2D
    if n_rows == 1: axes = axes.reshape(1, -1)

    for row_idx, iter_idx in enumerate(iter_indices):
      # Left column: Gamma distribution
      ax_gamma = axes[row_idx, 0]
      gamma_at_iter = gamma_array[iter_idx, :]
      gamma_nonzero = gamma_at_iter[gamma_at_iter > 0]
          
      if len(gamma_nonzero) > 0:
        counts, edges, patches = ax_gamma.hist(gamma_nonzero, bins=20, edgecolor='black', alpha=.99)
        ax_gamma.set_xlabel('Gamma Value', fontsize=11)
        ax_gamma.set_ylabel('Number of Features', fontsize=11)
        ax_gamma.set_title(f'Iteration {iter_idx}: Gamma Distribution', fontsize=12, fontweight='bold')
        ax_gamma.grid(True, alpha=0.3, axis='y')

        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        norm_centers = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
        for c, p in zip(norm_centers, patches): p.set_facecolor(cmap_gamma(c))

        # Add statistics text
        stats_text = f'Range: [{gamma_at_iter.min():.3f}, {gamma_at_iter.max():.3f}] '
        stats_text += f'Mean: {gamma_nonzero.mean():.3f}'
        ax_gamma.text(0.98, 0.98, stats_text,
              transform=ax_gamma.transAxes, fontsize=9, verticalalignment='top',
              horizontalalignment='right',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
      else:
        ax_gamma.text(0.5, 0.5, 'No active features', transform=ax_gamma.transAxes,
              ha='center', va='center', fontsize=12)
        ax_gamma.set_title(f'Iteration {iter_idx}: Gamma Distribution', fontsize=12, fontweight='bold')
          
      # Right column: Lambda distribution (log scale)
      ax_lambda = axes[row_idx, 1]
      lambda_at_iter = lambda_array[iter_idx, :]
      lambda_finite = lambda_at_iter[np.isfinite(lambda_at_iter)]

      if len(lambda_finite) > 0:
        counts, edges, patches = ax_lambda.hist(lambda_finite, bins=20, edgecolor='black', alpha=.99)
        ax_lambda.set_xlabel('Log10(Lambda)', fontsize=11)
        ax_lambda.set_ylabel('Number of Features', fontsize=11)
        ax_lambda.set_title(f'Iteration {iter_idx}: Lambda Distribution', fontsize=12, fontweight='bold')
        ax_lambda.grid(True, alpha=0.3, axis='y')

        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        span = bin_centers.max() - bin_centers.min()
        if span > 0: norm_centers = (bin_centers - bin_centers.min()) / span
        else: norm_centers = np.zeros_like(bin_centers)
        for c, p in zip(norm_centers, patches): p.set_facecolor(cmap_lambda(c))

        stats_text = (
          f'Log range: [{lambda_finite.min():.1f}, {lambda_finite.max():.1f}] '
          f'Log mean: {lambda_finite.mean():.1f}'
        )
        if np.any(~np.isfinite(lambda_at_iter)): stats_text += ' (non-positive λ omitted from log plot)'
        ax_lambda.text(0.98, 0.98, stats_text,
              transform=ax_lambda.transAxes, fontsize=9, verticalalignment='top',
              horizontalalignment='right',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
      else:
        ax_lambda.text(0.5, 0.5, 'No positive finite λ (log undefined)',
              transform=ax_lambda.transAxes, ha='center', va='center', fontsize=12)
        ax_lambda.set_title(f'Iteration {iter_idx}: Lambda Distribution', fontsize=12, fontweight='bold')
              
    notebook["cells"].append({
      "cell_type": "markdown",
      "metadata": {"collapsed": False, "jp-MarkdownHeadingCollapsed": False, "jupyter": {"outputs_hidden": False}},
      "source": [f"## Gamma / Lambda Histograms (by iteration)"]
    })
      
    buf = io.BytesIO()
    img_format = 'png' #if len(all_truths) < 10000 else 'svg'
    plt.savefig(buf, format=img_format, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_bytes = buf.getvalue()
    buf.close()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    mime = "image/svg+xml" if img_format == "svg" else "image/png"

    notebook["cells"].append({
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        f'<div align="center"><img src="data:{mime};base64,{img_base64}" '
        'style="width:100%;height:auto;"></div>'
      ]
    })
    

    # Create and display lambda heatmaps
      
    notebook["cells"].append({
      "cell_type": "markdown",
      "metadata": {"collapsed": False, "jp-MarkdownHeadingCollapsed": False, "jupyter": {"outputs_hidden": False}},
      "source": ["## Lambda Heatmaps\n"]
    })

    threshold = np.log10(self.threshold_lambda)
          
    for rank in range(int(basis_ranks.min()), int(basis_ranks.max())+1):
      if len(blist_rank[rank]) == 0: continue
      img_base64 = plot_rank_n(rank, blist_rank, rank_indices, 'Gamma', gamma_array.T, 1e-8, 'min' )


      #img_base64 = plot_rank_n(rank, blist_rank, rank_indices, 'Log10(Lambda)', lambda_array.T, threshold, 'max' )
      notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f'<div align="center"><img src="data:image/svg+xml;base64,{img_base64}"></div>']
      })




def _format_table_float(val):
  if pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
    return "-"
  if abs(val) < 1e-6 and val != 0:
    return f"{val:.2e}"
  return f"{val:.4f}"

def _build_gamma_env_table_html(gamma_final, blist, basis_ranks):
  """Booktabs-style table of sum(|gamma|) per atomic environment, grouped by body order."""
  env_n = defaultdict(int)
  env_sums = defaultdict(float)
  body_n = defaultdict(int)
  body_subtotal = defaultdict(float)

  for i, (rank, feat) in enumerate(zip(basis_ranks, blist)):
    rank = int(rank)
    if rank < 1: continue
    tokens = str(feat).split()
    if len(tokens) < rank + 1: continue
    env = " ".join(tokens[:rank + 1])
    body = rank + 1
    val = abs(float(gamma_final[i]))
    env_n[(body, env)] += 1
    env_sums[(body, env)] += val
    body_n[body] += 1
    body_subtotal[body] += val

  if not env_sums: return None

  grand_total = sum(body_subtotal.values())
  grand_n = sum(body_n.values())
  grand_avg = grand_total / grand_n if grand_n > 0 else 0.0
  body_orders = sorted(body_subtotal.keys())

  html = '<table style="border-collapse: collapse; table-layout: auto; width: 80%; font-size: 14px;">\n'
  fmt = "padding: 6px 10px; white-space: nowrap;"
  fmt_left = fmt + " text-align: left;"
  fmt_center = fmt + " text-align: center;"
  fmt_right = fmt + " text-align: right;"
  fmt_row = "padding: 1px 1.6px; font-family: monospace; white-space: nowrap; "
  fmt_left_row = fmt_row + "text-align: left; overflow: hidden; text-overflow: ellipsis;"
  fmt_center_row = fmt_row + "text-align: center;"
  fmt_right_row = fmt_row + "text-align: right;"

  html += '  <thead>\n'
  html += '    <tr style="border-top: 2px solid black; border-bottom: 2px solid black;">\n'
  html += f'      <th style="{fmt_left}"></th>\n'
  html += f'      <th style="{fmt_center}">n</th>\n'
  html += f'      <th style="{fmt_right}">&sum;&gamma; (&darr;)</th>\n'
  html += f'      <th style="{fmt_right}">&lt;&gamma;&gt;</th>\n'
  html += '    </tr>\n'
  html += '  </thead>\n'
  html += '  <tbody>\n'

  html += '    <tr style="font-weight: bold;">\n'
  html += (
    f'      <td style="{fmt_left_row}">'
    f'&sum;&gamma;<sub>all</sub> = {_format_table_float(grand_total)}</td>\n'
  )
  html += f'      <td style="{fmt_center_row}"><strong>{grand_n}</strong></td>\n'
  html += f'      <td style="{fmt_right_row}"><strong>{_format_table_float(grand_total)}</strong></td>\n'
  html += f'      <td style="{fmt_right_row}"><strong>{_format_table_float(grand_avg)}</strong>&nbsp;</td>\n'
  html += '    </tr>\n'

  for body in body_orders:
    n = body_n[body]
    subtotal = body_subtotal[body]
    mean = subtotal / n if n > 0 else 0.0

    html += '    <tr style="font-weight: bold;">\n'
    html += f'      <td style="{fmt_left_row}">&nbsp;&nbsp;{body} body</td>\n'
    html += f'      <td style="{fmt_center_row}"><strong>{n}</strong></td>\n'
    html += f'      <td style="{fmt_right_row}"><strong>{_format_table_float(subtotal)}</strong></td>\n'
    html += f'      <td style="{fmt_right_row}"><strong>{_format_table_float(mean)}</strong>&nbsp;</td>\n'
    html += '    </tr>\n'

    envs = [(env, val) for (b, env), val in env_sums.items() if b == body]
    envs.sort(
      key=lambda x: x[1] / env_n[(body, x[0])] if env_n[(body, x[0])] > 0 else 0.0,
      reverse=True,
    )
    for env, val in envs:
      e_n = env_n[(body, env)]
      e_mean = val / e_n if e_n > 0 else 0.0
      html += '    <tr>\n'
      html += f'      <td style="{fmt_left_row}">&nbsp;&nbsp;&nbsp;&nbsp;{env}</td>\n'
      html += f'      <td style="{fmt_center_row}">{e_n}</td>\n'
      html += f'      <td style="{fmt_right_row}">{_format_table_float(val)}</td>\n'
      html += f'      <td style="{fmt_right_row}">{_format_table_float(e_mean)}&nbsp;</td>\n'
      html += '    </tr>\n'

  html += '    <tr style="border-bottom: 2px solid black;"><td colspan="4"></td></tr>\n'
  html += '  </tbody>\n'
  html += '</table>\n\n'

  return html

def draw_labels(ax, y_positions, texts, x=-1):
  y_positions = np.array(y_positions)
  for y, txt in zip(y_positions, texts): ax.text(x, y, txt, ha="center", va="center", fontsize=8)


def plot_rank_n(rank, blist_rank, rank_indices, title, history_array, threshold=None, threshold_position='min'):
  
  n_features, n_iterations = history_array.shape

  if (heatmap_rows := len(rank_indices[rank])) == 0: return

  if rank >= 1:
    combined = list(zip(blist_rank[rank], rank_indices[rank]))

    def sort_key(item):
      basis = item[0]
      atoms = basis[0]
      ns = tuple(int(i) for i in basis[1].split(',')) if len(basis) > 1 and basis[1] else ()
      ls = tuple(int(j) for j in basis[2].replace(']', '').split(',')) if len(basis) > 2 and basis[2] else ()
      
      # Swapped hierarchy: Group by atoms -> ls -> ns
      return (atoms, ls, ns)

    combined.sort(key=sort_key)
    sorted_blist = [item[0] for item in combined]
    sorted_indices = [item[1] for item in combined]
  else:
    sorted_blist = blist_rank[rank]
    sorted_indices = rank_indices[rank]

  label_spacing = .01*(6*rank-4)*n_iterations
  if rank == 0:
    xlim, xticks_extra = -1.5, [-.5, -.5]
    figsize = (8, heatmap_rows)
  else:
    xlim, xticks_extra = -4*label_spacing-.5, [-2*label_spacing-.5, -label_spacing-.5]
    figsize = (10, max(6, heatmap_rows*11/72))

  finite = history_array[np.isfinite(history_array)]
  if finite.size == 0: dmin, dmax = 0.0, 1.0
  else: dmin, dmax = float(finite.min()), float(finite.max())

  if threshold is None:
    vmin, vmax = dmin, dmax
    cbar_extend = 'neither'
    cmap = plt.cm.turbo
  elif threshold_position == 'min':
    vmin, vmax = max(threshold, dmin), dmax
    cbar_extend = 'min'
    cmap = plt.cm.turbo
    cmap.set_under('white')
  else:
    vmin, vmax = dmin, min(threshold, dmax)
    cbar_extend = 'max'
    cmap = plt.cm.turbo.reversed()
    cmap.set_over('white')

  if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
    vmin, vmax = dmin, max(dmax, np.nextafter(dmin, np.inf))

  fig, ax = plt.subplots(figsize=figsize, layout="constrained")
  
  im_data = np.ma.masked_invalid(history_array[sorted_indices])
  im = ax.imshow(im_data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

  # --- atom labels ---
  atoms = Counter([basis[0] for basis in sorted_blist])
  y, y_positions, texts = 0, [], []
  for k, v in atoms.items():
    ax.add_patch(Rectangle((xlim, y - 0.5), xticks_extra[0]-xlim, v, fc='w', ec="k", zorder=9))
    ax.text((xlim+xticks_extra[0])/2, y + v / 2 - 0.5, k,
        ha="center", va="center", fontsize=10, fontweight="bold", zorder=10)
    y += v
      
  if rank >= 1:
    # --- ns & ls labels ---
    y = 0
    # Create contiguous blocks based on 'ls' instead of 'ns'
    ls_counts = Counter([f"{basis[0]}_{basis[2]}" for basis in sorted_blist])
    
    # Left column: ns headers
    ns_x = (xticks_extra[0] + xticks_extra[1]) / 2
    ax.text(ns_x, heatmap_rows, 'ns', ha="center", va="top", fontsize=10, fontweight="bold")
    
    # Right column: ls headers
    ls_x = (xticks_extra[1] - 0.5) / 2
    ax.text(ls_x, heatmap_rows, 'ls', ha="center", va="top", fontsize=10, fontweight="bold")

    for k, v in ls_counts.items():
      # 1. Merge 'ls' inside the RIGHT column (from xticks_extra[1] to -0.5)
      ax.add_patch(Rectangle((xticks_extra[1], y - 0.5), -0.5 - xticks_extra[1], v, fc='w', ec="k", zorder=9))
      ax.text(ls_x, y + v / 2 - 0.5, k.split('_')[1].replace(']', ''), ha="center", va="center", fontsize=8, zorder=10)
      
      # 2. Collect 'ns' elements to draw individually in the LEFT column
      for j in range(v):
        y_positions.append(y + j)
        texts.append(sorted_blist[y + j][1])
      y += v

    # 3. Draw individual 'ns' labels in the left column bounds
    draw_labels(ax, y_positions, texts, x=ns_x)

  tick_values = MaxNLocator(steps=[1, 5, 10],integer=True).tick_values(0, n_iterations)
  tick_positions = [t - .5 for t in tick_values]
  ax.set_xticks(tick_positions)
  ax.set_xticklabels([str(int(t)) if t>0 else '' for t in tick_values])
  ax.set_xlim(xlim, n_iterations-.5)
  ax.set_yticks([], minor=False)
  ax.set_yticks(np.arange(heatmap_rows+1)-0.5, minor=True)
  ax.tick_params(axis="y", which='both', length=0, left=False, labelleft=False, right=False, labelright=False)
  ax.grid(axis="x", which="major", color="k", linestyle="-", linewidth=1, antialiased=True)
  ax.grid(axis="y", which="both", color="k", linestyle="-", linewidth=1, antialiased=True, zorder=9)
  plt.setp(ax.get_xticklabels(), fontsize=10)
  fig.suptitle(f"{title} History (rank {rank})", fontsize="x-large", fontweight="bold")

  cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=2*min(.1,1/heatmap_rows), shrink=.8, extend=cbar_extend)
  span = vmax - vmin
  if np.isfinite(span) and span > 0:
    cbar_ticks = [vmin + span * f for f in [0, 0.25, 0.5, 0.75, 1.0]]
  else:
    cbar_ticks = [vmin]
  cbar.set_ticks(cbar_ticks)
  cbar.ax.set_xticklabels([f'{t:.2g}' for t in cbar_ticks])
  title_extra = "" if threshold is None else " (white: removed features)"
  cbar.set_label(title + title_extra, fontsize=8, fontweight='bold')

  if threshold is not None:
    cax_top = cbar.ax.twiny()
    xl0, xl1 = cbar.ax.get_xlim()
    if np.isfinite(xl0) and np.isfinite(xl1):
      cax_top.set_xlim(xl0, xl1)
      cax_top.set_xticks([threshold])
      cax_top.xaxis.set_ticks_position('top')
  
  buf = io.BytesIO()
  plt.savefig(buf, format='svg', bbox_inches='tight')
  plt.close()
  svg_text = buf.getvalue().decode('utf-8')
  buf.close()
  return base64.b64encode(svg_text.encode('utf-8')).decode('utf-8')
