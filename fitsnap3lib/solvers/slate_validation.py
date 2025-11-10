from fitsnap3lib.solvers.slate_common import SlateCommon

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from collections import Counter, defaultdict
import re, json, io, base64, adios2

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

class SlateValidation(SlateCommon):

    # --------------------------------------------------------------------------------------------

    def validation_notebook(self):
    
        if self.pt._rank != 0 or not self.config.sections["OUTFILE"].validation:
            return
    
        import pandas as pd
        import os
        
        output_prefix = self.config.sections['OUTFILE'].metrics.replace('.md', '')
        notebook_file = f"{output_prefix}.ipynb"
                
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
        
        # Cell 1: Title
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# SLATE {self.method} Validation Report\n{output_prefix}\n", "\n"]
        })
        
        # Cell 2: Config dictionary (collapsed by default)
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {
              "collapsed": True,
              "jp-MarkdownHeadingCollapsed": True,
              "jupyter": {"outputs_hidden": True}
            },
            "source": ["## Configuration"]
        })
        
        # Extract relevant config info
        config_dict = {}
        for section_name, section in self.config.sections.items():
            if hasattr(section, '__dict__'):
                config_dict[section_name] = {k: v for k, v in section.__dict__.items() if not k.startswith('_')}

        config_json = json.dumps(config_dict, cls=CompactJSONEncoder, indent=2, default=str)
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "\n",
                "# Configuration from FitSNAP run\n",
                f"config = {config_json}\n",
                "\n",
                "# Display key settings\n",
                "print('ARD Settings:')\n",
                "for key in ['max_iter', 'tol', 'threshold_lambda', 'pruning_method']:\n",
                "    if key in config.get('SLATE', {}): print(f'  {key}: {config[\"SLATE\"][key]}')"
            ]
        })
        
        # Cell 3: Error analysis tables as markdown
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Error Analysis"]
        })
        
        # Generate HTML tables from error_analysis DataFrame
        if hasattr(self, 'errors') and self.errors is not None and not self.errors.empty:
            # Determine which row types exist
            row_types = self.errors.index.get_level_values('Subsystem').unique()
            
            for row_type in row_types:
                if row_type in ['Energy', 'Force', 'Stress']:
                    # Create a markdown cell for this subsystem
                    reference_section = self.config.sections["REFERENCE"]
                    if row_type == 'Energy':
                        row_units = f" ({reference_section.energy_units})"
                    elif row_type == 'Force':
                        row_units = f" ({reference_section.force_units})"
                    elif row_type == 'Stress':
                        row_units = f" ({reference_section.stress_units})"
                    else:
                        row_units = ""
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
                                    if pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
                                        return "-"
                                    else:
                                        return int(val)

                                def format_value(val):
                                    # Handle NaN values
                                    if pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
                                        return "-"
                                    if abs(val) < 1e-6 and val != 0:
                                        # Use exponential notation: 1.23e-07 (8 chars like 0.123456)
                                        return f"{val:.2e}"
                                    else:
                                        return f"{val:.6f}"
                                
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
                                
                                # Bottomrule
                                html += '    <tr style="border-bottom: 2px solid black;">\n'
                                html += '      <td colspan="9"></td>\n'
                                html += '    </tr>\n'
                                
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
                    notebook["cells"].append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": subsystem_lines
                    })

        if self.method == 'ARD':
            self.validation_notebook_ard(notebook)
            
        # Write notebook to file
        with open(notebook_file, 'w') as f:
            json.dump(notebook, f, indent=2)
        
        self.pt.debug_single_print(f"Created validation notebook: {notebook_file}")


    def validation_notebook_ard(self, notebook):
    
        # Cell 4: Load gamma and lambda history from adios2
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Gamma and Lambda Evolution Heatmaps"]
        })
        
        # Cell 5: Create and display gamma heatmaps
        # Load data from adios2 to create plots

        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        output_prefix = self.config.sections['OUTFILE'].metrics.replace('.md', '')
        adios2_path = f"{output_prefix}.bp"
        with adios2.FileReader(adios2_path) as adios2_file:
            basis_ranks = adios2_file.read_attribute("basis_ranks")
            blist = adios2_file.read_attribute("blist")
        with adios2.Stream(adios2_path, "r") as adios2_stream:
            gamma_history, lambda_history = [], []
            for _ in adios2_stream.steps():
                gamma_history.append(adios2_stream.read("gamma"))
                lambda_history.append(adios2_stream.read("lambda"))
            gamma_array = np.array(gamma_history)
            lambda_array = np.log10(np.array(lambda_history)+1e-10)
            
        blist_rank = defaultdict(list)
        rank_indices = defaultdict(list)
        for i, (r, f) in enumerate(zip(basis_ranks, blist)):
            blist_rank[r].append(re.split(r" ls \[|\] ns \[| \[0\]$", f))
            rank_indices[r].append(i)
            
        # Create the gamma heatmap plot - 4x1 layout with custom colormap
        notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": ["## Gamma Heatmaps\n"]})

        threshold = self.threshold_gamma if self.pruning_method.lower() == 'gamma' else None
        for rank in range(int(basis_ranks.min()), int(basis_ranks.max())+1):
            img_base64 = plot_rank_n(rank, blist_rank, rank_indices, 'Gamma', gamma_array.T, threshold, 'min')
            notebook["cells"].append({
                "cell_type": "markdown", "metadata": {}, "source": [
                    f'<div align="center"><img src="data:image/svg+xml;base64,{img_base64}"></div>'
                ]
            })

        # Cell 6: Create and display lambda heatmaps
        # Create the lambda heatmap plot
        
        notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": ["## Lambda Heatmaps\n"]})
        
        threshold = self.threshold_lambda if self.pruning_method.lower() == 'lambda' else None
        for n in range(int(basis_ranks.min()), int(basis_ranks.max())+1):
            img_base64 = plot_rank_n(n, blist_rank, rank_indices, 'Log10(Lambda)', lambda_array.T, threshold, 'max')
            notebook["cells"].append({
                "cell_type": "markdown", "metadata": {}, "source": [
                    f'<div align="center"><img src="data:image/svg+xml;base64,{img_base64}"></div>'
                ]
            })

        # Cell 7: Summary statistics with side-by-side gamma and lambda distributions
        
        n_iterations, n_features = gamma_array.shape
        # Create summary plots - (n_iterations) x 2 grid
        # Select representative iterations to show
        if n_iterations <= 10:
            iter_indices = list(range(n_iterations))
        else:
            # Show first, last, and evenly spaced middle iterations
            iter_indices = [0] + list(np.linspace(1, n_iterations-2, min(8, n_iterations-2), dtype=int)) + [n_iterations-1]
        
        n_rows = len(iter_indices)
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4*n_rows))
        turbo = plt.get_cmap('turbo')
        
        # If only one row, axes won't be 2D
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
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
                for c, p in zip(norm_centers, patches):
                    p.set_facecolor(turbo(c))
                
                # Add statistics text
                n_active = np.sum(gamma_at_iter > 0.01)
                stats_text = f'Active: {n_active}/{n_features} ({100*n_active/n_features:.1f}%)\\n'
                stats_text += f'Range: [{gamma_at_iter.min():.3f}, {gamma_at_iter.max():.3f}]\\n'
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
            
            counts, edges, patches = ax_lambda.hist(lambda_at_iter, bins=20, edgecolor='black', alpha=.99)
            ax_lambda.set_xlabel('Log10(Lambda)', fontsize=11)
            ax_lambda.set_ylabel('Number of Features', fontsize=11)
            ax_lambda.set_title(f'Iteration {iter_idx}: Lambda Distribution', fontsize=12, fontweight='bold')
            ax_lambda.grid(True, alpha=0.3, axis='y')

            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            norm_centers = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
            for c, p in zip(norm_centers, patches):
                p.set_facecolor(turbo(c))

            # Add statistics text
            n_small_lambda = np.sum(lambda_at_iter < 1e3)
            stats_text = f'Active (λ<1e3): {n_small_lambda}/{n_features} ({100*n_small_lambda/n_features:.1f}%)\\n'
            stats_text += f'Log range: [{lambda_at_iter.min():.1f}, {lambda_at_iter.max():.1f}]\\n'
            stats_text += f'Log mean: {lambda_at_iter.mean():.1f}'
            ax_lambda.text(0.98, 0.98, stats_text,
                        transform=ax_lambda.transAxes, fontsize=9, verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        plt.close()
        svg_text = buf.getvalue().decode('utf-8')
        buf.close()
        img_base64_summary = base64.b64encode(svg_text.encode('utf-8')).decode('utf-8')
  
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Summary Statistics\n",
                f'<img src="data:image/svg+xml;base64,{img_base64_summary}" '
                'alt="Gamma Heatmap" style="width:100%;height:auto;">'
            ]
        })



def draw_labels(ax, y_positions, texts, x=-1):
    y_positions = np.array(y_positions)
    for y, txt in zip(y_positions, texts):
        ax.text(x, y, txt, ha="center", va="center", fontsize=8)

def plot_rank_n(rank, blist_rank, rank_indices, title, history_array, threshold=None, threshold_position='min'):
    
    n_features, n_iterations = history_array.shape

    if (heatmap_rows := len(rank_indices[rank])) == 0:
        return
        
    #sorted_rank_indices = sorted(rank_indices[rank])
    sorted_blist = sorted(blist_rank[rank], key=lambda basis: basis[1])

    label_spacing = .025*rank*n_iterations
    if rank == 0:
        xlim, xticks_extra = -1.5, [-.5, -.5]
        figsize = (8, heatmap_rows)
    else:
        xlim, xticks_extra = -4*label_spacing-.5, [-2*label_spacing-.5, -label_spacing-.5]
        figsize = (10, max(6, heatmap_rows*11/72))

    turbo = plt.cm.turbo
    if threshold is None:
        vmin, vmax = np.min(history_array), np.max(history_array)
        cbar_extend = 'neither'
    elif threshold_position == 'min':
        vmin, vmax = max(threshold, np.min(history_array)), np.max(history_array)
        cbar_extend = 'min'
        turbo.set_under('white')
    else:
        vmin, vmax = np.min(history_array), min(threshold, np.max(history_array))
        cbar_extend = 'max'
        turbo.set_over('white')

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    im = ax.imshow(history_array[rank_indices[rank]], aspect="auto", cmap=turbo, vmin=vmin, vmax=vmax)

    # --- atom labels ---
    atoms = Counter([basis[0] for basis in sorted_blist])
    y, y_positions, texts = 0, [], []
    for k, v in atoms.items():
        ax.add_patch(Rectangle((xlim, y - 0.5), xticks_extra[0]-xlim, v, fc='w', ec="k", zorder=9))
        ax.text((xlim+xticks_extra[0])/2, (atom_y := y + v / 2 - 0.5), k,
                ha="center", va="center", fontsize=10, fontweight="bold", zorder=10)
        y += v

    if rank >= 1:
        # --- ls labels ---
        y, ls = 0, Counter([f"{basis[0]}_{basis[1]}" for basis in sorted_blist])
        
        #for b in sorted_blist:
        #    print(f"{b}\n")


        ax.text((ls_x:=(xticks_extra[0]+xticks_extra[1])/2), heatmap_rows, 'ls', ha="center", va="top", fontsize=10, fontweight="bold")
        for k, v in sorted(ls.items()):
            print(f"*** k {k} v {v}")
            ax.add_patch(Rectangle((xticks_extra[0], y - 0.5), xticks_extra[1]-xticks_extra[0], v, fc='w', ec="k", zorder=9))
            ax.text(ls_x, y + v / 2 - 0.5, k.split('_')[-1], ha="center", va="center", fontsize=8, zorder=10)
            for j in range(v):
                y_positions.append(y + j)
                texts.append(sorted_blist[y + j][-1].replace(']', ''))
            y += v

        # --- ns labels ---
        ax.text((ns_x:=(xticks_extra[1]-.5)/2), heatmap_rows, 'ns', ha="center", va="top", fontsize=10, fontweight="bold")
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
    cbar_ticks = [vmin + (vmax - vmin) * f for f in [0, 0.25, 0.5, 0.75, 1.0]]
    cbar.set_ticks(cbar_ticks)
    cbar.ax.set_xticklabels([f'{t:.2g}' for t in cbar_ticks])
    title_extra = "" if threshold is None else " (white: removed features)"
    cbar.set_label(title + title_extra, fontsize=8, fontweight='bold')

    if threshold is not None:
        cax_top = cbar.ax.twiny()
        cax_top.set_xlim(cbar.ax.get_xlim())
        cax_top.set_xticks([threshold])
        cax_top.xaxis.set_ticks_position('top')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    plt.close()
    svg_text = buf.getvalue().decode('utf-8')
    buf.close()
    return base64.b64encode(svg_text.encode('utf-8')).decode('utf-8')
  


