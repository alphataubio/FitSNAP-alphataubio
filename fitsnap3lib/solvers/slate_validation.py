from fitsnap3lib.solvers.slate_common import SlateCommon
import numpy as np
import json

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
                    subsystem_lines = [f"### {row_type}\n\n"]
                    
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
                                
                                # Helper function to format numbers
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
                                    html += f'      <td style="{fmt_n}">{int(row["ncount"])}</td>\n'
                                    html += f'      <td style="{fmt_right}">{format_value(row["mae"])}</td>\n'
                                    html += f'      <td style="{fmt_right}">{format_value(row["rmse"])}</td>\n'
                                    html += f'      <td style="{fmt_right}">{format_value(row["rsq"])}</td>\n'
                                    html += f'      <td style="{fmt_n}">{int(row["ncount_test"])}</td>\n'
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
                    
        self.pt.single_print(f"*** ok 1")

        if self.method == 'ARD':
            self.validation_notebook_ard(notebook)
            
        # Write notebook to file
        with open(notebook_file, 'w') as f:
            json.dump(notebook, f, indent=2)
        
        self.pt.single_print(f"Created validation notebook: {notebook_file}")


    def validation_notebook_ard(self, notebook):
    
        # Cell 4: Load gamma and lambda history from adios2
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Gamma and Lambda Evolution Heatmaps"]
        })
        
        # Cell 5: Create and display gamma heatmaps
        # Load data from adios2 to create plots
        from adios2 import Stream
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import io
        import base64
        
        output_prefix = self.config.sections['OUTFILE'].metrics.replace('.md', '')
        with Stream(f"{output_prefix}.bp", 'r') as adios2_stream:
            basis_ranks = adios2_stream.read_attribute("basis_ranks")
            gamma_history = lambda_history = []
            for _ in adios2_stream.steps():
                gamma_history.append(adios2_stream.read("gamma"))
                lambda_history.append(adios2_stream.read("lambda"))
        
        gamma_array = np.array(gamma_history)
        lambda_array = np.array(lambda_history)
        n_iterations, n_features = gamma_array.shape
        
        # Create the gamma heatmap plot - 4x1 layout with custom colormap
        # TRANSPOSED: iterations on horizontal axis, features on vertical axis
        from matplotlib.colors import LinearSegmentedColormap
        
        # Create custom colormap: white for 0, turbo for >0
        turbo = plt.cm.turbo
        colors = [(1, 1, 1, 1)] + [turbo(i) for i in range(1, turbo.N)]
        custom_cmap = LinearSegmentedColormap.from_list('custom_turbo', colors, N=256)
        
        fig, axes = plt.subplots(5, 1, figsize=(16, 16))
        
        # Create gamma heatmaps for each rank
        for i, rank in enumerate(basis_ranks[:5]):
            ax = axes[i]
            # Filter features by rank
            rank_mask = basis_ranks == rank
            rank_gamma = gamma_array[:, rank_mask]
            
            # TRANSPOSE: rank_gamma.T so iterations are on x-axis, features on y-axis
            im = ax.imshow(rank_gamma.T, aspect='auto', cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=1)
            ax.set_title(f'PACE Rank {rank} Gamma Evolution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Feature Index', fontsize=12)
            
            # Add statistics
            final_gamma = rank_gamma[-1, :]
            n_active_final = np.sum(final_gamma > 0.01)
            ax.text(0.98, 0.98, f'Active: {n_active_final}/{rank_gamma.shape[1]}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add single colorbar at bottom
        fig.subplots_adjust(bottom=0.08)
        cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.015])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Gamma Value (white=0, removed features)', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        
        # Save to buffer and encode as base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64_gamma = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### Gamma Heatmaps"]
        })
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "image/png": img_base64_gamma
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": [
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "from matplotlib.colors import LinearSegmentedColormap\n",
                "\n",
                "# Create custom colormap: white for 0 (removed features), turbo for >0\n",
                "turbo = plt.cm.turbo\n",
                "colors = [(1, 1, 1, 1)] + [turbo(i) for i in range(1, turbo.N)]\n",
                "custom_cmap = LinearSegmentedColormap.from_list('custom_turbo', colors, N=256)\n",
                "\n",
                "# Determine which ranks to plot\n",
                "if feature_ranks is not None:\n",
                "    unique_ranks = sorted(set(feature_ranks))\n",
                "    feature_ranks_array = np.array(feature_ranks)\n",
                "else:\n",
                "    # Fallback: assume 4 ranks with equal division\n",
                "    unique_ranks = [1, 2, 3, 4]\n",
                "    features_per_rank = n_features // 4\n",
                "    feature_ranks_array = np.repeat(unique_ranks, features_per_rank)\n",
                "    if len(feature_ranks_array) < n_features:\n",
                "        feature_ranks_array = np.concatenate([feature_ranks_array,\n",
                "            np.full(n_features - len(feature_ranks_array), unique_ranks[-1])])\n",
                "\n",
                "# Create 4x1 layout for PACE ranks\n",
                "fig, axes = plt.subplots(4, 1, figsize=(16, 16))\n",
                "\n",
                "# Create gamma heatmaps for each rank\n",
                "for i, rank in enumerate(unique_ranks[:4]):\n",
                "    ax = axes[i]\n",
                "    # Filter features by rank\n",
                "    rank_mask = feature_ranks_array == rank\n",
                "    rank_gamma = gamma_array[:, rank_mask]\n",
                "    \n",
                "    # Transpose: iterations on x-axis (horizontal), features on y-axis (vertical)\n",
                "    im = ax.imshow(rank_gamma.T, aspect='auto', cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=1)\n",
                "    ax.set_title(f'PACE Rank {rank} Gamma Evolution', fontsize=14, fontweight='bold')\n",
                "    ax.set_xlabel('Iteration', fontsize=12)\n",
                "    ax.set_ylabel('Feature Index', fontsize=12)\n",
                "    \n",
                "    # Add statistics\n",
                "    final_gamma = rank_gamma[-1, :]\n",
                "    n_active_final = np.sum(final_gamma > 0.01)\n",
                "    ax.text(0.98, 0.98, f'Active: {n_active_final}/{rank_gamma.shape[1]}',\n",
                "            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',\n",
                "            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))\n",
                "\n",
                "# Add single horizontal colorbar at bottom\n",
                "fig.subplots_adjust(bottom=0.08)\n",
                "cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.015])\n",
                "cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')\n",
                "cbar.set_label('Gamma Value (white=0, removed features)', fontsize=12)\n",
                "\n",
                "plt.tight_layout(rect=[0, 0.04, 1, 1])\n",
                "plt.show()"
            ]
        })
        
        # Cell 6: Create and display lambda heatmaps
        # Create the lambda heatmap plot
        fig, axes = plt.subplots(4, 1, figsize=(16, 16))
        
        # Create lambda heatmaps for each rank
        for i, rank in enumerate(unique_ranks[:4]):
            ax = axes[i]
            # Filter features by rank
            rank_mask = feature_ranks_array == rank
            rank_lambda = lambda_array[:, rank_mask]
            
            # Use log scale for lambda visualization
            rank_lambda_log = np.log10(rank_lambda + 1e-10)  # Add small value to avoid log(0)
            
            # TRANSPOSE: rank_lambda_log.T so iterations are on x-axis, features on y-axis
            im = ax.imshow(rank_lambda_log.T, aspect='auto', cmap='viridis', interpolation='nearest')
            ax.set_title(f'PACE Rank {rank} Lambda Evolution (log10 scale)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Feature Index', fontsize=12)
            
            # Add statistics
            final_lambda = rank_lambda[-1, :]
            n_active_final = np.sum(final_lambda < 1e3)  # Count features with small lambda
            ax.text(0.98, 0.98, f'Active: {n_active_final}/{rank_lambda.shape[1]}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add single colorbar at bottom
        fig.subplots_adjust(bottom=0.08)
        cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.015])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Log10(Lambda) - higher values indicate less important features', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        
        # Save to buffer and encode as base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64_lambda = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### Lambda Heatmaps"]
        })
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "image/png": img_base64_lambda
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": [
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "\n",
                "# Create 4x1 layout for PACE ranks\n",
                "fig, axes = plt.subplots(4, 1, figsize=(16, 16))\n",
                "\n",
                "# Create lambda heatmaps for each rank\n",
                "for i, rank in enumerate(unique_ranks[:4]):\n",
                "    ax = axes[i]\n",
                "    # Filter features by rank\n",
                "    rank_mask = feature_ranks_array == rank\n",
                "    rank_lambda = lambda_array[:, rank_mask]\n",
                "    \n",
                "    # Use log scale for lambda visualization\n",
                "    rank_lambda_log = np.log10(rank_lambda + 1e-10)  # Add small value to avoid log(0)\n",
                "    \n",
                "    # Transpose: iterations on x-axis (horizontal), features on y-axis (vertical)\n",
                "    im = ax.imshow(rank_lambda_log.T, aspect='auto', cmap='viridis', interpolation='nearest')\n",
                "    ax.set_title(f'PACE Rank {rank} Lambda Evolution (log10 scale)', fontsize=14, fontweight='bold')\n",
                "    ax.set_xlabel('Iteration', fontsize=12)\n",
                "    ax.set_ylabel('Feature Index', fontsize=12)\n",
                "    \n",
                "    # Add statistics\n",
                "    final_lambda = rank_lambda[-1, :]\n",
                "    n_active_final = np.sum(final_lambda < 1e3)  # Count features with small lambda\n",
                "    ax.text(0.98, 0.98, f'Active: {n_active_final}/{rank_lambda.shape[1]}',\n",
                "            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',\n",
                "            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))\n",
                "\n",
                "# Add single horizontal colorbar at bottom\n",
                "fig.subplots_adjust(bottom=0.08)\n",
                "cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.015])\n",
                "cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')\n",
                "cbar.set_label('Log10(Lambda) - higher values indicate less important features', fontsize=12)\n",
                "\n",
                "plt.tight_layout(rect=[0, 0.04, 1, 1])\n",
                "plt.show()"
            ]
        })
        
        # Cell 7: Summary statistics with side-by-side gamma and lambda distributions
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Summary Statistics"]
        })
        
        # Create summary plots - (n_iterations) x 2 grid
        # Select representative iterations to show
        if n_iterations <= 10:
            iter_indices = list(range(n_iterations))
        else:
            # Show first, last, and evenly spaced middle iterations
            iter_indices = [0] + list(np.linspace(1, n_iterations-2, min(8, n_iterations-2), dtype=int)) + [n_iterations-1]
        
        n_rows = len(iter_indices)
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4*n_rows))
        
        # If only one row, axes won't be 2D
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for row_idx, iter_idx in enumerate(iter_indices):
            # Left column: Gamma distribution
            ax_gamma = axes[row_idx, 0]
            gamma_at_iter = gamma_array[iter_idx, :]
            gamma_nonzero = gamma_at_iter[gamma_at_iter > 0]
            
            if len(gamma_nonzero) > 0:
                ax_gamma.hist(gamma_nonzero, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
                ax_gamma.set_xlabel('Gamma Value', fontsize=11)
                ax_gamma.set_ylabel('Number of Features', fontsize=11)
                ax_gamma.set_title(f'Iteration {iter_idx}: Gamma Distribution', fontsize=12, fontweight='bold')
                ax_gamma.grid(True, alpha=0.3, axis='y')
                
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
            lambda_log = np.log10(lambda_at_iter + 1e-10)
            
            ax_lambda.hist(lambda_log, bins=50, edgecolor='black', alpha=0.7, color='coral')
            ax_lambda.set_xlabel('Log10(Lambda)', fontsize=11)
            ax_lambda.set_ylabel('Number of Features', fontsize=11)
            ax_lambda.set_title(f'Iteration {iter_idx}: Lambda Distribution', fontsize=12, fontweight='bold')
            ax_lambda.grid(True, alpha=0.3, axis='y')
            
            # Add statistics text
            n_small_lambda = np.sum(lambda_at_iter < 1e3)
            stats_text = f'Active (λ<1e3): {n_small_lambda}/{n_features} ({100*n_small_lambda/n_features:.1f}%)\\n'
            stats_text += f'Log range: [{lambda_log.min():.1f}, {lambda_log.max():.1f}]\\n'
            stats_text += f'Log mean: {lambda_log.mean():.1f}'
            ax_lambda.text(0.98, 0.98, stats_text,
                        transform=ax_lambda.transAxes, fontsize=9, verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save to buffer and encode as base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64_summary = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Create text output for statistics
        stats_text = f"Feature Selection Summary:\nTotal features: {n_features}\n"
        for iteration in [0, n_iterations//2, n_iterations-1]:
            gamma_at_iter = gamma_array[iteration, :]
            lambda_at_iter = lambda_array[iteration, :]
            n_active = np.sum(gamma_at_iter > 0.01)
            stats_text += f"\nIteration {iteration}:\n"
            stats_text += f"  Active features: {n_active}/{n_features} ({100*n_active/n_features:.1f}%)\n"
            stats_text += f"  Gamma range: [{gamma_at_iter.min():.4f}, {gamma_at_iter.max():.4f}]\n"
            stats_text += f"  Gamma mean: {gamma_at_iter[gamma_at_iter > 0].mean():.4f}\n"
            stats_text += f"  Lambda range: [{lambda_at_iter.min():.2e}, {lambda_at_iter.max():.2e}]\n"
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [stats_text]
                },
                {
                    "data": {
                        "image/png": img_base64_summary
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": [
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "\n",
                "# Compute summary statistics\n",
                "print('Feature Selection Summary:')\n",
                "print(f'Total features: {n_features}')\n",
                "\n",
                "for iteration in [0, n_iterations//2, n_iterations-1]:\n",
                "    gamma_at_iter = gamma_array[iteration, :]\n",
                "    lambda_at_iter = lambda_array[iteration, :]\n",
                "    n_active = np.sum(gamma_at_iter > 0.01)\n",
                "    print(f'\\nIteration {iteration}:')\n",
                "    print(f'  Active features: {n_active}/{n_features} ({100*n_active/n_features:.1f}%)') \n",
                "    print(f'  Gamma range: [{gamma_at_iter.min():.4f}, {gamma_at_iter.max():.4f}]')\n",
                "    print(f'  Gamma mean: {gamma_at_iter[gamma_at_iter > 0].mean():.4f}')\n",
                "    print(f'  Lambda range: [{lambda_at_iter.min():.2e}, {lambda_at_iter.max():.2e}]')\n",
                "\n",
                "# Create (n_iterations x 2) subplots for gamma and lambda distributions\n",
                "# Select representative iterations to show\n",
                "if n_iterations <= 10:\n",
                "    iter_indices = list(range(n_iterations))\n",
                "else:\n",
                "    # Show first, last, and evenly spaced middle iterations\n",
                "    iter_indices = [0] + list(np.linspace(1, n_iterations-2, min(8, n_iterations-2), dtype=int)) + [n_iterations-1]\n",
                "\n",
                "n_rows = len(iter_indices)\n",
                "fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4*n_rows))\n",
                "\n",
                "# If only one row, axes won't be 2D\n",
                "if n_rows == 1:\n",
                "    axes = axes.reshape(1, -1)\n",
                "\n",
                "for row_idx, iter_idx in enumerate(iter_indices):\n",
                "    # Left column: Gamma distribution\n",
                "    ax_gamma = axes[row_idx, 0]\n",
                "    gamma_at_iter = gamma_array[iter_idx, :]\n",
                "    gamma_nonzero = gamma_at_iter[gamma_at_iter > 0]\n",
                "    \n",
                "    if len(gamma_nonzero) > 0:\n",
                "        ax_gamma.hist(gamma_nonzero, bins=50, edgecolor='black', alpha=0.7, color='steelblue')\n",
                "        ax_gamma.set_xlabel('Gamma Value', fontsize=11)\n",
                "        ax_gamma.set_ylabel('Number of Features', fontsize=11)\n",
                "        ax_gamma.set_title(f'Iteration {iter_idx}: Gamma Distribution', fontsize=12, fontweight='bold')\n",
                "        ax_gamma.grid(True, alpha=0.3, axis='y')\n",
                "        \n",
                "        # Add statistics text\n",
                "        n_active = np.sum(gamma_at_iter > 0.01)\n",
                "        stats_text = f'Active: {n_active}/{n_features} ({100*n_active/n_features:.1f}%)\\\\n'\n",
                "        stats_text += f'Range: [{gamma_at_iter.min():.3f}, {gamma_at_iter.max():.3f}]\\\\n'\n",
                "        stats_text += f'Mean: {gamma_nonzero.mean():.3f}'\n",
                "        ax_gamma.text(0.98, 0.98, stats_text,\n",
                "                    transform=ax_gamma.transAxes, fontsize=9, verticalalignment='top',\n",
                "                    horizontalalignment='right',\n",
                "                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))\n",
                "    else:\n",
                "        ax_gamma.text(0.5, 0.5, 'No active features', transform=ax_gamma.transAxes,\n",
                "                    ha='center', va='center', fontsize=12)\n",
                "        ax_gamma.set_title(f'Iteration {iter_idx}: Gamma Distribution', fontsize=12, fontweight='bold')\n",
                "    \n",
                "    # Right column: Lambda distribution (log scale)\n",
                "    ax_lambda = axes[row_idx, 1]\n",
                "    lambda_at_iter = lambda_array[iter_idx, :]\n",
                "    lambda_log = np.log10(lambda_at_iter + 1e-10)\n",
                "    \n",
                "    ax_lambda.hist(lambda_log, bins=50, edgecolor='black', alpha=0.7, color='coral')\n",
                "    ax_lambda.set_xlabel('Log10(Lambda)', fontsize=11)\n",
                "    ax_lambda.set_ylabel('Number of Features', fontsize=11)\n",
                "    ax_lambda.set_title(f'Iteration {iter_idx}: Lambda Distribution', fontsize=12, fontweight='bold')\n",
                "    ax_lambda.grid(True, alpha=0.3, axis='y')\n",
                "    \n",
                "    # Add statistics text\n",
                "    n_small_lambda = np.sum(lambda_at_iter < 1e3)\n",
                "    stats_text = f'Active (λ<1e3): {n_small_lambda}/{n_features} ({100*n_small_lambda/n_features:.1f}%)\\\\n'\n",
                "    stats_text += f'Log range: [{lambda_log.min():.1f}, {lambda_log.max():.1f}]\\\\n'\n",
                "    stats_text += f'Log mean: {lambda_log.mean():.1f}'\n",
                "    ax_lambda.text(0.98, 0.98, stats_text,\n",
                "                transform=ax_lambda.transAxes, fontsize=9, verticalalignment='top',\n",
                "                horizontalalignment='right',\n",
                "                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        })
