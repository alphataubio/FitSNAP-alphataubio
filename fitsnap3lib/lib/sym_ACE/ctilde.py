import numpy as np
import collections
import itertools
import re
from copy import deepcopy

from fitsnap3lib.lib.sym_ACE.wigner_rpi import WignerRPI

# Regex for parsing element strings (e.g. "InIn" -> ["In", "In"])
element_patt = re.compile("([A-Z][a-z]?) ?")

ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
    'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
    'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
    'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100
}

def sort_by_atomic_number(elements_list):
    """Sorts a list of element symbols by atomic number."""
    return sorted(elements_list, key=lambda x: ATOMIC_NUMBERS.get(x, 999))

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

class CTildeBasisFunction:
    def __init__(self, mu0=0, rank=0, ndensity=0, num_ms_combs=0):
        self.mu0 = int(mu0)
        self.rank = int(rank)
        self.ndensity = int(ndensity)
        self.num_ms_combs = int(num_ms_combs)
        
        self.mus = np.zeros(rank, dtype=int)
        self.ns = np.zeros(rank, dtype=int)
        self.ls = np.zeros(rank, dtype=int)
        self.ms_combs = np.zeros(num_ms_combs * rank, dtype=int)
        self.ctildes = np.zeros(num_ms_combs * ndensity, dtype=float)

    def to_dict(self):
        return {
            "mu0": int(self.mu0),
            "rank": int(self.rank),
            "ndensity": int(self.ndensity),
            "num_ms_combs": int(self.num_ms_combs),
            "mus": self.mus.tolist(),
            "ns": self.ns.tolist(),
            "ls": self.ls.tolist(),
            "ms_combs": self.ms_combs.tolist(),
            "ctildes": self.ctildes.tolist()
        }

class EmbeddingSpecification:
    def __init__(self):
        self.ndensity = 0
        self.npoti = "FinnisSinclair"
        self.fs_parameters = []
        self.rho_core_cutoff = 0.0
        self.drho_core_cutoff = 0.0

    def to_dict(self):
        return {
            "ndensity": int(self.ndensity),
            "FS_parameters": self.fs_parameters,
            "npoti": self.npoti,
            "rho_core_cutoff": float(self.rho_core_cutoff),
            "drho_core_cutoff": float(self.drho_core_cutoff)
        }

class BondSpecification:
    def __init__(self):
        self.nradmax = 0
        self.lmax = 0
        self.nradbasemax = 0
        self.radbasename = "ACE.jl.base" 
        self.radparameters = []
        self.radcoefficients = [] 
        self.prehc = 0.0
        self.lambdahc = 0.0
        self.rcut = 0.0
        self.dcut = 0.0
        self.rcut_in = 0.0
        self.dcut_in = 0.0
        self.inner_cutoff_type = "distance"

    def to_dict(self):
        return {
            "nradmax": int(self.nradmax),
            "lmax": int(self.lmax),
            "nradbasemax": int(self.nradbasemax),
            "radbasename": self.radbasename,
            "radparameters": self.radparameters,
            "radcoefficients": self.radcoefficients,
            "prehc": float(self.prehc),
            "lambdahc": float(self.lambdahc),
            "rcut": float(self.rcut),
            "dcut": float(self.dcut),
            "rcut_in": float(self.rcut_in),
            "dcut_in": float(self.dcut_in),
            "inner_cutoff_type": self.inner_cutoff_type
        }

class CTildeBasisSet:
    def __init__(self):
        self.nelements = 0
        self.elements = []
        self.E0vals = []
        self.deltaSplineBins = 0.001
        self.embeddings = {} 
        self.bonds = {}      
        self.functions = collections.defaultdict(list) 
        self.bzeroflag = False

    @property
    def ncoeff(self):
        """Total number of basis functions across all species."""
        return sum(len(funcs) for funcs in self.functions.values())

    @property
    def number_functions_by_rank(self):
        """Dictionary mapping rank to the number of functions of that rank."""
        counts = collections.defaultdict(int)
        for funcs in self.functions.values():
            for f in funcs:
                counts[f.rank] += 1
        return counts

    @property
    def blist(self):
        """List of string descriptors for the basis functions."""
        blist = []
        if not self.bzeroflag:
            blist = [f"{t} [0]" for t in self.elements]
        
        # Iterate in standard element order (0, 1, 2...)
        for mu in sorted(self.functions.keys()):
            current_funcs = self.functions[mu]
            
            # Sort: Rank 1 first, then others (matching save_yaml order)
            rank1 = [f for f in current_funcs if f.rank == 1]
            others = [f for f in current_funcs if f.rank != 1]
            sorted_funcs = rank1 + others
            
            for func in sorted_funcs:
                central_el = self.elements[func.mu0]
                neighbor_els = " ".join([self.elements[m] for m in func.mus])
                blist.append(f"{central_el} {neighbor_els} ns {func.ns.tolist()} ls {func.ls.tolist()}")
                
        return blist

    @property
    def basis_ranks(self):
        """List of ranks for all basis functions."""
        ranks = []
        if not self.bzeroflag:
            # Rank 0 corresponds to atomic energy/background
            ranks.extend([0] * self.nelements)
        
        # Iterate in standard element order (0, 1, 2...)
        for mu in sorted(self.functions.keys()):
            current_funcs = self.functions[mu]
            
            # Sort: Rank 1 first, then others (matching blist order)
            rank1 = [f for f in current_funcs if f.rank == 1]
            others = [f for f in current_funcs if f.rank != 1]
            sorted_funcs = rank1 + others
            
            for func in sorted_funcs:
                ranks.append(int(func.rank))
        return ranks

    def save_yaml(self, filename):
        def format_flow_list(lst):
            items = []
            for x in lst:
                if isinstance(x, float): items.append(f"{x:.16g}")
                elif isinstance(x, (int, str, np.integer, np.floating)): items.append(str(x))
                elif isinstance(x, list): items.append(format_flow_list(x))
            return "[" + ", ".join(items) + "]"

        def write_indent(f, indent, key, value, flow=False):
            prefix = "  " * indent
            if flow and isinstance(value, (list, np.ndarray)):
                val_list = value.tolist() if isinstance(value, np.ndarray) else value
                f.write(f"{prefix}{key}: {format_flow_list(val_list)}\n")
            elif isinstance(value, dict):
                f.write(f"{prefix}{key}:\n")
            else:
                f.write(f"{prefix}{key}: {value}\n")

        with open(filename, 'w') as f:
            write_indent(f, 0, "elements", self.elements, flow=True)
            write_indent(f, 0, "E0", self.E0vals, flow=True)
            write_indent(f, 0, "deltaSplineBins", self.deltaSplineBins)
            
            f.write("embeddings:\n")
            for idx, spec in sorted(self.embeddings.items()):
                f.write(f"  {idx}:\n")
                d = spec.to_dict()
                write_indent(f, 2, "ndensity", d["ndensity"])
                write_indent(f, 2, "FS_parameters", d["FS_parameters"], flow=True)
                write_indent(f, 2, "npoti", d["npoti"])
                write_indent(f, 2, "rho_core_cutoff", d["rho_core_cutoff"])
                write_indent(f, 2, "drho_core_cutoff", d["drho_core_cutoff"])

            f.write("bonds:\n")
            sorted_bonds = sorted(self.bonds.items(), key=lambda x: (x[0][0], x[0][1]))
            for (i, j), spec in sorted_bonds:
                f.write(f"  [{i}, {j}]:\n")
                d = spec.to_dict()
                for k in ["nradmax", "lmax", "nradbasemax", "radbasename"]:
                    write_indent(f, 2, k, d[k])
                write_indent(f, 2, "radparameters", d["radparameters"], flow=True)
                write_indent(f, 2, "radcoefficients", d["radcoefficients"], flow=True)
                for k in ["prehc", "lambdahc", "rcut", "dcut", "rcut_in", "dcut_in", "inner_cutoff_type"]:
                    write_indent(f, 2, k, d[k])

            f.write("functions:\n")
            for mu in sorted(self.functions.keys()):
                f.write(f"  {mu}:\n")
                current_funcs = self.functions[mu]
                rank1 = [f for f in current_funcs if f.rank == 1]
                others = [f for f in current_funcs if f.rank != 1]
                sorted_funcs = rank1 + others
                
                for func in sorted_funcs:
                    f.write("    - rank: {}\n".format(func.rank))
                    d = func.to_dict()
                    f.write(f"      mu0: {d['mu0']}\n")
                    f.write(f"      ndensity: {d['ndensity']}\n")
                    f.write(f"      num_ms_combs: {d['num_ms_combs']}\n")
                    write_indent(f, 3, "mus", d["mus"], flow=True)
                    write_indent(f, 3, "ns", d["ns"], flow=True)
                    write_indent(f, 3, "ls", d["ls"], flow=True)
                    write_indent(f, 3, "ms_combs", d["ms_combs"], flow=True)
                    write_indent(f, 3, "ctildes", d["ctildes"], flow=True)

    def trim_basis_by_mask(self, mask):
        total_funcs = sum(len(funcs) for funcs in self.functions.values())
        if len(mask) != total_funcs:
            raise ValueError(f"Mask size ({len(mask)}) != total functions ({total_funcs})")

        mask_idx = 0
        for mu in sorted(self.functions.keys()):
             current_funcs = self.functions[mu]
             rank1_funcs = [f for f in current_funcs if f.rank == 1]
             other_funcs = [f for f in current_funcs if f.rank != 1]
             ordered_funcs = rank1_funcs + other_funcs
             
             kept_funcs = []
             for func in ordered_funcs:
                 if mask[mask_idx]:
                     kept_funcs.append(func)
                 mask_idx += 1
             self.functions[mu] = kept_funcs

# -----------------------------------------------------------------------------
# Configuration Processing & Logic Ported from multispecies_basisextension.py
# -----------------------------------------------------------------------------

def generate_species_keys(elements, r):
    keys = set()
    for el in elements:
        for comb in itertools.combinations_with_replacement(elements, r):
            keys.add((el,) + comb)
    return sorted(keys)

def species_key_to_bonds(key):
    if len(key) == 1:
        bonds = [(key[0], key[0])]
    else:
        k0 = key[0]
        rkeys = key[1:]
        bonds = []
        for rk in rkeys:
            bonds.append((k0, rk))
            bonds.append((rk, k0))
    return bonds

def generate_functions_ext(potential_config):
    elements = sort_by_atomic_number(potential_config["elements"])
    raw_functions = potential_config.get("functions", {})
    functions_ext = collections.defaultdict(dict)

    if 'ALL' in raw_functions:
        all_spec = raw_functions['ALL']
        max_rank = len(all_spec.get('nmax_by_rank', []))
        
        if max_rank == 0 and 'nmax' in all_spec:
             max_rank = 1 
        
        for rank in range(0, max_rank + 1):
            for species in generate_species_keys(elements, rank):
                if rank == 0:
                    functions_ext[species].update({})
                else:
                    #functions_ext[species] = deepcopy(all_spec)
                    
                    functions_ext[species]['rank'] = rank
                    
                    nmax_by_rank = all_spec.get('nmax_by_rank', [1]*rank)
                    lmin_by_rank = all_spec.get('lmin_by_rank', [0]*rank)
                    lmax_by_rank = all_spec.get('lmax_by_rank', [0]*rank)
                    
                    #functions_ext[species]['nmax_by_rank'] = nmax_by_rank
                    #functions_ext[species]['lmin_by_rank'] = lmin_by_rank
                    #functions_ext[species]['lmax_by_rank'] = lmax_by_rank
                    
                    idx = rank - 1
                    if idx < len(nmax_by_rank):
                        functions_ext[species]['nmax'] = nmax_by_rank[idx]
                    if idx < len(lmin_by_rank):
                        functions_ext[species]['lmin'] = lmin_by_rank[idx]
                    if idx < len(lmax_by_rank):
                        functions_ext[species]['lmax'] = lmax_by_rank[idx]

    for k, v in raw_functions.items():
        if k != 'ALL':
            if isinstance(k, str):
                key = tuple(element_patt.findall(k))
            else:
                key = tuple(k)
            
            if key not in functions_ext:
                functions_ext[key] = {}
            functions_ext[key].update(v)
            if 'lmin' not in functions_ext[key]:
                functions_ext[key]['lmin'] = 0

    return max_rank, {k: v for k, v in functions_ext.items() if len(v) > 0}

def generate_bonds_ext(potential_config):
    elements = sort_by_atomic_number(potential_config["elements"])
    raw_bonds = potential_config.get("bonds", {})
    bonds_ext = {pair: {} for pair in itertools.product(elements, repeat=2)}
    
    if 'ALL' in raw_bonds:
        for pair in bonds_ext:
            bonds_ext[pair].update(raw_bonds['ALL'])
            
    for pair, val in raw_bonds.items():
        if pair != 'ALL':
            if isinstance(pair, str):
                dpair = tuple(element_patt.findall(pair))
                if len(dpair) == 1: dpair = (dpair[0], dpair[0])
            else:
                dpair = tuple(pair)
            
            if dpair in bonds_ext:
                bonds_ext[dpair].update(val)
            else:
                bonds_ext[dpair] = val
                
            r_pair = tuple(reversed(dpair))
            if r_pair in bonds_ext:
                bonds_ext[r_pair].update(val)
            else:
                bonds_ext[r_pair] = val

    return {k: v for k, v in bonds_ext.items() if len(v) > 0}

def update_bonds_ext(bonds_ext, functions_ext):
    if not bonds_ext: return bonds_ext
    bonds_ext_updated = deepcopy(bonds_ext)
    
    for key, funcs_spec in functions_ext.items():
        rank = len(key) - 1
        if rank < 1: continue 
        
        nradmax = 0
        nradbasemax = 0
        lmax = funcs_spec.get('lmax', 0)
        
        if 'nmax_by_rank' in funcs_spec:
            nradbasemax = max(funcs_spec['nmax_by_rank'][:1]) if funcs_spec['nmax_by_rank'] else 0
            if len(funcs_spec['nmax_by_rank']) > 1:
                nradmax = max(funcs_spec['nmax_by_rank'][1:])
        else:
            val = funcs_spec.get('nmax', 1)
            if rank == 1: nradbasemax = val
            else: nradmax = val

        if 'lmax_by_rank' in funcs_spec:
            lmax = max([lmax] + funcs_spec['lmax_by_rank'])

        for bkey in species_key_to_bonds(key):
            if bkey not in bonds_ext_updated: continue
            bond = bonds_ext_updated[bkey]
            
            if 'nradbase' not in bond or bond['nradbase'] < nradbasemax:
                bond['nradbase'] = nradbasemax
            if 'nradmax' not in bond or bond['nradmax'] < nradmax:
                bond['nradmax'] = nradmax
            if 'lmax' not in bond or bond['lmax'] < lmax:
                bond['lmax'] = lmax

    return bonds_ext_updated

def generate_embeddings_ext(potential_config):
    elements = sort_by_atomic_number(potential_config["elements"])
    raw_embs = potential_config.get("embeddings", {})
    embeddings_ext = {(el,): {} for el in elements}
    
    if 'ALL' in raw_embs:
        for el in elements:
            embeddings_ext[(el,)].update(raw_embs['ALL'])
            
    for el, val in raw_embs.items():
        if el in elements:
            embeddings_ext[(el,)].update(val)
        elif el != 'ALL':
            raise ValueError(f"Embedding element {el} not found in elements list {elements}")
            
    return {k[0]: v for k, v in embeddings_ext.items() if len(v) > 0}

def parse_full_potential_config(potential_config):
    embs_expanded = generate_embeddings_ext(potential_config)
    bonds_expanded = generate_bonds_ext(potential_config)
    max_rank, funcs_expanded = generate_functions_ext(potential_config)
    bonds_expanded = update_bonds_ext(bonds_expanded, funcs_expanded)
    return max_rank, funcs_expanded, bonds_expanded, embs_expanded

def create_ctilde_basis(potential_config):
    """
    Main entry point. Creates a CTildeBasisSet directly from configuration 
    using Wigner PA-RPI construction.
    """
    
    cbasis = CTildeBasisSet()
    cbasis.elements = sort_by_atomic_number(potential_config['elements'])
    cbasis.nelements = len(cbasis.elements)
    cbasis.E0vals = potential_config.get('erefs', [0.0]*cbasis.nelements) 
    cbasis.deltaSplineBins = potential_config.get('deltaSplineBins', 0.001)
    
    # Read bzeroflag from config
    cbasis.bzeroflag = bool(int(potential_config.get('bzeroflag', 0)))
    
    elem_map = {e: i for i, e in enumerate(cbasis.elements)}
    
    max_rank, funcs_spec, bonds_spec, embs_spec = parse_full_potential_config(potential_config)

    for el, spec in embs_spec.items():
        idx = elem_map[el]
        es = EmbeddingSpecification()
        es.ndensity = spec.get('ndensity', 1)
        es.npoti = spec.get('npoti', 'FinnisSinclair')
        es.fs_parameters = spec.get('fs_parameters', [])
        es.rho_core_cutoff = spec.get('rho_core_cut', 100000.0)
        es.drho_core_cutoff = spec.get('drho_core_cut', 250.0)
        cbasis.embeddings[idx] = es

    for pair, spec in bonds_spec.items():
        try:
            i, j = elem_map[pair[0]], elem_map[pair[1]]
        except KeyError as e:
            print(f"Warning: Could not map element in bond pair {pair}. Available elements: {cbasis.elements}")
            raise e
            
        bs = BondSpecification()
        bs.radbasename = spec.get('radbase', 'ChebExpCos')
        bs.radparameters = spec.get('radparameters', [])
        bs.rcut = spec.get('rcut', 5.0)
        bs.dcut = spec.get('dcut', 0.0)
        bs.rcut_in = spec.get('rcut_in', 0.0)
        bs.dcut_in = spec.get('dcut_in', 0.0)
        bs.inner_cutoff_type = spec.get('inner_cutoff_type', 'distance')
        
        nradmax = spec.get('nradmax', 1)
        lmax = spec.get('lmax', 0)
        nradbase = spec.get('nradbase', nradmax)
        
        bs.nradmax = nradmax
        bs.lmax = lmax
        bs.nradbasemax = nradbase
        
        if 'radcoefficients' in spec:
            bs.radcoefficients = spec['radcoefficients']
        else:
            crad = np.zeros((nradmax, lmax + 1, nradbase))
            for n in range(min(nradmax, nradbase)):
                crad[n, :, n] = 1.0
            bs.radcoefficients = crad.tolist()
            
        bs.prehc = spec.get('core-repulsion', [0,0])[0]
        bs.lambdahc = spec.get('core-repulsion', [0,0])[1]
        
        cbasis.bonds[(i, j)] = bs
    
    lmax_by_rank = [0]*max_rank
    for f in funcs_spec.values():
        rank = f['rank']
        lmax_by_rank[rank-1] = max(lmax_by_rank[rank-1], f['lmax'])
    
    wigner_rpi = WignerRPI(lmax_by_rank=lmax_by_rank)

    for species_tuple, spec in funcs_spec.items():
        rank = len(species_tuple) - 1
        if rank < 1: continue 
        
        try:
            mu_leaves = [elem_map[s] for s in species_tuple[1:]] 
            mu0 = elem_map[species_tuple[0]]
        except KeyError:
            print(f"Warning: Skipping unknown species tuple {species_tuple}")
            continue
        
        nmax = spec.get('nmax', 1)
        lmax = spec.get('lmax', 0)
        lmin = spec.get('lmin', 0)
        
        ndensity = cbasis.embeddings[mu0].ndensity

        ns_range = range(1, nmax + 1)
        ls_range = range(0, lmax + 1)
        
        processed_shells = set()
        
        neighbor_mu = [elem_map[s] for s in species_tuple[1:]]
        
        slot_options = []
        for _ in neighbor_mu:
            slot_options.append(list(itertools.product(ns_range, ls_range)))
        
        for config in itertools.product(*slot_options):
            current_ns = [c[0] for c in config]
            current_ls = [c[1] for c in config]
            if sum(current_ls) % 2 != 0: continue
            if any(l < lmin or l>lmax for l in current_ls): continue
            shell_key = tuple(sorted(zip(neighbor_mu, current_ns, current_ls)))
            if shell_key in processed_shells: continue
            processed_shells.add(shell_key)
            
            can_mu = [x[0] for x in shell_key]
            can_n = [x[1] for x in shell_key]
            can_l = [x[2] for x in shell_key]
            
            # CALL TO WIGNER RPI: No explicit cache argument
            rpi_funcs = wigner_rpi.get_rpi_basis_vectors(can_mu, can_n, can_l)
            
            for rpi_f in rpi_funcs:
                new_func = CTildeBasisFunction(
                    mu0=mu0,
                    rank=rank,
                    ndensity=ndensity,
                    num_ms_combs=len(rpi_f['m_configs'])
                )
                
                new_func.mus = np.array(rpi_f['mus'], dtype=int)
                new_func.ns = np.array(rpi_f['ns'], dtype=int)
                new_func.ls = np.array(rpi_f['ls'], dtype=int)
                
                flat_ms = []
                for m_tup in rpi_f['m_configs']: flat_ms.extend(m_tup)
                new_func.ms_combs = np.array(flat_ms, dtype=int)
                
                vec = rpi_f['vector']
                ctildes_matrix = np.outer(vec, np.ones(ndensity))
                new_func.ctildes = ctildes_matrix.flatten()
                
                cbasis.functions[mu0].append(new_func)
               

    #print(f">>> Basis creation complete. Total functions: {sum(len(f) for f in cbasis.functions.values())}")
    return cbasis
