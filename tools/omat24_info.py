import argparse
from fairchem.core.datasets import AseDBDataset
from collections import defaultdict

def list_atoms_info_keys_with_examples(path: str, max_samples: int = 1000):
    dataset = AseDBDataset(config=dict(src=path))
    key_examples = defaultdict(list)
    seen_keys = set()

    print(f"Scanning up to {max_samples} samples from: {path}")
    for i in range(max_samples):
        #atoms = data['atoms']
        atoms = dataset.get_atoms(i)
        if not hasattr(atoms, 'info'):
            continue
        for k, v in atoms.info.items():
            seen_keys.add(k)
            if len(key_examples[k]) < 10:
                key_examples[k].append(v)
        if i + 1 >= max_samples:
            break

    print("\nUnique atoms.info keys found:")
    for k in sorted(seen_keys):
        print(f"\n{k}:")
        for idx, example in enumerate(key_examples[k], 1):
            # limit output for readability
            s = str(example)
            if len(s) > 100:
                s = s[:97] + "..."
            print(f"  example {idx}: {s}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List unique atoms.info keys and example values from an OMat24 LMDB dataset."
    )
    parser.add_argument("path", type=str, help="Path to LMDB dataset (e.g. train.lmdb)")
    parser.add_argument("--max-samples", type=int, default=1000, help="Number of entries to inspect (default: 1000)")
    args = parser.parse_args()

    list_atoms_info_keys_with_examples(args.path, args.max_samples)
