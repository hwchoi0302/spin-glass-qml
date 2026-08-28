#!/usr/bin/env python3
"""Build the spin-glass model and write results/<Lx>x<Ly>/model_config.json.

This is the single place where the couplings J are generated. Every other
stage of the pipeline - the Python trainer, the Julia trainer, the Qiskit
circuit builders - loads the resulting JSON instead of regenerating J on its
own. That matters because the Python and Julia random number generators are
different and because bond index k must always refer to the same physical
bond on both sides.

Usage:
    python scripts/00_build_model.py
    python scripts/00_build_model.py --set model.Lx=10 --set model.Ly=10
    python scripts/00_build_model.py --force        # overwrite existing config
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from config import apply_overrides, load_config, output_dir  # noqa: E402
from hamiltonians import SpinGlass2D, frustration_ratio      # noqa: E402


def build_model(config: dict) -> SpinGlass2D:
    """Instantiate SpinGlass2D from the ``model`` config block."""
    m = config['model']
    return SpinGlass2D(
        Lx=m['Lx'], Ly=m['Ly'], h=m['transverse_field_h'],
        coupling_type=m['coupling_distribution'],
        j_magnitude=m['j_magnitude'], seed=m['seed'],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='section.key=value',
                        help='Override a config value')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite an existing model_config.json')
    args = parser.parse_args()

    config = apply_overrides(load_config(), args.overrides)
    out_dir = output_dir(config)
    path = os.path.join(out_dir, 'model_config.json')

    model = build_model(config)

    if os.path.exists(path) and not args.force:
        with open(path) as f:
            existing = json.load(f)
        same = (existing.get('J') == [float(j) for j in model.J]
                and existing.get('bonds') == [list(b) for b in model.bonds])
        if same:
            print(f"model_config.json already matches this config: {path}")
            return
        raise SystemExit(
            f"Refusing to overwrite {path}: it holds a different model.\n"
            f"Existing results in {out_dir} were produced with those "
            f"couplings. Re-run with --force only if you intend to invalidate "
            f"them."
        )

    payload = model.to_config_dict()
    # Record the settings the rest of the pipeline needs, so a results
    # directory is self-describing.
    payload.update({
        'delta_t': config['target']['delta_t'],
        'dt_fine': config['target']['dt'],
        'trotter_order': config['target']['trotter_order'],
        'cutoff_target': config['target']['cutoff'],
        'n_layers': config['ansatz']['n_layers'],
    })

    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)

    # The Julia pipeline has no YAML dependency, so the merged configuration is
    # also mirrored to JSON next to the model. Both languages then read exactly
    # the same settings.
    run_config_path = os.path.join(out_dir, 'run_config.json')
    with open(run_config_path, 'w') as f:
        json.dump({k: v for k, v in config.items() if k != 'paths'},
                  f, indent=2)

    fr = frustration_ratio(model.J, model.bonds, model.Lx, model.Ly)
    print(f"  lattice     : {model.Lx}x{model.Ly} = {model.num_qubits} qubits")
    print(f"  bonds       : {model.num_bonds} "
          f"({(model.Lx - 1) * model.Ly} horizontal + "
          f"{model.Lx * (model.Ly - 1)} vertical)")
    print(f"  couplings   : {config['model']['coupling_distribution']}, "
          f"seed={model.seed}, sum J = {model.J.sum():+.1f}")
    print(f"  frustration : {fr:.3f} of plaquettes")
    print(f"  substeps    : " + ", ".join(
        f"{s}:{len(v)}" for s, v in model.substep_bonds.items()))
    print(f"  written     : {path}")
    print(f"  written     : {run_config_path}")


if __name__ == '__main__':
    main()
