"""Configuration loading for the spin-glass-qml pipeline.

Every runnable script reads its settings from ``configs/*.yaml`` rather than
from module-level constants, so that the Python and Julia halves of the
pipeline can be driven from one place.

Layout::

    configs/model_2d_spinglass.yaml   lattice, couplings, transverse field
    configs/ansatz_ibm.yaml           HVA / target-Trotter / hardware-Trotter
    configs/optimizer.yaml            optimiser stages, truncation schedule

``load_config()`` merges the three into a single nested dict.  Individual
values can be overridden from the command line via ``--set section.key=value``
(see :func:`apply_overrides`).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'configs')

_CONFIG_FILES = (
    'model_2d_spinglass.yaml',
    'ansatz_ibm.yaml',
    'optimizer.yaml',
    'time_sweep.yaml',
)


def load_config(config_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load and merge every YAML file under ``configs/``.

    Args:
        config_dir: Directory holding the YAML files. Defaults to
            ``<project root>/configs``.

    Returns:
        Nested dict with top-level keys ``model``, ``ansatz``, ``target``,
        ``hardware_trotter``, ``optimizer``, ``truncation``, ``paths``.
    """
    config_dir = config_dir or CONFIG_DIR
    merged: Dict[str, Any] = {}

    for name in _CONFIG_FILES:
        path = os.path.join(config_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing config file: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value

    merged.setdefault('paths', {})
    merged['paths'].setdefault('project_root', PROJECT_ROOT)
    return merged


def output_dir(config: Dict[str, Any], create: bool = True) -> str:
    """Resolve the results directory for the configured lattice size."""
    model = config['model']
    template = config['paths'].get('results_dir', 'results/{Lx}x{Ly}')
    rel = template.format(Lx=model['Lx'], Ly=model['Ly'])
    path = rel if os.path.isabs(rel) else os.path.join(PROJECT_ROOT, rel)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _coerce(text: str) -> Any:
    """Parse a command-line override value.

    YAML scalar rules first, then a numeric retry: YAML 1.1 does not recognise
    exponent forms without a decimal point, so ``1e-10`` would otherwise stay a
    string and silently poison a truncation threshold.
    """
    return _numeric(yaml.safe_load(text))


def _numeric(value: Any) -> Any:
    """Recursively retry numeric parsing on strings YAML left alone."""
    if isinstance(value, str):
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
        return value
    if isinstance(value, list):
        return [_numeric(v) for v in value]
    if isinstance(value, dict):
        return {k: _numeric(v) for k, v in value.items()}
    return value


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply ``section.key=value`` overrides in place.

    Example::

        apply_overrides(cfg, ["model.Lx=10", "ansatz.n_layers=5"])
    """
    for item in overrides:
        if '=' not in item:
            raise ValueError(f"Override must be 'section.key=value', got: {item}")
        dotted, raw = item.split('=', 1)
        keys = dotted.split('.')
        node = config
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                raise KeyError(f"Unknown config section: {dotted}")
            node = node[key]
        if keys[-1] not in node:
            raise KeyError(f"Unknown config key: {dotted}")
        node[keys[-1]] = _coerce(raw)
    return config


def describe(config: Dict[str, Any]) -> str:
    """One-block human-readable summary, for run logs."""
    m, a, t, o = (config['model'], config['ansatz'],
                  config['target'], config['optimizer'])
    return (
        f"  model    : {m['Lx']}x{m['Ly']} ({m['Lx'] * m['Ly']}q), "
        f"h={m['transverse_field_h']}, {m['coupling_distribution']}, seed={m['seed']}\n"
        f"  ansatz   : HVA {a['n_layers']} layers "
        f"({a['single_qubit_rotation']} + {a['entanglement_gate']})\n"
        f"  target   : S{t['trotter_order']} Trotter dt={t['dt']}, "
        f"delta_t={t['delta_t']}, cutoff={t['cutoff']}\n"
        f"  optimizer: {o['stage1']['name']}(lr={o['stage1']['learning_rate']}, "
        f"{o['stage1']['epochs']}ep)"
        + (f" -> {o['stage2']['name']}({o['stage2']['max_iter']} it)"
           if o['stage2']['enabled'] else "")
    )
