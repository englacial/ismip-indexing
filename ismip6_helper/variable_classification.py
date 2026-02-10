"""Variable classification for ISMIP6 datasets.

Classifies ISMIP6 variables as state (ST) or flux (FL) based on
their temporal_type in variables.yaml.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Set

import yaml


@lru_cache(maxsize=1)
def _load_variables() -> dict:
    """Load and cache the variables.yaml file."""
    yaml_path = Path(__file__).resolve().parent.parent / "ismip_metadata" / "variables.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data["variables"]


def get_state_variables() -> Set[str]:
    """Return set of state (ST) variable names."""
    variables = _load_variables()
    return {name for name, info in variables.items() if info.get("temporal_type") == "ST"}


def get_flux_variables() -> Set[str]:
    """Return set of flux (FL) variable names."""
    variables = _load_variables()
    return {name for name, info in variables.items() if info.get("temporal_type") == "FL"}


def get_temporal_type(name: str) -> Optional[str]:
    """Return the temporal type ('ST' or 'FL') for a variable, or None if unknown."""
    variables = _load_variables()
    info = variables.get(name)
    if info is None:
        return None
    return info.get("temporal_type")
