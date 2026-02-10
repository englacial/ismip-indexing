"""Unit tests for ismip6_helper.variable_classification."""

import yaml
from pathlib import Path

from ismip6_helper.variable_classification import (
    get_state_variables,
    get_flux_variables,
    get_temporal_type,
    _load_variables,
)


class TestVariableSets:
    def test_state_variables_non_empty(self):
        assert len(get_state_variables()) > 0

    def test_flux_variables_non_empty(self):
        assert len(get_flux_variables()) > 0

    def test_sets_are_disjoint(self):
        state = get_state_variables()
        flux = get_flux_variables()
        assert state.isdisjoint(flux), f"Overlap: {state & flux}"

    def test_sets_cover_all_yaml_variables(self):
        """State + flux should cover every variable in variables.yaml."""
        all_vars = set(_load_variables().keys())
        state = get_state_variables()
        flux = get_flux_variables()
        assert state | flux == all_vars

    def test_known_state_variables(self):
        state = get_state_variables()
        for var in ["lithk", "orog", "xvelsurf", "sftgif", "lim"]:
            assert var in state, f"{var} should be ST"

    def test_known_flux_variables(self):
        flux = get_flux_variables()
        for var in ["acabf", "dlithkdt", "hfgeoubed", "tendacabf"]:
            assert var in flux, f"{var} should be FL"


class TestGetTemporalType:
    def test_state_variable(self):
        assert get_temporal_type("lithk") == "ST"

    def test_flux_variable(self):
        assert get_temporal_type("acabf") == "FL"

    def test_unknown_variable(self):
        assert get_temporal_type("nonexistent_var") is None

    def test_all_variables_have_type(self):
        """Every variable in YAML should return ST or FL."""
        for name in _load_variables():
            result = get_temporal_type(name)
            assert result in ("ST", "FL"), f"{name} has unexpected type: {result}"
