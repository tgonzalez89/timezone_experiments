"""Dashboard controls and their translation into an experiment configuration."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest
from factories import (
    FakeStreamlit,
)

from timezone_experiments.app.dashboard.inputs import dashboard_inputs
from timezone_experiments.app.dashboard.settings import NOT_IN_DASHBOARD, DashboardInputs, build_config
from timezone_experiments.core.config import (
    ConfigError,
    ExperimentConfig,
)


def test_dashboard_countries_text_becomes_the_country_selection() -> None:
    config = ExperimentConfig(countries=("MEX", "USA"))
    inputs = dashboard_inputs(FakeStreamlit(has_upload=False, run=False, rules=None), config)
    assert inputs.countries == "MEX, USA"
    assert build_config(inputs, config) == config
    assert build_config(replace(inputs, countries=" ; "), config).countries is None
    assert build_config(replace(inputs, countries="can;usa"), config).countries == ("CAN", "USA")


def test_dashboard_has_a_control_for_every_configuration_setting() -> None:
    controls = {field.name for field in fields(DashboardInputs)}
    settings = {field.name for field in fields(ExperimentConfig)}
    assert settings >= NOT_IN_DASHBOARD
    assert settings - NOT_IN_DASHBOARD <= controls


def test_dashboard_admin_level_is_the_fallback_for_default_subdivision_rules() -> None:
    config = ExperimentConfig(default_admin_level=2)
    fake = FakeStreamlit(has_upload=False, run=False, rules=None)
    inputs = dashboard_inputs(fake, config)
    result = build_config(inputs, config)
    assert result == config
    assert result.admin_level_for("USA") == 1
    assert result.admin_level_for("FRA") == 0
    assert result.admin_level_for("ARG") == 2
    widget = fake.number_inputs["Default administrative level"]
    assert "default_admin_level" in str(widget["help"])


@pytest.mark.parametrize(
    ("field", "text", "message"),
    [
        ("subdivision_rules", "{", "Expected a JSON object"),
        ("subdivision_rules", "[1]", "subdivision_rules must map country codes"),
        ("subdivision_rules", '{"USA": -1}', "subdivision_rules administrative levels must be non-negative"),
        ("subdivision_rules", '{"../X": 1}', "Country codes must be three"),
        ("merged_subdivision_levels", '{"FRA": 0}', "merged_subdivision_levels administrative levels must be positive"),
        ("utc_offsets", "1, two", "Candidate UTC offsets must be comma-separated numbers"),
        ("countries", "USA, ../x", "three letters"),
    ],
)
def test_dashboard_text_inputs_are_validated(field: str, text: str, message: str) -> None:
    config = ExperimentConfig(workers=1)
    inputs = dashboard_inputs(FakeStreamlit(has_upload=False, run=False, rules=None), config)
    with pytest.raises(ConfigError, match=message):
        build_config(replace(inputs, **{field: text}), config)


def test_dashboard_offsets_accept_commas_semicolons_and_blanks() -> None:
    config = ExperimentConfig(workers=1)
    inputs = dashboard_inputs(FakeStreamlit(has_upload=False, run=False, rules=None), config)
    assert build_config(replace(inputs, utc_offsets="-1; 0.5, ,2"), config).utc_offsets == (-1, 0.5, 2)
