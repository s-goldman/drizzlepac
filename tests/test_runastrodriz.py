"""Tests for the SKIP_APOSTERIORI_IF_VISIT_LEVEL_ALIGNMENT environment
variable used by runastrodriz to keep visit-level (FIT_SVM) a posteriori
WCS solutions and skip further alignment.
"""
import os

import pytest

from drizzlepac import runastrodriz, util

ENVVAR_NAME = 'SKIP_APOSTERIORI_IF_VISIT_LEVEL_ALIGNMENT'


@pytest.fixture(autouse=True)
def clean_envvar():
    """Keep the process environment free of the switch around every test."""
    os.environ.pop(ENVVAR_NAME, None)
    yield
    os.environ.pop(ENVVAR_NAME, None)


def read_switch():
    """Read the switch exactly as runastrodriz.process() does."""
    return util.get_envvar_switch(
        runastrodriz.envvar_skip_visit_level_aposteriori_alignment_name,
        description="'Skip visit-level a posteriori alignment'",
        default=False,
    )


def test_envvar_constant_matches_documented_name():
    """The module constant must match the documented variable name."""
    assert (runastrodriz.envvar_skip_visit_level_aposteriori_alignment_name
            == ENVVAR_NAME)
    assert ENVVAR_NAME in runastrodriz.__doc__


def test_defaults_to_off():
    """When unset, the switch must default to False (no behavior change)."""
    assert read_switch() is False


@pytest.mark.parametrize("env_value,expected",[("true", True), ("True", True),
        ("TRUE", True), ("yes", True), ("YES", True), ("on", True), ("ON", True), 
        ("false", False), ("False", False), ("FALSE", False), ("no", False),
        ("NO", False), ("off", False), ("OFF", False)])
def test_envvar_values(env_value, expected):
    """The switch must honor all documented on/off values."""
    os.environ[ENVVAR_NAME] = env_value
    assert read_switch() is expected


def test_invalid_value_raises():
    """An unrecognized value must raise a ValueError."""
    os.environ[ENVVAR_NAME] = 'maybe'
    with pytest.raises(ValueError, match=ENVVAR_NAME):
        read_switch()


@pytest.mark.parametrize("skip_switch,wcsnames,expected_skip", [
    # switch on + all visit-level solutions -> skip
    (True, ['IDC_2731450pi-FIT_SVM_GAIADR2', 'IDC_2731450pi-FIT_SVM_GAIAeDR3'], True),
    # switch on but one exposure lacks a visit-level solution -> no skip
    (True, ['IDC_2731450pi-FIT_SVM_GAIADR2', 'IDC_2731450pi-HSC30'], False),
    # other a posteriori or a priori solutions do not count as visit-level
    (True, ['IDC_2731450pi-FIT_REL_GAIAeDR3'], False),
    (True, ['IDC_2731450pi-GSC240'], False),
    # switch off -> never skip, even when all are visit-level
    (False, ['IDC_2731450pi-FIT_SVM_GAIADR2'], False),
])
def test_skip_decision_logic(skip_switch, wcsnames, expected_skip):
    """Alignment is only skipped when the switch is on AND every input
    exposure's primary WCSNAME contains 'FIT_SVM'."""
    svm_wcsname = runastrodriz.all_visit_level_wcsnames(wcsnames)
    assert (skip_switch and svm_wcsname) is expected_skip


def test_apriori_wcs_fallback_selects_newest_non_fit_solution():
    # The FIT_SVM headerlet has the newest IDCTAB but must be ignored;
    # of the two a priori candidates the newer 2024 IDCTAB wins.
    selected = runastrodriz._find_most_recent_apriori_wcs(
        ['IDC_2731450pi-FIT_SVM_GAIAeDR3', 'IDC_0461802ej', 'IDC_2731450pi'],
        ['IDC_2731450pi-FIT_SVM_GAIAeDR3', 'IDC_0461802ej', 'IDC_2731450pi'],
        ['2025-01-01', '2023-01-01', '2024-01-01'],
    )

    assert selected == ('2024-01-01', 'IDC_2731450pi', 'IDC_2731450pi')


def test_apriori_wcs_fallback_returns_none_for_fit_only_solutions():
    # Only FIT_SVM and FIT_REL solutions are present, so no a priori 
    # WCS should be selected.
    selected = runastrodriz._find_most_recent_apriori_wcs(
        ['IDC_2731450pi-FIT_SVM_GAIAeDR3', 'IDC_2731450pi-FIT_REL_GAIAeDR3'],
        ['IDC_2731450pi-FIT_SVM_GAIAeDR3', 'IDC_2731450pi-FIT_REL_GAIAeDR3'],
        ['2025-01-01', '2024-01-01'],
    )

    assert selected is None
