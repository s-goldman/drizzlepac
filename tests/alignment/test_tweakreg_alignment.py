import os
import shutil
from pathlib import Path

import pytest
from astropy.io import fits

from drizzlepac import tweakreg
from drizzlepac.haputils import astroquery_utils as aqutils


def _retrieve_from_mast(dataset: str, cache_dir: Path) -> Path:
    """Download all visit products once and return the desired FLC file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    desired = cache_dir / f"{dataset.lower()}_flc.fits"
    if desired.exists():
        return desired

    original_cwd = os.getcwd()
    os.chdir(cache_dir)
    try:
        retrieved = aqutils.retrieve_observation(
            "JCDUA3*",
            suffix=["FLC"],
            product_type="pipeline",
            clobber=False,
        )
    except Exception as exc:
        os.chdir(original_cwd)
        pytest.skip(f"Failed to retrieve JCDUA3* from MAST: {exc}")
    finally:
        os.chdir(original_cwd)

    if not retrieved:
        pytest.skip("No products retrieved for JCDUA3*")

    for fname in retrieved:
        candidate = cache_dir / Path(fname).name
        if candidate.exists() and candidate.name.lower() == desired.name:
            return candidate

    if desired.exists():
        return desired

    matches = list(cache_dir.glob(f"{dataset.lower()}*flc.fits"))
    if matches:
        return matches[0]

    pytest.fail(f"Unable to locate downloaded FLC for {dataset}")


@pytest.mark.bigdata
@pytest.mark.remote_data
def test_alignment_shiftfile(tmp_path):
    cache_dir = tmp_path / "mast_cache"
    ref_copy = _retrieve_from_mast("jcdua3f4q", cache_dir)
    target_copy = _retrieve_from_mast("jcdua3f8q", cache_dir)

    # Work in an isolated folder that mirrors tweakreg usage.
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    ref_work = work_dir / ref_copy.name
    target_work = work_dir / target_copy.name
    shutil.copy2(ref_copy, ref_work)
    shutil.copy2(target_copy, target_work)

    out_shifts = work_dir / "tweakreg_alignment_shifts.txt"
    out_wcs = work_dir / "tweakreg_alignment_ref_wcs.fits"
    wcs_name = "TWEAKREG_REGTEST"

    original_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        # Align the visit pair and capture the solved shifts.
        tweakreg.TweakReg(
            f"{ref_work.name},{target_work.name}",
            imagefindcfg=dict(
                threshold=50,
                conv_width=4.5,
                skysigma=20.0,
                dqbits=None,
                computesig=False,
            ),
            refimagefindcfg=dict(
                threshold=50,
                conv_width=4.5,
                skysigma=20.0,
                dqbits=None,
                computesig=False,
            ),
            refimage=ref_work.name,
            updatehdr=True,
            wcsname=wcs_name,
            shiftfile=True,
            outshifts=out_shifts.name,
            outwcs=out_wcs.name,
            clean=False,
            writecat=False,
            interactive=False,
        )
    finally:
        os.chdir(original_cwd)

    assert out_shifts.exists()
    assert out_wcs.exists()

    with out_shifts.open("r", encoding="utf-8") as fh:
        rows = [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]

    assert len(rows) == 2

    try:
        row = next(r for r in rows if r.split()[0].endswith(target_work.name))
    except StopIteration as err:
        pytest.fail(f"Missing alignment entry for {target_work.name}: {rows}")

    tokens = row.split()
    xshift, yshift = map(float, tokens[1:3])
    rotation = float(tokens[3])
    scale = float(tokens[4])
    xrms = float(tokens[5])
    yrms = float(tokens[6])

    # Guard against regressions in the solved affine transform.
    assert xshift == pytest.approx(-3.81885, abs=0.005)
    assert yshift == pytest.approx(-1.21188, abs=0.005)
    assert rotation == pytest.approx(0.018055, abs=5e-4)
    assert scale == pytest.approx(0.999867, abs=5e-5)
    assert max(xrms, yrms) < 0.1

    with fits.open(target_work) as hdul:
        sci1_header = hdul["SCI", 1].header
        # Confirm tweakreg tagged the updated WCS in-place.
        assert sci1_header.get("WCSNAME") == wcs_name
