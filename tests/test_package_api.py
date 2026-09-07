"""Top-level package contract: __all__, lazy subpackage access, light import."""

import subprocess
import sys

import aimct


def test_all_lists_subpackages_and_version():
    assert "__version__" in aimct.__all__
    for name in ("systems", "controllers", "planning", "robust", "estimation",
                 "sysid", "trajectories", "simulate", "benchmarks", "ml", "rl",
                 "viz", "dev", "deploy", "hil"):
        assert name in aimct.__all__
    assert sorted(dir(aimct)) == sorted(aimct.__all__)


def test_lazy_subpackage_attribute_access():
    # available after a bare ``import aimct`` with no explicit submodule import
    assert aimct.controllers.LQR.__name__ == "LQR"
    assert aimct.systems.CartPole.__name__ == "CartPole"
    assert aimct.simulate.simulate_batch.__name__ == "simulate_batch"
    assert aimct.estimation.ParticleFilter.__name__ == "ParticleFilter"


def test_unknown_attribute_raises():
    import pytest

    with pytest.raises(AttributeError):
        aimct.does_not_exist


def test_bare_import_does_not_pull_heavy_optional_subpackages():
    # a fresh interpreter: importing aimct must not import viz (matplotlib) or
    # rl (gymnasium) — those load only on first access.
    code = (
        "import sys, aimct; "
        "assert 'aimct.viz' not in sys.modules, 'viz eagerly imported'; "
        "assert 'aimct.rl' not in sys.modules, 'rl eagerly imported'; "
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
