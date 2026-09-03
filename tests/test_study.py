"""The 'bring your own problem' entry point: aimct.study / python -m aimct."""

import subprocess
import sys

import numpy as np

from aimct.study import auto_controllers, run_study
from aimct.systems import CartPole, DynamicalSystem, MassSpringDamper


def test_auto_controllers_designs_lqr_and_mpc_for_cartpole():
    ctrls = auto_controllers(CartPole(), Q=np.diag([1.0, 1, 10, 1]),
                             R=np.array([[0.1]]))
    assert "LQR" in ctrls and "LinearMPC" in ctrls
    assert not any(k.startswith("_") for k in ctrls)


def test_run_study_on_mass_spring_damper_regulates():
    res = run_study(MassSpringDamper(), x0=[1.0, 0.0], dt=0.01, t_final=20.0)
    assert set(res.metrics) == {"LQR", "LinearMPC"}
    for name in res.metrics:
        assert res.status[name] == "Stable"
        assert abs(res.trajectories[name].x[-1, 0]) < 1e-2


def test_run_study_writes_artifacts(tmp_path):
    out = tmp_path / "study"
    run_study(CartPole(), x0=[0, 0, 0.15, 0], dt=0.01, t_final=4.0,
              output_index=2, Q=np.diag([10.0, 1, 100, 10]), R=np.array([[0.1]]),
              u_bounds=(-20.0, 20.0), out_dir=out)
    assert (out / "metrics.md").exists()
    assert (out / "metrics.csv").exists()
    assert any(p.suffix == ".png" for p in out.iterdir())


def test_run_study_accepts_a_user_system_subclass():
    class DoubleIntegrator(DynamicalSystem):
        n_states, n_inputs = 2, 1

        def dynamics(self, t, x, u):
            x, u = self._prep(x, u)
            return np.array([x[1], u[0]])

        def linearize(self, x_eq=None, u_eq=None, eps=1e-6):
            return np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]])

    res = run_study(DoubleIntegrator(), x0=[2.0, 0.0], dt=0.02, t_final=12.0,
                    u_bounds=(-1.0, 1.0))
    assert res.status["LQR"] == "Stable"
    assert abs(res.trajectories["LQR"].x[-1, 0]) < 0.05


def test_cli_list_and_compare_run():
    r = subprocess.run([sys.executable, "-m", "aimct", "list"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "cartpole" in r.stdout

    r = subprocess.run(
        [sys.executable, "-m", "aimct", "compare", "--system",
         "mass_spring_damper", "--t-final", "8"],
        capture_output=True, text=True)
    assert r.returncode == 0
    assert "LQR" in r.stdout and "LinearMPC" in r.stdout
