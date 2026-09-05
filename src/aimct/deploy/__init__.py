r"""Export a designed controller to a portable form and run it anywhere.

The hardware-bridge promise is *the simulated control law and the deployed
control law are the same artefact*.  This module serialises a **static /
explicit** controller -- one that needs no online optimisation -- to a plain
:file:`controller.json`, and gives you two ways to consume it:

* :func:`load_controller` -> a pure-numpy :class:`PortableController` with the
  exact ``update(y, dt) / reset()`` protocol the rest of ``aimct`` uses, so it
  drops straight into :func:`aimct.simulate.simulate` or
  :class:`aimct.hil.RealTimeLoop`.
* :func:`emit_c` / :func:`emit_micropython` -> a small self-contained
  reference executor (matvec + saturation, or discrete PID with anti-windup
  and a filtered derivative) with the gains baked in, for the embedded target.

Supported: :class:`~aimct.controllers.StateFeedback` / :class:`~aimct.controllers.LQR`
(``u = u_ref - K (x - x_ref)``, optional per-channel output limits) and
:class:`~aimct.controllers.PID`.  Anything that solves an optimisation every
step (``LinearMPC``, ``ILQR``, ``SamplingMPC``) or carries a neural policy is
**not** portable this way -- distil it to a table or a gain first.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "ControllerSpec",
    "UnsupportedControllerError",
    "export_controller",
    "load_controller",
    "PortableController",
    "emit_c",
    "emit_micropython",
]

SCHEMA = "aimct.deploy/1"


class UnsupportedControllerError(TypeError):
    """Raised when a controller needs an online solver and cannot be exported."""


# --------------------------------------------------------------------------- spec


@dataclass
class ControllerSpec:
    r"""A serialisable description of a static control law.

    ``kind == "state_feedback"``: ``K`` ``(n_u, n_x)``, ``x_ref`` ``(n_x,)``,
    ``u_ref`` ``(n_u,)``, optional ``u_min`` / ``u_max`` ``(n_u,)``.

    ``kind == "pid"``: ``kp`` / ``ki`` / ``kd`` / ``setpoint`` each ``(n_u,)``,
    ``derivative_on`` in ``{"measurement", "error"}``, ``tau_d`` seconds,
    optional ``u_min`` / ``u_max`` ``(n_u,)``.  Windup is handled by
    conditional integration against the output limits (no separate integral
    clamp is exported).
    """

    kind: str
    n_x: int
    n_u: int
    dt: float | None = None
    # state feedback
    K: list | None = None
    x_ref: list | None = None
    u_ref: list | None = None
    # pid
    kp: list | None = None
    ki: list | None = None
    kd: list | None = None
    setpoint: list | None = None
    derivative_on: str = "measurement"
    tau_d: float = 0.0
    # common
    u_min: list | None = None
    u_max: list | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA

    def to_json(self, path: str | None = None, *, indent: int = 2) -> str:
        s = json.dumps(asdict(self), indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(s + "\n")
        return s

    @classmethod
    def from_json(cls, spec_or_path: "str | dict | ControllerSpec") -> "ControllerSpec":
        if isinstance(spec_or_path, ControllerSpec):
            return spec_or_path
        if isinstance(spec_or_path, dict):
            d = dict(spec_or_path)
        else:
            if isinstance(spec_or_path, (os.PathLike, pathlib.Path)) or (isinstance(spec_or_path, str) and "\n" not in spec_or_path and spec_or_path.rstrip().endswith(".json")):
                with open(spec_or_path, encoding="utf-8") as fh:
                    txt = fh.read()
            else:
                txt = spec_or_path
            d = json.loads(txt)
        if d.get("schema") != SCHEMA:
            raise ValueError(f"unknown spec schema {d.get('schema')!r}; expected {SCHEMA!r}")
        d.pop("schema", None)
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known}, schema=SCHEMA)


# ------------------------------------------------------------------------- export


def _vec(x, n) -> list:
    a = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    if a.size == 1 and n > 1:
        a = np.full(n, a.item())
    if a.size != n:
        raise ValueError(f"expected length {n}, got {a.size}")
    return a.tolist()


def _bounds(limits, n):
    """(low, high) tuple with None sides -> (u_min list|None, u_max list|None)."""
    if limits is None:
        return None, None
    lo, hi = limits
    lo = None if lo is None else _vec(lo, n)
    hi = None if hi is None else _vec(hi, n)
    return lo, hi


def export_controller(
    controller: Any,
    path: str | None = None,
    *,
    dt: float | None = None,
    u_bounds: tuple | None = None,
    meta: dict | None = None,
) -> ControllerSpec:
    r"""Introspect ``controller`` and return (and optionally write) a
    :class:`ControllerSpec`.

    ``u_bounds`` -- an optional ``(low, high)`` saturation to bake into the
    exported law (e.g. the actuator limit you passed to ``simulate``; static
    feedback classes don't store one themselves).  ``meta`` is merged into the
    spec's ``meta`` after an auto-stamp of ``created`` and the controller class.
    """
    from aimct.controllers import PID, StateFeedback

    stamp = {
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source_class": type(controller).__name__,
    }
    stamp.update(meta or {})

    if isinstance(controller, StateFeedback):
        K = np.atleast_2d(np.asarray(controller.K, dtype=float))
        n_u, n_x = K.shape
        lo, hi = _bounds(u_bounds, n_u)
        spec = ControllerSpec(
            kind="state_feedback",
            n_x=n_x,
            n_u=n_u,
            dt=dt,
            K=K.tolist(),
            x_ref=_vec(controller.x_ref, n_x),
            u_ref=_vec(controller.u_ref, n_u),
            u_min=lo,
            u_max=hi,
            meta=stamp,
        )
    elif isinstance(controller, PID):
        n_u = int(np.atleast_1d(np.asarray(controller.kp, dtype=float)).size)
        n_u = max(n_u, np.atleast_1d(np.asarray(controller.setpoint)).size) or 1
        has_out = getattr(controller, "output_limits", (None, None)) != (None, None)
        olo, ohi = _bounds(controller.output_limits if has_out else u_bounds, n_u)
        if getattr(controller, "integral_limits", (None, None)) != (None, None):
            import warnings

            warnings.warn(
                "PID.integral_limits are not exported; the portable executor "
                "relies on conditional anti-windup against the output limits.",
                stacklevel=2,
            )
        spec = ControllerSpec(
            kind="pid",
            n_x=n_u,
            n_u=n_u,
            dt=getattr(controller, "dt", None) or dt,
            kp=_vec(controller.kp, n_u),
            ki=_vec(controller.ki, n_u),
            kd=_vec(controller.kd, n_u),
            setpoint=_vec(controller.setpoint, n_u),
            derivative_on=getattr(controller, "derivative_on", "measurement"),
            tau_d=float(getattr(controller, "tau_d", 0.0)),
            u_min=olo,
            u_max=ohi,
            meta=stamp,
        )
    else:
        raise UnsupportedControllerError(
            f"{type(controller).__name__} is not portable: it needs an online "
            "solver or a learned policy. Export a static feedback gain, an "
            "explicit-MPC lookup table, or a distilled policy instead."
        )

    if path is not None:
        spec.to_json(path)
    return spec


# --------------------------------------------------------------------- runtime


class PortableController:
    r"""Pure-numpy executor for a :class:`ControllerSpec`.

    Implements the ``aimct`` controller protocol (``update(measurement, dt)``
    returning ``u``; ``reset()``), so a loaded ``controller.json`` runs
    unchanged inside :func:`aimct.simulate.simulate` and
    :class:`aimct.hil.RealTimeLoop` -- the same artefact you hand to the
    embedded target.
    """

    def __init__(self, spec: ControllerSpec) -> None:
        self.spec = spec
        self.kind = spec.kind
        self.n_x, self.n_u = spec.n_x, spec.n_u
        self._umin = None if spec.u_min is None else np.asarray(spec.u_min, float)
        self._umax = None if spec.u_max is None else np.asarray(spec.u_max, float)
        if spec.kind == "state_feedback":
            self.K = np.asarray(spec.K, float).reshape(self.n_u, self.n_x)
            self.x_ref = np.asarray(spec.x_ref, float)
            self.u_ref = np.asarray(spec.u_ref, float)
        elif spec.kind == "pid":
            self.kp = np.asarray(spec.kp, float)
            self.ki = np.asarray(spec.ki, float)
            self.kd = np.asarray(spec.kd, float)
            self.setpoint = np.asarray(spec.setpoint, float)
            self.derivative_on = spec.derivative_on
            self.tau_d = float(spec.tau_d)
        else:  # pragma: no cover - guarded at construction
            raise ValueError(f"unknown kind {spec.kind!r}")
        self.reset()

    # -- protocol -------------------------------------------------------------

    def reset(self) -> None:
        self._integ = np.zeros(self.n_u)
        self._prev_err = None
        self._prev_meas = None
        self._deriv = np.zeros(self.n_u)
        self.output = 0.0 if self.n_u == 1 else np.zeros(self.n_u)

    def update(self, measurement, dt: float | None = None):
        y = np.atleast_1d(np.asarray(measurement, dtype=float))
        if self.kind == "state_feedback":
            u = self.u_ref - self.K @ (y - self.x_ref)
        else:
            u = self._pid(y, dt)
        u = self._saturate(u)
        self.output = float(u[0]) if self.n_u == 1 else u
        return self.output

    # -- internals ----------------------------------------------------------

    def _saturate(self, u):
        if self._umin is not None:
            u = np.maximum(u, self._umin)
        if self._umax is not None:
            u = np.minimum(u, self._umax)
        return u

    def _pid(self, y, dt):
        h = self.spec.dt if dt is None else dt
        if h is None or h <= 0:
            raise ValueError("PID needs a positive dt (pass it, or set spec.dt)")
        err = self.setpoint - y
        p = self.kp * err

        if self._prev_err is None:
            self._prev_err = err
            self._prev_meas = y.copy()

        raw_d = (err - self._prev_err) / h if self.derivative_on == "error" \
            else -(y - self._prev_meas) / h
        if self.tau_d > 0.0:
            a = h / (self.tau_d + h)
            self._deriv = self._deriv + a * (raw_d - self._deriv)
        else:
            self._deriv = raw_d
        d = self.kd * self._deriv

        # conditional anti-windup: only accept the new integral if the
        # unsaturated command it produces is itself within the output limits
        # (identical to the emitted C / MicroPython executor).
        integ_next = self._integ + err * h
        unsat = p + self.ki * integ_next + d
        sat = self._saturate(unsat.copy())
        self._integ = np.where(sat == unsat, integ_next, self._integ)

        out = p + self.ki * self._integ + d
        self._prev_err = err
        self._prev_meas = y.copy()
        return out


def load_controller(spec_or_path) -> PortableController:
    """Build a :class:`PortableController` from a spec, dict, JSON text, or
    ``*.json`` path."""
    return PortableController(ControllerSpec.from_json(spec_or_path))


# --------------------------------------------------------------------- codegen


def _c_array(name, arr, ctype="float") -> str:
    arr = np.atleast_2d(np.asarray(arr, float))
    if arr.shape[0] == 1:
        body = ", ".join(f"{v:.10g}f" for v in arr[0])
        return f"static const {ctype} {name}[{arr.shape[1]}] = {{ {body} }};"
    rows = ",\n    ".join("{ " + ", ".join(f"{v:.10g}f" for v in r) + " }" for r in arr)
    return f"static const {ctype} {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n    {rows}\n}};"


def emit_c(spec: ControllerSpec) -> str:
    """A self-contained C reference executor for ``spec`` (no dynamic memory,
    no libc beyond nothing).  One function ``controller_update``."""
    s = ControllerSpec.from_json(spec)
    hdr = [
        "/* Generated by aimct.deploy.emit_c -- do not edit by hand. */",
        f"/* schema: {s.schema}   kind: {s.kind}   n_x={s.n_x} n_u={s.n_u} */",
    ]
    for k, v in s.meta.items():
        hdr.append(f"/*   {k}: {v} */")
    hdr += [f"#define AIMCT_N_X {s.n_x}", f"#define AIMCT_N_U {s.n_u}", ""]

    if s.kind == "state_feedback":
        lines = hdr + [
            _c_array("AIMCT_K", s.K),
            _c_array("AIMCT_X_REF", [s.x_ref]),
            _c_array("AIMCT_U_REF", [s.u_ref]),
            _c_array("AIMCT_U_MIN", [s.u_min]) if s.u_min else "",
            _c_array("AIMCT_U_MAX", [s.u_max]) if s.u_max else "",
            "",
            "void controller_update(const float y[AIMCT_N_X], float u[AIMCT_N_U]) {",
            "    for (int i = 0; i < AIMCT_N_U; ++i) {",
            "        float acc = AIMCT_U_REF[i];",
            "        for (int j = 0; j < AIMCT_N_X; ++j)",
            "            acc -= AIMCT_K[i][j] * (y[j] - AIMCT_X_REF[j]);",
        ]
        if s.u_min:
            lines.append("        if (acc < AIMCT_U_MIN[i]) acc = AIMCT_U_MIN[i];")
        if s.u_max:
            lines.append("        if (acc > AIMCT_U_MAX[i]) acc = AIMCT_U_MAX[i];")
        lines += ["        u[i] = acc;", "    }", "}", ""]
        return "\n".join(x for x in lines if x != "")

    # pid
    lines = hdr + [
        _c_array("AIMCT_KP", [s.kp]), _c_array("AIMCT_KI", [s.ki]),
        _c_array("AIMCT_KD", [s.kd]), _c_array("AIMCT_SP", [s.setpoint]),
        f"static const float AIMCT_DT = {float(s.dt or 0.0):.10g}f;",
        f"static const float AIMCT_TAU_D = {s.tau_d:.10g}f;",
        f"#define AIMCT_DERIV_ON_ERROR {1 if s.derivative_on == 'error' else 0}",
        _c_array("AIMCT_U_MIN", [s.u_min]) if s.u_min else "",
        _c_array("AIMCT_U_MAX", [s.u_max]) if s.u_max else "",
        "",
        "static float aimct_integ[AIMCT_N_U], aimct_deriv[AIMCT_N_U];",
        "static float aimct_prev_err[AIMCT_N_U], aimct_prev_meas[AIMCT_N_U];",
        "static int   aimct_have_prev = 0;",
        "",
        "void controller_reset(void) {",
        "    for (int i = 0; i < AIMCT_N_U; ++i) {",
        "        aimct_integ[i] = aimct_deriv[i] = 0.0f;",
        "    }",
        "    aimct_have_prev = 0;",
        "}",
        "",
        "void controller_update(const float y[AIMCT_N_X], float dt, float u[AIMCT_N_U]) {",
        "    float h = (dt > 0.0f) ? dt : AIMCT_DT;",
        "    for (int i = 0; i < AIMCT_N_U; ++i) {",
        "        float err = AIMCT_SP[i] - y[i];",
        "        if (!aimct_have_prev) { aimct_prev_err[i] = err; aimct_prev_meas[i] = y[i]; }",
        "        float raw_d = AIMCT_DERIV_ON_ERROR ? (err - aimct_prev_err[i]) / h",
        "                                           : -(y[i] - aimct_prev_meas[i]) / h;",
        "        if (AIMCT_TAU_D > 0.0f) {",
        "            float a = h / (AIMCT_TAU_D + h);",
        "            aimct_deriv[i] += a * (raw_d - aimct_deriv[i]);",
        "        } else aimct_deriv[i] = raw_d;",
        "        float integ_next = aimct_integ[i] + err * h;",
        "        float p = AIMCT_KP[i] * err;",
        "        float d = AIMCT_KD[i] * aimct_deriv[i];",
        "        float unsat = p + AIMCT_KI[i] * integ_next + d;",
        "        float sat = unsat;",
    ]
    if s.u_min:
        lines.append("        if (sat < AIMCT_U_MIN[i]) sat = AIMCT_U_MIN[i];")
    if s.u_max:
        lines.append("        if (sat > AIMCT_U_MAX[i]) sat = AIMCT_U_MAX[i];")
    lines += [
        "        if (sat == unsat) aimct_integ[i] = integ_next;  /* anti-windup */",
        "        float out = AIMCT_KP[i]*err + AIMCT_KI[i]*aimct_integ[i] + d;",
    ]
    if s.u_min:
        lines.append("        if (out < AIMCT_U_MIN[i]) out = AIMCT_U_MIN[i];")
    if s.u_max:
        lines.append("        if (out > AIMCT_U_MAX[i]) out = AIMCT_U_MAX[i];")
    lines += [
        "        u[i] = out;",
        "        aimct_prev_err[i] = err; aimct_prev_meas[i] = y[i];",
        "    }",
        "    aimct_have_prev = 1;",
        "}",
        "",
    ]
    return "\n".join(x for x in lines if x != "")


def emit_micropython(spec: ControllerSpec) -> str:
    """A self-contained MicroPython/Python reference executor for ``spec``.

    Plain lists and loops -- no numpy -- so it runs on an RP2040/ESP32 as-is
    and is import-compatible with CPython for a numeric cross-check.
    """
    s = ControllerSpec.from_json(spec)
    head = [
        "# Generated by aimct.deploy.emit_micropython -- do not edit by hand.",
        f"# schema: {s.schema}   kind: {s.kind}   n_x={s.n_x} n_u={s.n_u}",
    ]
    head += [f"#   {k}: {v}" for k, v in s.meta.items()]
    head.append("")

    if s.kind == "state_feedback":
        body = head + [
            f"K = {[[float(v) for v in row] for row in np.asarray(s.K, float).tolist()]}",
            f"X_REF = {[float(v) for v in s.x_ref]}",
            f"U_REF = {[float(v) for v in s.u_ref]}",
            f"U_MIN = {None if s.u_min is None else [float(v) for v in s.u_min]}",
            f"U_MAX = {None if s.u_max is None else [float(v) for v in s.u_max]}",
            "",
            "def controller_reset():",
            "    pass",
            "",
            "def controller_update(y, dt=None):",
            "    u = []",
            "    for i in range(len(U_REF)):",
            "        acc = U_REF[i]",
            "        for j in range(len(X_REF)):",
            "            acc -= K[i][j] * (y[j] - X_REF[j])",
            "        if U_MIN is not None and acc < U_MIN[i]: acc = U_MIN[i]",
            "        if U_MAX is not None and acc > U_MAX[i]: acc = U_MAX[i]",
            "        u.append(acc)",
            "    return u",
            "",
        ]
        return "\n".join(body)

    body = head + [
        f"KP = {[float(v) for v in s.kp]}",
        f"KI = {[float(v) for v in s.ki]}",
        f"KD = {[float(v) for v in s.kd]}",
        f"SP = {[float(v) for v in s.setpoint]}",
        f"DT = {float(s.dt or 0.0)}",
        f"TAU_D = {float(s.tau_d)}",
        f"DERIV_ON_ERROR = {s.derivative_on == 'error'}",
        f"U_MIN = {None if s.u_min is None else [float(v) for v in s.u_min]}",
        f"U_MAX = {None if s.u_max is None else [float(v) for v in s.u_max]}",
        "",
        "_integ = [0.0] * len(KP)",
        "_deriv = [0.0] * len(KP)",
        "_prev_err = None",
        "_prev_meas = None",
        "",
        "def controller_reset():",
        "    global _integ, _deriv, _prev_err, _prev_meas",
        "    _integ = [0.0] * len(KP)",
        "    _deriv = [0.0] * len(KP)",
        "    _prev_err = None",
        "    _prev_meas = None",
        "",
        "def controller_update(y, dt=None):",
        "    global _prev_err, _prev_meas",
        "    h = dt if (dt is not None and dt > 0.0) else DT",
        "    if _prev_err is None:",
        "        _prev_err = [SP[i] - y[i] for i in range(len(KP))]",
        "        _prev_meas = list(y)",
        "    u = []",
        "    for i in range(len(KP)):",
        "        err = SP[i] - y[i]",
        "        raw_d = (err - _prev_err[i]) / h if DERIV_ON_ERROR else -(y[i] - _prev_meas[i]) / h",
        "        if TAU_D > 0.0:",
        "            a = h / (TAU_D + h)",
        "            _deriv[i] += a * (raw_d - _deriv[i])",
        "        else:",
        "            _deriv[i] = raw_d",
        "        integ_next = _integ[i] + err * h",
        "        p = KP[i] * err",
        "        d = KD[i] * _deriv[i]",
        "        unsat = p + KI[i] * integ_next + d",
        "        sat = unsat",
        "        if U_MIN is not None and sat < U_MIN[i]: sat = U_MIN[i]",
        "        if U_MAX is not None and sat > U_MAX[i]: sat = U_MAX[i]",
        "        if sat == unsat:",
        "            _integ[i] = integ_next",
        "        out = p + KI[i] * _integ[i] + d",
        "        if U_MIN is not None and out < U_MIN[i]: out = U_MIN[i]",
        "        if U_MAX is not None and out > U_MAX[i]: out = U_MAX[i]",
        "        u.append(out)",
        "    for i in range(len(KP)):",
        "        _prev_err[i] = SP[i] - y[i]",
        "        _prev_meas[i] = y[i]",
        "    return u",
        "",
    ]
    return "\n".join(body)
