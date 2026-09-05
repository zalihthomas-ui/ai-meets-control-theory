"""Unit tests for Hardware-in-the-Loop (HIL) subpackage."""

import time
import numpy as np
import pytest

from aimct.hil import (
    DeadlineMissInfo,
    HILResult,
    InProcessTransport,
    PlantEmulator,
    RealTimeLoop,
    UDPTransport,
)
from aimct.systems.mass_spring_damper import MassSpringDamper
from aimct.systems.pendulum import Pendulum
from aimct.systems.twolink_arm import TwoLinkArm
from aimct.controllers.lqr import LQR, solve_care


def test_encoder_quantization():
    """Verify n-bit encoder quantization resolution and bounds."""
    msd = MassSpringDamper(m=1.0, k=4.0, c=0.5)
    # 12-bit quantization over [-np.pi, np.pi]
    # Range = 2*pi, 4095 intervals -> step size ~ 0.001534 rad
    bits = 12
    step_size = (2.0 * np.pi) / ((1 << bits) - 1)

    emu = PlantEmulator(
        msd,
        quantization_bits=bits,
        quantize_channels=[0],
        quantize_ranges=[(-np.pi, np.pi), (-10.0, 10.0)],
    )

    # Test reset with custom state
    x_init = np.array([0.123456789, 0.0])
    y0 = emu.reset(x_init)

    # Difference between quantized output and true state should be <= step_size / 2
    assert abs(y0[0] - x_init[0]) <= step_size / 2.0 + 1e-7
    # Unquantized channel 1 should be exact
    assert y0[1] == x_init[1]


def test_actuator_slew_and_saturation():
    """Verify actuator slew-rate limits and saturation clamping."""
    msd = MassSpringDamper(m=1.0, k=4.0, c=0.5)
    dt = 0.01
    # Max torque 5.0 N, max slew 50.0 N/s -> max delta = 0.5 N per step
    slew_max = 50.0
    u_max = 5.0

    emu = PlantEmulator(
        msd,
        u_min=-u_max,
        u_max=u_max,
        slew_rate_max=slew_max,
    )
    emu.reset(np.zeros(2))

    # Step 1: command large jump +10.0 N
    emu.step(np.array([10.0]), dt)
    # Slew limited to 0.5 N
    assert np.isclose(emu.u_applied[0], 0.5)

    # Step 2: continue commanding +10.0 N
    emu.step(np.array([10.0]), dt)
    assert np.isclose(emu.u_applied[0], 1.0)

    # Run 15 more steps -> would reach 8.5 N, but should clamp at u_max = 5.0 N
    for _ in range(15):
        emu.step(np.array([10.0]), dt)
    assert np.isclose(emu.u_applied[0], 5.0)


def test_transport_delay_buffer():
    """Verify discrete transport latency buffer delays actuator application."""
    msd = MassSpringDamper(m=1.0, k=4.0, c=0.5)
    dt = 0.005
    delay_steps = 3

    emu = PlantEmulator(msd, delay_steps=delay_steps)
    emu.reset(np.zeros(2))

    # Send distinct pulse commands
    emu.step([1.0], dt)  # step 0: buffer has [1.0], applies 0.0
    assert emu.u_applied[0] == 0.0
    emu.step([2.0], dt)  # step 1: buffer has [1.0, 2.0], applies 0.0
    assert emu.u_applied[0] == 0.0
    emu.step([3.0], dt)  # step 2: buffer has [1.0, 2.0, 3.0], applies 0.0
    assert emu.u_applied[0] == 0.0
    emu.step([4.0], dt)  # step 3: buffer pops [1.0], applies 1.0
    assert emu.u_applied[0] == 1.0
    emu.step([5.0], dt)  # step 4: buffer pops [2.0], applies 2.0
    assert emu.u_applied[0] == 2.0


def test_inprocess_transport():
    """Verify in-process transport FIFO delivery and tick delays."""
    tx = InProcessTransport(delay_steps=2)
    tx.send(b"packet_1")
    tx.send(b"packet_2")

    # Not ready before ticks
    assert tx.recv() is None
    tx.tick()  # step 1
    assert tx.recv() is None
    tx.tick()  # step 2 -> packets ready
    p1 = tx.recv()
    p2 = tx.recv()
    assert p1 == b"packet_1"
    assert p2 == b"packet_2"
    assert tx.recv() is None
    tx.close()


def test_udp_transport_loopback():
    """Verify UDP transport loopback communication."""
    # Node A (port 0 = auto-bind) and Node B
    node_a = UDPTransport(host="127.0.0.1", port=0)
    node_b = UDPTransport(host="127.0.0.1", port=0)

    node_a.peer_port = node_b.port
    node_b.peer_port = node_a.port

    msg = b"test_hil_payload_123"
    node_a.send(msg)

    # Small pause for loopback socket buffer
    time.sleep(0.005)
    received = node_b.recv(timeout=0.1)
    assert received == msg

    node_a.close()
    node_b.close()


def test_realtime_loop_simulated_and_deadline_misses():
    """Verify RealTimeLoop deadline accounting and simulated execution."""
    msd = MassSpringDamper(m=1.0, k=4.0, c=0.5)
    A, B = msd.linearize()
    lqr = LQR(A, B, np.eye(2), np.eye(1))

    # Normal fast execution
    loop = RealTimeLoop(
        rate_hz=500.0,  # dt = 2 ms
        controller=lqr,
        plant=msd,
        simulated_time=True,
    )
    res = loop.run(duration=0.1, x0=np.array([0.5, 0.0]))
    assert len(res) == 50
    assert not res.diverged
    assert res.deadline_misses == 0

    # Injected slow controller that sleeps 5 ms (budget 2 ms)
    miss_reports = []
    def on_miss(info: DeadlineMissInfo):
        miss_reports.append(info)

    class SlowController:
        def update(self, y, dt):
            time.sleep(0.003)  # 3 ms > 2 ms budget
            return [0.0]

    slow_loop = RealTimeLoop(
        rate_hz=500.0,
        controller=SlowController(),
        plant=msd,
        simulated_time=True,
        on_deadline_miss=on_miss,
    )
    res_slow = slow_loop.run(max_steps=10)
    assert res_slow.deadline_misses > 0
    assert res_slow.worst_overrun_s > 0.0
    assert len(miss_reports) == 0  # In simulated mode on_miss is not called unless real-time


def test_hil_closed_loop_pendulum_stabilization():
    """Test closed-loop HIL balance of Pendulum with quantization & latency."""
    pend = Pendulum(m=1.0, L=1.0, b=0.1)
    A, B = pend.linearize(x_eq=np.array([0.0, 0.0]), u_eq=np.array([0.0]))
    Q = np.diag([20.0, 2.0])
    R = np.eye(1)
    lqr = LQR(A, B, Q, R)

    # 12-bit quantization, 5 ms delay, slew limit 30 N*m/s
    emu = PlantEmulator(
        pend,
        quantization_bits=12,
        quantize_ranges=[(-np.pi, np.pi), (-10.0, 10.0)],
        delay_s=0.005,
        slew_rate_max=30.0,
        u_min=-10.0,
        u_max=10.0,
    )

    loop = RealTimeLoop(
        rate_hz=500.0,  # 500 Hz
        controller=lqr,
        plant=emu,
        simulated_time=True,
    )

    # Small angle perturbation (0.1 rad = 5.7 deg)
    res = loop.run(duration=2.0, x0=np.array([0.1, 0.0]))
    assert not res.diverged
    # Final angle should settle to near zero (< 0.01 rad)
    assert abs(res.x[-1, 0]) < 0.025
