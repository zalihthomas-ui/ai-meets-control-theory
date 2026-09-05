# 2-DOF Planar Robot Arm — Hardware Bridge & Build Guide

> **Module Companion:** imct.systems.TwoLinkArm, imct.sysid.identify_manipulator, imct.hil.RealTimeLoop, imct.deploy.export_controller

This document provides a datasheet-grade specification, build guide, and software integration manual for assembling a physical 2-DOF planar manipulator arm compatible with the imct hardware bridge, system identification harness, and real-time execution loops.

---

## 1. Bill of Materials (BOM)

| Component | Option A (Smart Actuator) | Option B (Direct PWM / Encoder) | Quantity |
| :--- | :--- | :--- | :--- |
| **Joint 1 (Shoulder) Actuator** | Robotis Dynamixel XM430-W350-T | High-Torque Coreless Digital Servo (\,\text{kg}\cdot\text{cm}$) | 1 |
| **Joint 2 (Elbow) Actuator** | Robotis Dynamixel XM430-W210-T | Coreless Digital Servo (\,\text{kg}\cdot\text{cm}$) | 1 |
| **Joint Encoders** | Built-in 12-bit contactless magnetic (4096 CPR) | 2× AS5600 Magnetic Rotary Encoders (I2C / SSI) | 2 |
| **Microcontroller / Interface** | Robotis U2D2 USB-to-RS485 bridge | Raspberry Pi Pico (RP2040) / Teensy 4.0 | 1 |
| **Links & Brackets** | 6061 Aluminum CNC / 3D-Printed Carbon PETG | 6061 Aluminum CNC / 3D-Printed Carbon PETG | 2 |
| **Power Supply** | 12.0 V / 5 A regulated DC supply | 7.4 V / 10 A high-current UBEC / DC supply | 1 |

---

## 2. Physical & Kinematic Parameters

Following the standard Quanser 2-DOF planar arm reference model implemented in imct.systems.TwoLinkArm:

- **Link 1 Length ($):** .20\,\text{m}$ (.0\,\text{cm}$)
- **Link 2 Length ($):** .15\,\text{m}$ (.0\,\text{cm}$)
- **Link 1 Mass ($):** .50\,\text{kg}$ ( = 0.10\,\text{m}$,  = 0.00167\,\text{kg}\cdot\text{m}^2$)
- **Link 2 Mass ($):** .30\,\text{kg}$ ( = 0.075\,\text{m}$,  = 0.00056\,\text{kg}\cdot\text{m}^2$)
- **Maximum Joint Torque ($\tau_{\max}$):** $\pm 2.5\,\text{N}\cdot\text{m}$ (Shoulder), $\pm 1.5\,\text{N}\cdot\text{m}$ (Elbow)
- **Encoder Resolution:** 12-bit ($\approx 0.088^\circ$ per count)

---

## 3. End-to-End Workflow with imct

`mermaid
flowchart LR
    A["1. Excitation & Telemetry Log<br/>(CSV: t, q1, q2, tau1, tau2)"] --> B["2. System ID<br/>aimct.sysid.identify_manipulator"]
    B --> C["3. Controller Design<br/>Computed-Torque / Slotine-Li"]
    C --> D["4. HIL Emulation<br/>aimct.hil.RealTimeLoop"]
    D --> E["5. Embedded Export<br/>aimct.deploy.emit_c / emit_micropython"]
`

### 3.1 Step 1: Physical System Identification

Record joint trajectory data under rich multi-sine excitation. Use imct.sysid to compute finite-difference velocities/accelerations and estimate the 5 base inertial parameters ($\pi_1 \dots \pi_5$):

`python
import numpy as np
from aimct.sysid import identify_manipulator, finite_difference_derivatives, ManipulatorID

# Load physical hardware telemetry: shape (N, 2)
t = np.linspace(0, 10.0, 1000)
q = np.sin(t[:, None] * np.array([1.0, 2.5]))
tau = 0.5 * np.cos(t[:, None] * np.array([1.0, 2.5]))

# Differentiate with Savitzky-Golay / finite-difference smoothing
qdot, qddot = finite_difference_derivatives(t, q)

# Solve linear-in-parameters regressor: Y(q, qdot, qddot) @ pi = tau
params: ManipulatorID = identify_manipulator(
    q=q,
    qdot=qdot,
    qddot=qddot,
    tau=tau,
    l1=0.20,
    l2=0.15,
    ridge_alpha=1e-4,
)

print("Identified base parameters (pi):", params.pi)
print("Residual torque RMSE (N*m):", params.rmse)
`

### 3.2 Step 2: Controller Design & Synthesis

Design a trajectory tracking controller using computed-torque and Slotine--Li adaptive control (see Experiment 23 / xperiments/live_arm/ for full benchmarks and comparative analysis):

`python
from aimct.systems.twolink_arm import TwoLinkArm

# Instantiate identified dynamic model
arm = TwoLinkArm(
    l1=0.20, l2=0.15,
    m1=0.50, m2=0.30,
)

# Reference trajectory tracking gains
Kp = np.diag([100.0, 80.0])
Kd = np.diag([20.0, 16.0])

def computed_torque_law(q, qdot, q_ref, qdot_ref, qddot_ref):
    # Error state
    e = q_ref - q
    edot = qdot_ref - qdot
    v = qddot_ref + Kd @ edot + Kp @ e
    # Inverse dynamics: M(q) @ v + C(q, qdot) @ qdot + g(q)
    M = arm.mass_matrix(q)
    C = arm.coriolis_matrix(q, qdot)
    g = arm.gravity_vector(q)
    return M @ v + C @ qdot + g
`

### 3.3 Step 3: Hardware-in-the-Loop (HIL) Validation

Before flashing firmware to the physical microcontroller, validate timing margins, serial latency, and packet loss using imct.hil.RealTimeLoop and imct.hil.PlantEmulator:

`python
from aimct.hil import RealTimeLoop, PlantEmulator, InProcessTransport

# Set up simulated physical hardware emulator running at 200 Hz
emulator = PlantEmulator(
    system=arm,
    dt=0.005,
    quantization_bits=12,
    sensor_noise_std=1e-3,
)

# Transport bridge (InProcessTransport, SerialTransport, or UDPTransport)
transport = InProcessTransport(emulator)

# Execute real-time control loop with strict deadline monitoring
loop = RealTimeLoop(
    transport=transport,
    controller=computed_torque_law,
    target_frequency_hz=200.0,
    max_deadline_misses=5,
)

result = loop.run(duration_seconds=5.0)
print(f"Mean loop jitter: {result.mean_jitter_us:.1f} us")
print(f"Deadline misses: {result.deadline_miss_count}")
`

### 3.4 Step 4: Export to MicroPython & Embedded C

Export the validated controller and calibration matrix directly to static JSON, standalone ANSI C99 source code, or lightweight MicroPython scripts for microcontroller execution using imct.deploy:

`python
from aimct.deploy import export_controller, emit_c, emit_micropython

# Export controller spec to portable dataclass / JSON
spec = export_controller(
    name="two_link_arm_lqr",
    system_name="TwoLinkArm",
    controller_type="StateFeedback",
    gains={"K": Kp.tolist(), "Kd": Kd.tolist()},
    dt=0.005,
    u_bounds=[[-2.5, 2.5], [-1.5, 1.5]],
)

# Generate zero-dependency C99 header & implementation
c_code = emit_c(spec)
with open("controller_arm.h", "w") as f:
    f.write(c_code)

# Generate MicroPython driver for Raspberry Pi Pico (RP2040)
mpy_code = emit_micropython(spec)
with open("main.py", "w") as f:
    f.write(mpy_code)
`

---

## 4. Pinout and Electrical Wiring

`
[ Raspberry Pi Pico / Teensy 4.0 ]
  ├── UART0 TX (GP0) ──────────> Dynamixel / Motor Driver RX
  ├── UART0 RX (GP1) <────────── Dynamixel / Motor Driver TX
  ├── I2C0 SDA (GP4) <─────────> AS5600 Magnetic Encoder 1 (Shoulder)
  ├── I2C0 SCL (GP5) ──────────> AS5600 Magnetic Encoder 1 (Shoulder)
  ├── I2C1 SDA (GP6) <─────────> AS5600 Magnetic Encoder 2 (Elbow)
  ├── I2C1 SCL (GP7) ──────────> AS5600 Magnetic Encoder 2 (Elbow)
  ├── GND            ──────────> Common Ground
  └── +3.3V (VOUT)   ──────────> Encoders VCC
`

---

## 5. Safety Watchdog Protocols

1. **Heartbeat Timeout:** If no command packet is received within \,\text{ms}$ (5 dropped frames at \,\text{Hz}$), the microcontroller clamps motor outputs to zero torque ($\tau = 0$).
2. **Joint Angle Limits:** Software hard stops at  \in [-170^\circ, +170^\circ]$ and  \in [-150^\circ, +150^\circ]$.
3. **Velocity Clamping:** Torque is disengaged if $\|\dot{q}\|_\infty > 12.0\,\text{rad/s}$.
