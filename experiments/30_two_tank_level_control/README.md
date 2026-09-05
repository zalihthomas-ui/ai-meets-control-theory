# Experiment 30: Coupled Two-Tank Level Regulation & Interactive Capacity Benchmark

## 1. Executive Summary

This experiment evaluates classical single-loop feedback (**SISO PI**), state-space optimal control (**Multivariable LQR**), and receding-horizon quadratic programming (**Linear MPC**) on the **Coupled Two-Tank** process control benchmark using parameters from the **Quanser Coupled Tanks** laboratory apparatus.

```
                  +-------------------------+
                  | Variable-Speed DC Pump  |
                  |     Flow F_in(u)        |
                  +------------+------------+
                               |
                               v
                       +---------------+
                       |    TANK 1     |  Area A1 = 15.55 cm^2
                       |   Level h1    |
                       +-------+-------+
                               | Inter-tank orifice a12 (Torricelli)
                               v
                       +---------------+
                       |    TANK 2     |  Area A2 = 15.55 cm^2
                       |   Level h2    |  Process Variable (Target h2*)
                       +-------+-------+
                               | Bottom drain orifice a2
                               v (Gravity Outflow)
```

---

## 2. Physical Principles & Governing Hydraulics

The system dynamics are derived in [`docs/references/two-tank-reference.md`](../../docs/references/two-tank-reference.md):

$$\begin{aligned}
\frac{dh_1}{dt} &= \frac{1}{A_1} \left[ K_p V_p - a_{12} \operatorname{sign}(h_1 - h_2) \sqrt{2 g |h_1 - h_2|} \right] \\
\frac{dh_2}{dt} &= \frac{1}{A_2} \left[ a_{12} \operatorname{sign}(h_1 - h_2) \sqrt{2 g |h_1 - h_2|} - a_2 \sqrt{2 g h_2} \right]
\end{aligned}$$

### 2.1 Canonical Hardware Parameters (Quanser Coupled Tanks Standard)
- **Tanks**: $A_1 = A_2 = 1.555 \times 10^{-3}\text{ m}^2$ ($D = 4.45\text{ cm}$), $h_{\max} = 0.30\text{ m}$ ($30\text{ cm}$).
- **Orifices**: $a_{12} = a_2 = 1.81 \times 10^{-5}\text{ m}^2$ ($d = 4.8\text{ mm}$).
- **Pump**: $K_p = 3.3 \times 10^{-6}\text{ m}^3/(\text{s}\cdot\text{V})$, $V_{\max} = 12.0\text{ V}$.
- **Operating Equilibrium**: $h_{2,0} = 10\text{ cm} \implies h_{1,0} = 20\text{ cm}, V_{p,0} = 7.68\text{ V}$.
- **Step Target**: $h_{2}^* = 15\text{ cm} \implies h_{1}^* = 30\text{ cm}, V_{p}^* = 9.41\text{ V}$.

---

## 3. Benchmark Quantitative Results

| Controller | Rise $t_r$ [s] | Settling $t_s$ [s] | Overshoot $M_p$ [%] | Steady error $e_{ss}$ [cm] | RMSE [cm] | Energy $E_u$ [$\text{V}^2\cdot\text{s}$] | Peak Voltage $V_{\max}$ [V] | Saturation [%] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SISO PI** | 20.30 | 32.70 | 0.0 | 0.00 | 1.90 | 8546.4 | 11.50 | 0.0 | **Stable** |
| **Multivariable LQR** | 49.60 | 75.00 | 0.0 | 0.80 | 2.73 | 6667.7 | 9.45 | 0.0 | **Stable** |
| **Linear MPC** | 49.90 | 75.00 | 0.0 | 0.82 | 2.74 | 6659.0 | 9.45 | 0.0 | **Stable** |

---

## 4. Key Findings

1. **Interactive Capacity Lag**: The intermediate liquid accumulation in Tank 1 acts as a dynamic hydraulic buffer, imposing a characteristic non-minimum-phase/lag relationship between pump voltage $V_p$ and process variable $h_2$.
2. **SISO PI with Anti-Windup**: Single-loop PI with anti-windup clamping successfully drives steady-state error to zero ($e_{ss} = 0.00\text{ cm}$) by integrating residual error against the nonlinear $\sqrt{h}$ square-root loss.
3. **Multivariable State Coordination**: LQR and Linear MPC coordinate both tank levels $[h_1, h_2]$, maintaining energy-optimal voltage profiles ($E_u \approx 6660\text{ V}^2\cdot\text{s}$ vs. PI's $8546\text{ V}^2\cdot\text{s}$) while strictly respecting physical level ($h_1 \le 30\text{ cm}$) and voltage ($V_p \le 12\text{ V}$) limits.

---

## 5. Artifacts & Reproducibility

- `config.yaml`: Canonical plant parameters and controller hyperparameter definitions.
- `run.py`: Reproducible evaluation runner generating figures and metric tables.
- `two_tank_benchmark.png`: 4-panel comparison for step level transition ($h_1(t)$, $h_2(t)$, $V_p(t)$, Phase portrait).
- `two_tank_setpoint.png`: Multi-step setpoint tracking trajectory ($10\text{ cm} \to 15\text{ cm} \to 8\text{ cm}$) demonstrating nonlinear capacity handling.
- `table.md` / `table.csv`: Raw and formatted benchmark data.
