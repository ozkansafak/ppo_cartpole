# PPO Cartpole

This project trains a PPO (Proximal Policy Optimization) agent to solve the classic CartPole problem in Gymnasium. Gymnasium simulates the dynamics of the cart.

At each time step, the agent chooses one of two actions:
- apply a constant force to the left
- apply a constant force to the right.

On top of standard CartPole, this project adds two complications, each implemented as a Gymnasium wrapper in the notebook:

1. **External forcing** (`ExternalForcing`). Wind blows on the pole: a steady mean wind plus random turbulent gusts, which produce an aerodynamic drag load along the pole. The agent never observes the wind directly; it only sees its effect on the cart and pole. See [External forcing: wind](#external-forcing-wind).

2. **Perception latency** (`PerceptionLatency`). The agent's perception lags the true state by 0.04 s (2 time steps). To compensate, each observation contains the two most recent delayed states plus the two actions the agent has taken since, from which it can infer the current state. See [Perception latency](#perception-latency).

## Why neural networks instead of a lookup table?

Tabular RL stores $Q(s,a)$ or $V(s)$ in a lookup table and updates them with Bellman equations. This method can be employed when the set of states are finite and small.

In this project, time is discretized ($\Delta t = 0.02$ s) and the actions are discrete (two choices), but the state $[x, \dot{x}, \theta, \dot{\theta}]$ is continuous: each variable is a real number and is never binned. A lookup table would need the state discretized first, and the table grows exponentially with the number of variables. With the perception latency, each observation has 10 numbers, so even a coarse 20 bins per variable would give $20^{10} \approx 10^{13}$ cells.

So PPO uses neural networks, which take the real-valued state directly and generalize between nearby states, and learns from sampled trajectories instead of sweeping over all states. GAE still uses Bellman-style temporal-difference errors.

## Physics model

Cartpole is a simple coupled mechanical system: a cart moves along the x-axis while a pole rotates about the pivot.

Gymnasium models the pole as a uniform rod of mass $m$ and length $2l$ (its `length` parameter is the half-length $l$) on a cart of mass $M$. The angle $\theta$ is measured from upright, positive when the pole leans right. With a horizontal force $F$ on the cart, the coupled equations of motion are:

$$
(M + m)\,\ddot{x} + m l \cos\theta\,\ddot{\theta} - m l \dot{\theta}^2 \sin\theta = F
$$

$$
\tfrac{4}{3} m l^2\,\ddot{\theta} + m l \cos\theta\,\ddot{x} - m g l \sin\theta = 0
$$

$\tfrac{4}{3} m l^2$ is the rod's moment of inertia about the pivot. The $\cos\theta$ terms couple the two: accelerating the cart tips the pole, and the swinging pole pushes back on the cart. Gravity ($+m g l \sin\theta$) makes the upright position unstable.

There is **no friction**: neither between the cart and the track nor at the pivot. (The original Barto et al. (1983) model had both; Gymnasium drops them.)

| Parameter | Symbol | Value |
|---|---|---|
| Cart mass | $M$ | 1.0 kg |
| Pole mass | $m$ | 0.1 kg |
| Pole half-length | $l$ | 0.5 m (full pole 1.0 m) |
| Gravity | $g$ | 9.8 m/s² |
| Agent's force on the cart | $F$ | ±10 N |
| Time step | $\Delta t$ | 0.02 s |

## Time integration

Gymnasium advances the system in discrete time steps of $\Delta t = 0.02$ s. At each step:

1. The agent's force $F = \pm 10$ N is held constant over the whole step.
2. The equations of motion above are solved for $\ddot{x}$ and $\ddot{\theta}$ from the current state.
3. The state is advanced with **explicit (forward) Euler** integration. Positions use the velocities from the *start* of the step:

$$
x_{n+1} = x_n + \Delta t\,\dot{x}_n, \qquad \dot{x}_{n+1} = \dot{x}_n + \Delta t\,\ddot{x}_n
$$

$$
\theta_{n+1} = \theta_n + \Delta t\,\dot{\theta}_n, \qquad \dot{\theta}_{n+1} = \dot{\theta}_n + \Delta t\,\ddot{\theta}_n
$$

This is a first-order scheme with no sub-stepping. Gymnasium can also use semi-implicit Euler (`kinematics_integrator="semi-implicit euler"`), which updates velocities first and then advances positions with the new velocities. This project uses the default, explicit Euler.

The step size resolves the dynamics comfortably. Linearized about upright, the pole falls away exponentially at rate $\lambda = \sqrt{g \,/\, \big(l\,(\tfrac{4}{3} - \tfrac{m}{M+m})\big)} \approx 4.0\ \text{s}^{-1}$, so $\lambda\,\Delta t \approx 0.08$: about 12 steps per e-folding of the instability. Explicit Euler is not energy-conserving, but the pole never swings past 12° (see below), so the drift is negligible here.

## Episodes, termination and reward

An **episode** is one attempt to balance the pole, from a reset until it fails or runs out of time. The start state is random, with each of $x, \dot{x}, \theta, \dot{\theta}$ drawn uniformly from $[-0.05, 0.05]$.

- **Failure (termination):** the episode ends as soon as the pole tilts more than 12° from upright ($|\theta| > 0.2094$ rad) or the cart leaves the track ($|x| > 2.4$ m).
- **Time limit (truncation):** otherwise the episode is cut off after 500 steps, i.e. 10 s of simulated time.
- **Reward:** +1 for every step the pole stays up, so an episode's total reward is the number of steps it survived. The maximum is 500.

These rules are part of the environment, so they apply every time the policy runs: during training (a failed episode triggers a reset and a new attempt) and when rendering the GIF (the recording stops at the end of the episode).

**Why failure drives learning.** The reward is the same +1 on every step, so the only thing that distinguishes good behavior from bad is *when the episode ends*. Failing early means fewer +1 rewards; the 12° and 2.4 m limits are what turn "balance the pole" into a learnable objective. In training, the two kinds of episode end are treated differently:

- **Failure:** the future reward is zero, since the pole has fallen. Actions that led toward failure get lower advantages, and the policy learns to avoid them.
- **Time limit:** the episode was cut off, not failed. The pole could have stayed up, so the critic's estimate of future reward $V(s)$ is used in place of the missing future (bootstrapping). Treating the 500-step cut-off as a failure would wrongly teach the agent that surviving to the end is bad.

## External forcing: wind

Wind blows horizontally on the pole, modeled the way wind loads are modeled in engineering: a mean wind plus turbulent gusts, converted to a force by aerodynamic drag.

**Wind speed.** The wind speed is a steady mean plus a random fluctuation:

$$
u(t) = U + u'(t)
$$

The gust $u'(t)$ follows the **von Kármán turbulence spectrum**, the standard model for atmospheric turbulence:

$$
S_u(f) = \sigma_u^2\,\frac{4\,T_L}{\big(1 + 70.8\,(f\,T_L)^2\big)^{5/6}}, \qquad T_L = \frac{L}{U}
$$

where $\sigma_u$ is the gust intensity and $T_L$ the integral time scale (roughly, how long a gust lasts). At high frequency the spectrum falls off as $f^{-5/3}$, Kolmogorov's inertial-range law. $u'(t)$ is synthesized as a sum of 400 Fourier modes between 0.02 and 20 Hz, with amplitudes $\sqrt{2\,S_u(f)\,\Delta f}$ and random phases. The phases are drawn fresh at the start of every episode, so every episode has different gusts with the same statistics.

**Drag force.** The wind exerts quadratic drag on the pole, treated as a circular cylinder:

$$
F(t) = \tfrac{1}{2}\,\rho\,C_d\,A\,|u(t)|\,u(t)
$$

with frontal area $A$ = pole diameter × pole length. The load is spread uniformly along the pole, so its resultant acts at **mid-pole**, a height $l$ above the pivot (at $x + l\sin\theta$).

| Parameter | Symbol | Value |
|---|---|---|
| Mean wind speed | $U$ | 2.5 m/s (a light breeze) |
| Turbulence intensity | $\sigma_u / U$ | 30% ($\sigma_u$ = 0.75 m/s) |
| Turbulence length scale | $L$ | 5 m ($T_L$ = 2 s) |
| Air density | $\rho$ | 1.2 kg/m³ |
| Drag coefficient (cylinder) | $C_d$ | 1.2 |
| Pole diameter | $D$ | 2 cm (frontal area 0.02 m²) |
| Resulting drag | $F$ | about 0.09 N mean, 0.01–0.22 N over an episode |

The same wind model is used in training and in the GIF.

![Wind speed and drag force over one 10 s episode](resources/external_forcing.png)

The figure shows one 10 s episode (seeded for reproducibility). Top: the wind speed, with slow gusts lasting a few seconds and small fast fluctuations on top, as in real wind. Bottom: the resulting drag. Because drag goes as $u^2$, gusts are amplified: the wind varies by about ±30%, but the force varies by more than a factor of ten.

**Applying the force.** Over one time step $\Delta t$, $F$ is applied as an impulse $J = F\,\Delta t$ with generalized components $(J,\; l \cos\theta\,J)$, which changes $\dot{x}$ and $\dot{\theta}$ through the mass matrix of the equations above. The impulse is applied just before Gymnasium's Euler step.

**Why the wind is limited to a light breeze.** Real wind has a nonzero mean, so to stay balanced the pole must lean *into* the wind. With the drag and the pole's weight both acting at mid-pole, the steady equilibrium lean is

$$
\sin\theta_{eq} = \frac{F}{m\,g}
$$

The pole weighs only 0.1 kg ($m g$ = 0.98 N), so it is very sensitive to wind. At 2.5 m/s the mean drag of 0.09 N gives a lean of about 5°. At 4 m/s the lean would be about 14°, beyond the 12° failure limit: no controller could keep the pole up. The largest tolerable mean drag is $m g \sin 12° \approx 0.20$ N.

**What the agent has to do.** The per-step kick from the wind is small compared with the agent's own push:

| Force | $\Delta\dot{\theta}$ per step |
|---|---|
| Agent's 10 N push on the cart | 0.29 rad/s |
| Mean drag, 0.09 N at mid-pole | 0.03 rad/s |
| Peak gust drag, 0.22 N at mid-pole | 0.06 rad/s |

The difficulty is that the wind pushes **persistently in one direction**. The whole system is blown downwind, so the agent has to hold the pole tilted into the wind and push back on average to keep the cart from drifting off the track, all while the gusts change the required lean every few seconds. Without that correction, even a good balancing controller is blown off the end of the track within a few seconds.

## Perception latency

The agent perceives the state $\tau = 0.04$ s (2 time steps) late: at step $n$ it sees $s_{n-2}$, not $s_n$. The simulation itself always uses the true state.

To compensate, each observation contains 10 numbers:

$$
\big[\; s_{n-3} \;\big|\; s_{n-2} \;\big|\; a_{n-1},\ a_n \;\big]
$$

the two newest perceived frames (4 numbers each), plus the two actions taken since the newest one (+1 right, −1 left). From these, the network can predict the current state forward, which is what latency compensation does in an autonomous-vehicle stack.

| Parameter | Value |
|---|---|
| Perception latency $\tau$ | 0.04 s (2 steps) |
| Perceived frames per observation | 2 |
| Observation size | 10 |

## Initial conditions

- **Cart and pole:** each of $x, \dot{x}, \theta, \dot{\theta}$ is drawn uniformly from $[-0.05, 0.05]$ (m, m/s, rad, rad/s), so the pole starts within about ±2.9° of upright and nearly at rest.

- **Wind:** every episode starts at $t = 0$ with new random gust phases, so each episode has a different gust history with the same statistics.

- **Perception buffer:** at reset there is no history yet, so all perceived frames are filled with the initial state and the recent actions with 0.

- **Random seeds:** PyTorch and NumPy are seeded with 42 at the top of the notebook. Training episodes are not individually seeded, so training results vary slightly from run to run. The GIF and the evaluation episodes use fixed seeds 0–19 and are reproducible for a given trained policy.

## PPO agent

In this project:

- the actor network outputs action probabilities from the observed state,
- the critic network estimates the expected return from that state,
- PPO updates the policy using a clipped objective to keep learning stable.

This is the key idea behind modern reinforcement learning: use function approximation to generalize across a huge or continuous state space.

## Trained agent animation

![Trained agent balancing the pole in turbulent wind](resources/cartpole_trained_rollout.gif)

The GIF shows one episode of inference with the trained policy. 

- **Deterministic policy:** at each step the agent takes its most likely action instead of sampling one as in training: $a = \arg\max_a \pi(a \mid s)$.

- **Same conditions as training:** the same wind model and the same 0.04 s perception latency.

- **Which episode:** the notebook first runs 20 evaluation episodes with fixed seeds 0–19 and prints the length of each. The GIF replays the longest one. In the current run, 18 of the 20 episodes lasted the full 500 steps (the other two failed at 336 and 357 steps), so the GIF shows typical behavior, not a lucky exception. Training episodes are not seeded, so these numbers vary somewhat from run to run: recent runs gave 18 to 20 of 20.

- **What is drawn:** Gymnasium renders the *true* cart–pole state, not the delayed state the agent perceives. The blue arrows show the wind as a **uniform load along the pole**, drawn like a load diagram: the arrowheads touch the upwind face of the pole, and a continuous blue line joins the tails (the load envelope). The arrows point downwind, and their length is proportional to the drag force (350 px per newton, so the mean 0.09 N gives about 30 px). The text at the top left gives the current wind speed and drag.

- **Length:** the episode runs until it fails (see [Episodes, termination and reward](#episodes-termination-and-reward)) or reaches the 500-step (10 s) limit.

- **Speed:** frames play at about 33 per second, while the simulation advances 50 steps per second ($\Delta t = 0.02$ s), so the GIF plays at about 2/3 of real time.

## Project contents

- `ppo_cartpole.ipynb` — full PPO implementation, training loop, diagnostics, and animation export
- `pyproject.toml` — project dependencies
- `resources/` — generated diagnostic figures and animation artifacts.

## Environment

- Python
- Gymnasium
- PyTorch
- NumPy
- Matplotlib
- Pillow
