# PPO Cartpole with Wind Forcing and Perception Latency

This project trains a PPO (Proximal Policy Optimization) agent to solve the classic CartPole problem in Gymnasium. Gymnasium simulates the dynamics of the cart.

At each time step, the agent chooses one of two actions:
- apply a constant force to the left
- apply a constant force to the right.

On top of standard CartPole, this project adds two complications, each implemented as a Gymnasium wrapper in the notebook:

1. **External forcing**. Wind blows on the pole and the cart: a steady mean wind plus random turbulent gusts, which produce aerodynamic drag on both. The agent never observes the wind directly; it only sees its effect on the cart and pole. See [External forcing: wind](#external-forcing-wind).

2. **Perception latency**. The agent's perception lags the true state by 0.04 s (2 time steps): it sees the state $[x, \dot{x}, \theta, \dot{\theta}]$ from 2 steps ago and acts on it as if it were the current state. See [Perception latency](#perception-latency).

## Why neural networks instead of a lookup table?

Tabular RL stores $Q(s,a)$ or $V(s)$ in a lookup table and updates them with Bellman equations. This method can be employed when the set of states is finite and small.

The state $[x, \dot{x}, \theta, \dot{\theta}]$ is continuous: each variable is a real number. But time is discretized ($\Delta t = 0.02$ s) and there are only two discrete actions of the agent. A lookup table would need the state to be discretized first, and the table grows exponentially with the number of variables: with 4 variables, 20 bins each gives $20^4 = 160{,}000$ cells, and a finer 100 bins each gives $100^4 = 10^8$. Coarse bins lose the precision needed near upright, and fine bins make the table too large to fill from experience.

PPO uses neural networks, which take the real-valued state directly and generalize between nearby states, and learns from sampled trajectories instead of sweeping over all states. GAE still uses Bellman style temporal-difference errors.

## Physics model

Cartpole is a simple coupled mechanical system: a cart moves along the x-axis while a pole rotates about the pivot. The task of the agent is to keep the pole balanced even though a random lateral wind blows on the pole and the cart, and the agent perceives the state with a fixed latency (0.04 s, or 2 time steps).

Gymnasium models the pole as a uniform rod of mass $m$ and length $2l$ (its `length` parameter is $l$) on a cart of mass $M$. The angle $\theta$ is measured clockwise from upright position. With a horizontal force $F$ applied by the agent on the cart, the coupled equations of motion are:

Conservation of linear momentum in x-dir:
$$
F = (M + m)\,\ddot{x} + m l \cos\theta\,\ddot{\theta} - m l \dot{\theta}^2 \sin\theta
$$

Conservation of angular momentum around z axis:
$$
m g l \sin\theta = \tfrac{4}{3} m l^2\,\ddot{\theta} + m l \cos\theta\,\ddot{x}
$$

The first equation is $F = M\ddot{x} + m\,\ddot{x}_{cm}$, where $x_{cm} = x + l\sin\theta$ is the horizontal position of the pole's center of mass:

$$
\ddot{x}_{cm} = \ddot{x} + l\cos\theta\,\ddot{\theta} - l\sin\theta\,\dot{\theta}^2
$$

- $\ddot{x}$: the pivot moves with the cart.
- $l\cos\theta\,\ddot{\theta}$: horizontal component of the pole's tangential acceleration.
- $-l\sin\theta\,\dot{\theta}^2$: horizontal component of its centripetal acceleration. Relative to the pivot, the center of mass moves on a circle of radius $l$, so it has acceleration $l\dot{\theta}^2$ directed along the pole toward the pivot. A fast-swinging pole pulls on the pivot along its axis, and when tilted, part of that pull acts horizontally on the cart. The term is nonlinear, of order $\theta\dot{\theta}^2$ near upright, so it vanishes when the equations are linearized; the simulation keeps it.

In the second equation, the term $m l \cos\theta\,\ddot{x}$ reflects the inertial torque from the accelaration of the pivot that connects the pole to the cart, since it accelerates with the cart. 

$\tfrac{4}{3} m l^2$ is the rod's moment of inertia about the pivot. The $\cos\theta$ terms couple the two: accelerating the cart tips the pole, and the swinging pole pushes back on the cart. Gravity ($+m g l \sin\theta$) makes the upright position unstable.

These are Gymnasium's equations, without wind. The wind adds external forces on both bodies: the drag on the cart $F_{cart}$ and the drag on the pole $F_{pole}$ (acting at mid-pole). With them the equations become

$$
F + F_{cart} + F_{pole} = (M + m)\,\ddot{x} + m l \cos\theta\,\ddot{\theta} - m l \dot{\theta}^2 \sin\theta
$$

$$
m g l \sin\theta + F_{pole}\, l \cos\theta = \tfrac{4}{3} m l^2\,\ddot{\theta} + m l \cos\theta\,\ddot{x}
$$

In matrix form, with the mass matrix $\mathbf{M}(\theta)$:

$$
\underbrace{\begin{bmatrix} M+m & m l\cos\theta \\ m l\cos\theta & \tfrac{4}{3} m l^2 \end{bmatrix}}_{\mathbf{M}(\theta)}
\begin{bmatrix} \ddot{x} \\ \ddot{\theta} \end{bmatrix}
=
\begin{bmatrix} F + F_{cart} + F_{pole} + m l\dot{\theta}^2\sin\theta \\ m g l\sin\theta + F_{pole}\, l\cos\theta \end{bmatrix}
$$

The diagonal entries are the inertia of each coordinate (total mass, and the pole's moment of inertia about the pivot); the off-diagonal $m l\cos\theta$ couples cart and pole. $\mathbf{M}(\theta)$ is symmetric and positive definite, so the accelerations can always be solved for.

See [External forcing: wind](#external-forcing-wind) for how the wind forces are computed and applied.

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

An episode is one attempt to balance the pole, starting near upright and ending when it fails or after 500 steps (10 s). Each of the state variables $x, \dot{x}, \theta, \dot{\theta}$ starts uniformly random in $[-0.05, 0.05]$.

- **Failure (termination):** the episode ends as soon as the pole tilts more than 12° from upright ($|\theta| > 0.2094$ rad) or the cart leaves the track ($|x| > 2.4$ m).
- **Time limit (truncation):** otherwise the episode is cut off after 500 steps, that is 10 secs.
- **Reward:** +1 for every step the pole stays up, so an episode's total reward is the number of steps it survived. The maximum is 500.

These rules are part of the environment, so they apply every time the policy runs: during training (a failed episode triggers a reset and a new attempt) and when rendering the GIF (the recording stops at the end of the episode).

**Why failure drives learning.** The reward is the same +1 on every step, so the only thing that distinguishes good behavior from bad is *when the episode ends prematurely*. Failing early means fewer discounted future rewards.  The 12° and 2.4 m limits are what turn balancing the pole into a learnable objective. 

In training, the two kinds of episode endings have different effects on learning a policy:

- **Failure:** Angle exceeds 12 degrees or the cart moves outside the 2.4 m section, the future reward is zero. Actions that led toward failure get lower advantages, and the policy learns to avoid them.

- **Time limit:** the episode was cut off, not failed. The pole could have stayed up, so the critic's estimate of future reward $V(s)$ is used in place of the missing future (bootstrapping). Treating the 500-step cut-off as a failure would wrongly teach the agent that surviving to the end is bad.

## External forcing: wind

Wind blows horizontally on the pole and the cart, modeled the way wind loads are modeled in engineering: a mean wind plus turbulent gusts, converted to forces by aerodynamic drag.

The wind speed is a steady mean plus a random fluctuation:

$$
u(t) = U + u'(t)
$$

The gust $u'(t)$ follows the **von Kármán turbulence spectrum**, the standard model for atmospheric turbulence:

$$
S_u(f) = \sigma_u^2\,\frac{4\,T_L}{\big(1 + 70.8\,(f\,T_L)^2\big)^{5/6}}, \qquad T_L = \frac{L}{U}
$$

where $\sigma_u$ is the gust intensity and $T_L$ the integral time scale (roughly, how long a gust lasts). At high frequency the spectrum falls off as $f^{-5/3}$, Kolmogorov's inertial-range law. $u'(t)$ is synthesized as a sum of 400 Fourier modes between 0.02 and 20 Hz, with amplitudes $\sqrt{2\,S_u(f)\,\Delta f}$ and random phases. The phases are drawn fresh at the start of every episode, so every episode has different gusts with the same statistics.

**Drag force.** The same wind acts on both bodies. Each experiences quadratic drag

$$
F(t) = \tfrac{1}{2}\,\rho\,C_d\,A\,|u(t)|\,u(t)
$$

with its own drag coefficient $C_d$ and frontal area $A$ (the area facing the wind):

- **Pole:** a circular cylinder, $A$ = pole diameter × pole length. The load is spread uniformly along the pole, so its resultant $F_{pole}$ acts at **mid-pole**, a height $l$ above the pivot (at $x + l\sin\theta$).
- **Cart:** a cube whose side is 1/5 of the pole length (0.2 m), so $A$ = 0.2 m × 0.2 m. Its drag $F_{cart}$ acts on the cart itself.

| Parameter | Symbol | Value |
|---|---|---|
| Mean wind speed | $U$ | 2.5 m/s (a light breeze) |
| Turbulence intensity | $\sigma_u / U$ | 30% ($\sigma_u$ = 0.75 m/s) |
| Turbulence length scale | $L$ | 5 m ($T_L$ = 2 s) |
| Air density | $\rho$ | 1.2 kg/m³ |
| Pole: drag coefficient (cylinder) | $C_d$ | 1.2 |
| Pole: diameter | $D$ | 2 cm (frontal area 0.02 m²) |
| Cart: drag coefficient (cube, face-on) | $C_d$ | 1.05 |
| Cart: side length | | 0.2 m (frontal area 0.04 m²) |
| Resulting drag on the pole | $F_{pole}$ | about 0.09 N mean, 0.01–0.22 N over an episode |
| Resulting drag on the cart | $F_{cart}$ | about 0.16 N mean, up to 0.39 N over an episode |

The cart has twice the pole's frontal area, so it catches more wind than the pole. The same wind model is used in training and in the GIF.

![Wind speed and drag forces on the pole and cart over one 10 s episode](resources/external_forcing.png)

The figure shows one 10 s episode (seeded for reproducibility). Top: the wind speed, with slow gusts lasting a few seconds and small fast fluctuations on top, as in real wind. Bottom: the resulting drag force on the cart and on the pole. The force on the cart is larger because of its larger crossectional area. Since the wind force is proportional to $u^2$, gusts are amplified. The wind varies by about ±30%, but the forces vary by more than a factor of ten.

**Applying the forces.** Over one time step $\Delta t$, the wind force is applied as an impulse $J = F\,\Delta t$, which changes $\dot{x}$ and $\dot{\theta}$ through the mass matrix of the equations of motion: $\big[\Delta\dot{x},\ \Delta\dot{\theta}\big]^T = \mathbf{M}(\theta)^{-1}\,\big[J_x,\ J_\theta\big]^T$. The generalized impulse depends on where the force acts:

- **Pole** (at height $l$): $(J_{pole},\; l\cos\theta\,J_{pole})$. It pushes the system and tips the pole directly.
- **Cart** (at the cart): $(J_{cart},\; 0)$. It exerts no torque on the pole directly, but accelerating the cart tips the pole through the $\cos\theta$ coupling, just like the agent's own push.

Both impulses are applied just before Gymnasium's Euler step.

**Why the wind is limited to a light breeze.** Real wind has a nonzero mean, so to stay balanced the pole must lean *into* the wind. With the pole's drag and weight both acting at mid-pole, and the cart held stationary on average, the steady equilibrium lean is

$$
\sin\theta_{eq} = \frac{F}{m\,g}
$$

where $F$ is the drag on the pole; the cart's drag does not change the lean. The pole weighs only 0.1 kg ($m g$ = 0.98 N), so it is very sensitive to wind. At 2.5 m/s the mean drag of 0.09 N gives a lean of about 5°. At 4 m/s the lean would be about 14°, beyond the 12° failure limit: no controller could keep the pole up. The largest tolerable mean drag is $m g \sin 12° \approx 0.20$ N.

**What the agent has to do.** The per-step kick from the wind is small compared with the agent's own push:

| Force | $\Delta\dot{\theta}$ per step |
|---|---|
| Agent's 10 N push on the cart | 0.29 rad/s |
| Mean pole drag, 0.09 N at mid-pole | 0.03 rad/s |
| Peak pole drag, 0.22 N at mid-pole | 0.06 rad/s |
| Mean cart drag, 0.16 N on the cart | 0.005 rad/s (tips the pole the other way) |

The difficulty is that the wind pushes **persistently in one direction**. Together the pole and cart catch about 0.25 N of mean drag, so the whole system is blown downwind. The agent has to hold the pole tilted into the wind and push back on average to keep the cart from drifting off the track, all while the gusts change the required lean every few seconds. Without that correction, even a good balancing controller is blown off the end of the track within a few seconds. The cart's drag adds mostly to this drift, which makes keeping the cart on the track harder.

## Perception latency

The agent perceives the state $\tau = 0.04$ s (2 time steps) late. At step $n$ its only input is the state from 2 steps earlier,

$$
o_n = s_{n-2} = [x, \dot{x}, \theta, \dot{\theta}]_{n-2}
$$

and it acts on it as if it were the current state. This models the time it takes to sense the cart and pole and process what it sees before acting. 

| Parameter | Value |
|---|---|
| Perception latency, $\tau$ | 0.04 s (2 steps) |
| Observation | $[x, \dot{x}, \theta, \dot{\theta}]$ at $t - \tau$ (4 numbers) |

**What the delay costs.** Acting on a 40 ms-old state is not optimal: during those 2 steps the agent's own pushes and the wind have already moved the cart and pole.

**Efference copy.** The brain faces the same problem: sensory feedback arrives 100–200 ms late. It compensates with an *efference copy*, an internal copy of each motor command sent to the brain regions that predict movement. Combining the delayed sensory signal with the commands issued since, it estimates the body's current state before the feedback arrives (a forward model). Autonomous vehicles do the same, under the name latency compensation.

The PPO agent here has no efference copy: it acts on the delayed state alone. An earlier version gave it one, in the form of its two most recent actions (plus one extra delayed frame), so it could extrapolate to the present. With the same wind, that version kept the pole up for the full 500 steps in 18 of 20 evaluation episodes, against 11 of 20 with the pure delay.

## Initial conditions

- **Cart and pole:** each of $x, \dot{x}, \theta, \dot{\theta}$ is drawn uniformly from $[-0.05, 0.05]$ (m, m/s, rad, rad/s), so the pole starts within about ±2.9° of upright and nearly at rest.

- **Wind:** every episode starts at $t = 0$ with new random gust phases, so each episode has a different gust history with the same statistics.

- **Perception buffer:** at reset there is no history yet, so for the first 2 steps the agent perceives the initial state.

- **Random seeds:** PyTorch and NumPy are seeded with 42 at the top of the notebook, and the training environment with 42 at its first reset, so the whole notebook is reproducible: re-running it gives the same trained policy, figures and GIF. The evaluation episodes and the GIF use fixed seeds 0–19.

## PPO agent

In this project:

- the actor network outputs action probabilities from the observed state,
- the critic network estimates the expected return from that state,
- PPO updates the policy using a clipped objective to keep learning stable.

This is the key idea behind modern reinforcement learning: use function approximation to generalize across a huge or continuous state space.

Derivations of the policy gradient, the clipped objective and GAE: [PPO notes](PPO_NOTES.md).

## Trained agent animation

![Trained agent balancing the pole in turbulent wind](resources/cartpole_trained_rollout.gif)

The GIF shows one episode of inference. 

- **Deterministic policy:** at each step the agent takes its most likely action instead of sampling one as in training: $a = \arg\max_a \pi(a \mid s)$.

- **Same conditions as training:** the same wind model and the same 0.04 s perception latency.

- **Which episode:** the notebook first runs 20 evaluation episodes with fixed seeds 0–19 and prints the length of each. The GIF replays the longest one. With the current seeds, 11 of the 20 episodes last the full 500 steps; the other nine fail between 113 and 488 steps (median over all 20: 500). So the GIF shows a successful episode, which is slightly more common than not, but far from guaranteed.

- **What is drawn:** Gymnasium renders the *true* cart–pole state, not the delayed state the agent perceives. The blue arrows show the wind as a **uniform load along the pole**, drawn like a load diagram: the arrowheads touch the upwind face of the pole, and a continuous blue line joins the tails (the load envelope). The arrows point downwind, and their length is proportional to the drag force (350 px per newton, so the mean 0.09 N gives about 30 px). The blue arrows along the full height of the cart's upwind face show the wind on the cart, at the same scale. The red arrow just below the cart is the agent's push: it points in the direction of the push, toward the side of the cart being pushed. The push is always 10 N, so this arrow has a fixed length and only its direction changes. A legend at the top left identifies the red and blue arrows.

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
- pygame (used by Gymnasium to render the animation)
