# PPO Cartpole

This project trains a PPO (Proximal Policy Optimization) agent to solve the classic CartPole problem in Gymnasium. Gymnasium simulates the dynamics of the cart.

At each time step, the agent chooses one of two actions:
- apply a constant force to the left
- apply a constant force to the right.

On top of standard CartPole, this project adds two complications, each implemented as a Gymnasium wrapper in the notebook:

1. **External forcing** (`ExternalForcing`). A horizontal force acts on the top of the pole: a harmonic force plus random impulses. The agent never observes the forcing directly, but only responds to the changes in the velocities of the cart and the pole. See [External forcing](#external-forcing).
2. **Perception latency** (`PerceptionLatency`). The agent's perception lags the true state by 0.04 s (2 time steps). To compensate, each observation contains the two most recent delayed states plus the two actions the agent has taken since, from which it can infer the current state. See [Perception latency](#perception-latency).

## Why neural networks instead of a lookup table?

Tabular RL stores $Q(s,a)$ or $V(s)$ in a lookup table and updates them with Bellman equations. That requires a finite, reasonably small set of states.

In this project, **time** is discrete (steps of $\Delta t = 0.02$ s) and the **actions** are discrete (two choices), but the **state** $[x, \dot{x}, \theta, \dot{\theta}]$ is continuous: each variable is a real number and is never binned. A lookup table would need the state discretized first, and the table grows exponentially with the number of variables. With the perception latency, each observation has 10 numbers, so even a coarse 20 bins per variable would give $20^{10} \approx 10^{13}$ cells.

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

## External forcing

A horizontal force $F_{tip}(t)$ acts at the pole tip, at $x + 2l\sin\theta$. It has a deterministic part and a random part:

$$
F_{tip}(t) = A \sin(\omega t) + F_{imp}(t)
$$

- **Harmonic forcing** $A\sin(\omega t)$: a smooth, periodic back-and-forth push. It is deterministic, and its phase restarts at $t = 0$ at the start of every episode.
- **Random impulses** $F_{imp}(t)$: in each time step, with probability $r\,\Delta t$, a force drawn uniformly from $[-F_{max}, F_{max}]$ acts for that one step (0.02 s). Otherwise $F_{imp} = 0$. On average that gives $r$ impulses per second, at random times, with random sign and size.

| Parameter | Symbol | Value |
|---|---|---|
| Harmonic amplitude | $A$ | 0.2 N |
| Angular frequency | $\omega$ | 17.5 rad/s (2.79 Hz, period 0.36 s) |
| Mean impulse rate | $r$ | 2 per second (probability 0.04 per step) |
| Max impulse force | $F_{max}$ | 1.0 N (held for one step) |

The same forcing is used in training and in the GIF.

![External tip force over one 10 s episode: a 0.2 N harmonic push with random impulses of up to ±1 N on top](resources/external_forcing.png)

The figure shows one 10 s episode of the tip force. The blue curve is the harmonic part; the orange line is the total force actually applied, with a dot at each random impulse (20 in this episode, matching the expected $r \times 10\,\text{s} = 20$). The zoomed lower panel shows that the force is piecewise constant: each value is held for one 0.02 s step. The impulses are seeded, so the figure is reproducible; in training they are drawn fresh every episode.

Over one time step $\Delta t$, $F_{tip}$ is applied as an impulse $J = F_{tip}\,\Delta t$ with generalized components $(J,\; 2 l \cos\theta\,J)$, which changes $\dot{x}$ and $\dot{\theta}$ through the mass matrix of the equations above. The impulse is applied just before Gymnasium's Euler step.

For scale, the change in $\dot{\theta}$ from one step of force:

| Force | $\Delta\dot{\theta}$ per step |
|---|---|
| Agent's 10 N push on the cart | 0.29 rad/s |
| 0.2 N at the tip (harmonic peak) | 0.12 rad/s |
| 1.0 N at the tip (largest impulse) | 0.62 rad/s |

A force at the tip is very effective: the pole is light (0.1 kg) and the tip has the longest lever arm, so the largest impulse moves the pole about twice as much as the agent's own push.

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
- **Forcing:** the harmonic term starts at phase 0 ($t = 0$) every episode; the random impulses start fresh.
- **Perception buffer:** at reset there is no history yet, so all perceived frames are filled with the initial state and the recent actions with 0.
- **Random seeds:** PyTorch and NumPy are seeded with 42 at the top of the notebook. Training episodes are not individually seeded, so training results vary slightly from run to run. The GIF and the evaluation episodes use fixed seeds 0–19 and are reproducible for a given trained policy.

## PPO agent

In this project:

- the actor network outputs action probabilities from the observed state,
- the critic network estimates the expected return from that state,
- PPO updates the policy using a clipped objective to keep learning stable.

This is the key idea behind modern reinforcement learning: use function approximation to generalize across a huge or continuous state space.

## Trained agent animation

![Trained agent balancing the pole under external tip forcing](resources/cartpole_trained_rollout.gif)

The GIF shows one episode of inference with the trained policy. 

- **Deterministic policy:** at each step the agent takes its most likely action instead of sampling one as in training: $a = \arg\max_a \pi(a \mid s)$.
- **Same conditions as training:** the same external forcing and the same 0.04 s perception latency.
- **Which episode:** the notebook first runs 20 evaluation episodes with fixed seeds 0–19 and prints the length of each. The GIF replays the longest one. In the current run, 18 of the 20 episodes lasted the full 500 steps (the other two failed at 274 and 424 steps), so the GIF shows typical behavior, not a lucky exception.
- **What is drawn:** Gymnasium renders the *true* cart–pole state, not the delayed state the agent perceives. The red arrow at the pole tip shows the **direction** of the tip force. Its length is a fixed display size and does not show how strong the force is.
- **Length:** the episode runs until it fails (see [Episodes, termination and reward](#episodes-termination-and-reward)) or reaches the 500-step (10 s) limit.
- **Speed:** frames play at about 33 per second, while the simulation advances 50 steps per second ($\Delta t = 0.02$ s), so the GIF plays at about 2/3 of real time.

## Project contents

- `ppo_cartpole.ipynb` — full PPO implementation, training loop, diagnostics, and animation export
- `pyproject.toml` — project dependencies
- `resources/` — generated diagnostic figures and animation artifacts

## Run the notebook

Open the notebook in Jupyter and run the cells in order.

## Environment

- Python
- Gymnasium
- PyTorch
- NumPy
- Matplotlib
- Pillow
