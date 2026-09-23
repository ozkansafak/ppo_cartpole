# PPO Cartpole

This project trains a PPO (Proximal Policy Optimization) agent to solve the classic CartPole problem in Gymnasium. Gymnasium simulates the cart's horizontal motion and the pole's rotational dynamics.

At each time step, the agent chooses one of two actions:
- apply a force to the left
- apply a force to the right.
Gymnasium uses that force to compute the physics of the system. The custom disturbance wrapper applies an additional horizontal force at the pole tip through the true dynamics before each step and delays the observation returned to the policy. Because the policy only sees the past, each observation contains the last few delayed frames plus the actions taken since the newest one, so the agent can infer where the pole is now.

## Why neural networks instead of a lookup table?

Tabular RL stores $Q(s,a)$ or $V(s)$ in a lookup table and updates them with Bellman equations. Cartpole has a continuous state, $[x, \dot{x}, \theta, \dot{\theta}]$, so PPO uses neural networks and sampled trajectories instead. GAE still uses Bellman-style temporal-difference errors.

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

## Disturbance

The wrapper adds a horizontal force $F_{tip}$ at the pole tip, at $x + 2l\sin\theta$. It is the sum of a steady oscillating push and occasional random "slaps". Over one time step $\Delta t$ it is applied as an impulse $J = F_{tip}\,\Delta t$ with generalized components $(J,\; 2 l \cos\theta\,J)$, which changes $\dot{x}$ and $\dot{\theta}$ through the mass matrix of the equations above. The impulse is applied just before Gymnasium's Euler step.

## PPO agent

In this project:

- the actor network outputs action probabilities from the observed state,
- the critic network estimates the expected return from that state,
- PPO updates the policy using a clipped objective to keep learning stable.

This is the key idea behind modern reinforcement learning: use function approximation to generalize across a huge or continuous state space.

## Trained agent animation

![Trained agent balancing the pole under tip-force disturbances](resources/cartpole_trained_rollout.gif)

The GIF shows one episode of **inference** with the trained policy. It is recorded after training finishes, and nothing is learned while it runs.

- **Deterministic policy:** at each step the agent takes its most likely action instead of sampling one as in training.
- **Harder than training:** the disturbance is stronger than what the agent trained on, so the GIF tests how well the policy copes with pushes it hasn't seen:

  | | Training | GIF |
  |---|---|---|
  | Oscillating push amplitude | 0.2 N | 0.35 N |
  | Oscillating push frequency (rad/step) | 0.35 | 0.25 |
  | Slap probability per step | 4% | 6% |
  | Max slap | ±1.0 N | ±1.5 N |

  The observation delay (2 steps) and stacking (2 frames) are the same as in training.
- **What is drawn:** Gymnasium renders the *true* cart–pole state, not the delayed observation the agent sees. The red arrow at the pole tip shows the **direction** of the applied tip force. Its length is a fixed display size and does not show how strong the force is.
- **Length:** one episode from a random start state, recorded until it ends (the pole falls or the cart leaves the track) or reaches the 500-step limit. Each run of the notebook produces a different episode.
- **Speed:** frames are saved at 30 fps, while the simulation advances 50 steps per second ($\Delta t = 0.02$ s), so the GIF plays at 0.6× real time.

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
