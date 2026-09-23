# PPO CartPole

This project trains a PPO (Proximal Policy Optimization) agent to solve the classic CartPole problem inside the environment from Gymnasium.

## Why not a lookup table?

CartPole has a continuous state space.

- State is 4 real numbers: $[x, \dot{x}, \theta, \dot{\theta}]$
- These are continuous values, hence the possible states are uncountable.
- A table-based method would require one entry for every possible combination of these values.
- So instead of storing Q-values in a table, we train a neural network to approximate the policy and value function.

## Physics model

CartPole is a coupled mechanical system: a cart moves along the x-axis while a pole rotates about the pivot.

The cart is driven by a horizontal force $F$, so its translational motion follows Newton’s second law:

$$
F = m_{cart} \ddot{x}
$$

The pole has rotational inertia, so its angular motion satisfies:

$$
I \ddot{\theta} = -m g l \sin(\theta) + \text{(coupling from cart motion)}
$$

with $I \approx m l^2$ for a pole of length $l$ and mass $m$. In other words, the pole resists changes in its angular motion because its mass is distributed away from the pivot.

Gymnasium simulates this by stepping the dynamics numerically each time the agent acts. At every step, the environment applies the control force, integrates the cart–pole system forward in time, and returns the next state $[x, \dot{x}, \theta, \dot{\theta}]$ and the reward.

In this project:

- the actor network outputs action probabilities from the observed state,
- the critic network estimates the expected return from that state,
- PPO updates the policy using a clipped objective to keep learning stable.

This is the key idea behind modern reinforcement learning: use function approximation to generalize across a huge or continuous state space.

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
