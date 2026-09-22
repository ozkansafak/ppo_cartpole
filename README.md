# PPO CartPole

This project trains a Proximal Policy Optimization (PPO) agent to solve the classic CartPole environment from Gymnasium.

## Why not a lookup table?

CartPole has a continuous state space.

- State is 4 real numbers: $[x, \dot{x}, \theta, \dot{\theta}]$
- Each variable can take infinitely many values.
- That means the number of possible states is effectively uncountable.
- A table-based method would require one entry for every possible combination of these values, which is impossible.

So instead of storing Q-values in a table, we train a neural network to approximate the policy and value function.

In this project:

- the actor network outputs action probabilities from the observed state,
- the critic network estimates the expected return from that state,
- PPO updates the policy using a clipped objective to keep learning stable.

This is the key idea behind modern reinforcement learning: use function approximation to generalize across a huge or continuous state space.

## Project contents

- `ppo_cartpole.ipynb` — full PPO implementation, training loop, diagnostics, and animation export
- `pyproject.toml` — project dependencies
- generated figures and animation artifacts in the project root

## Run the notebook

Open the notebook in Jupyter and run the cells in order.

## Environment

- Python
- Gymnasium
- PyTorch
- NumPy
- Matplotlib
- Pillow
