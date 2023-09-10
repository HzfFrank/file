import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN


class LCPEnv(gym.Env):
    # Define constants for clearer code
    LEFT = 0
    RIGHT = 1

    def __init__(self, dim: int, A: np.ndarray, b: np.ndarray):
        super(LCPEnv, self).__init__()

        # Dimension of LCP
        self.dim = dim

        assert A.shape == (dim, dim)
        assert b.shape == (dim,)

        # Parameters of LCP
        param = np.hstack((A, b)).astype(np.float32)
        self.param = param / np.max(np.abs(param))

        # Define action and observation space
        self.action_space = spaces.Discrete(dim)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(dim, dim + 1, 3), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # Initialize the classification
        row_label = np.ones_like(self.param)
        col_label = np.ones_like(self.param)
        col_label[:, -1] = 0.0

        self.state = np.stack((self.param, row_label, col_label), axis=2)

        return self.state, {}  # empty info dict

    def solve(self):
        key = self.state[:, 0, 1]
        clamp = np.count_nonzero(key == -1.0)
        order = np.argsort(key)

        A = self.state[:, :-1, 0][order, :][:, order]
        b = self.state[:, -1, 0][order]

        A_cc = A[:clamp, :clamp]
        b_c = b[:clamp]
        f_c = np.linalg.solve(A_cc, -b_c)

        A_nc = A[clamp:, :clamp]
        b_n = b[clamp:]
        a_n = A_nc @ f_c + b_n

        return f_c, a_n

    def step(self, action):
        self.state[action, :, 1] *= -1
        self.state[:, action, 2] *= -1

        f_c, a_n = self.solve()
        f_err = np.clip(f_c, None, 0.0)
        a_err = np.clip(a_n, None, 0.0)

        error = np.linalg.norm(f_err) ** 2 + np.linalg.norm(a_err) ** 2
        reward = -error

        terminated = bool(error < 1e-6)
        truncated = False  # we do not limit the number of steps here

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            self.state,
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        pass


# Instantiate the env
env = LCPEnv()
# Define and Train the agent
# model = DQN("MlpPolicy", env, verbose=1, )
model = DQN(
    "MlpPolicy",
    env=env,
    learning_rate=5e-4,
    batch_size=128,
    buffer_size=50000,
    learning_starts=0,
    target_update_interval=250,
    policy_kwargs={"net_arch": [256, 256]},
    verbose=1,
    tensorboard_log="./tensorboard/LCP/"
)
model.learn(total_timesteps=100000, log_interval=4)
model.save("LCP")
