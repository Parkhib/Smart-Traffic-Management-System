import sys, random, collections, os
import numpy as np
import matplotlib.pyplot as plt

try:
    import gym
    from gym import spaces
except Exception:
    !pip -q install gym==0.26.2
    import gym
    from gym import spaces

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    !pip -q install torch --extra-index-url https://download.pytorch.org/whl/cpu
    import torch
    import torch.nn as nn
    import torch.optim as optim

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrafficEnv(gym.Env):
    """Single intersection: actions switch NS/EW green. State = queues [N,S,E,W]."""
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_queue: int = 30,
        max_steps: int = 120,
        arrival_low: int = 0,
        arrival_high: int = 3,
        discharge_low: int = 1,
        discharge_high: int = 5,
        switch_penalty: float = 0.0,  
    ):
        super().__init__()
        self.max_queue = max_queue
        self.max_steps = max_steps
        self.arrival_low = arrival_low
        self.arrival_high = arrival_high
        self.discharge_low = discharge_low
        self.discharge_high = discharge_high
        self.switch_penalty = switch_penalty

        self.action_space = spaces.Discrete(2)  # 0 = NS green, 1 = EW green
        self.observation_space = spaces.Box(
            low=0, high=self.max_queue, shape=(4,), dtype=np.int32
        )

        self.state = None
        self.steps = 0
        self.prev_action = None

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.state = np.random.randint(0, 10, size=(4,), dtype=np.int32)
        self.steps = 0
        self.prev_action = None
        return self.state, {}

    def step(self, action: int):
        n, s, e, w = self.state.astype(np.int32)

        
        arr = lambda: np.random.randint(self.arrival_low, self.arrival_high + 1)
        arr_n, arr_s, arr_e, arr_w = arr(), arr(), arr(), arr()

        
        if action == 0:  
            n = max(0, n - np.random.randint(self.discharge_low, self.discharge_high))
            s = max(0, s - np.random.randint(self.discharge_low, self.discharge_high))
            e = min(self.max_queue, e + arr_e)
            w = min(self.max_queue, w + arr_w)
            
            n = min(self.max_queue, n + arr_n // 3)
            s = min(self.max_queue, s + arr_s // 3)
        else:
            e = max(0, e - np.random.randint(self.discharge_low, self.discharge_high))
            w = max(0, w - np.random.randint(self.discharge_low, self.discharge_high))
            n = min(self.max_queue, n + arr_n)
            s = min(self.max_queue, s + arr_s)
            e = min(self.max_queue, e + arr_e // 3)
            w = min(self.max_queue, w + arr_w // 3)

        self.state = np.array([n, s, e, w], dtype=np.int32)

       
        reward = -float(self.state.sum())


        if self.prev_action is not None and action != self.prev_action:
            reward -= self.switch_penalty
        self.prev_action = action

        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps
        info = {"total_queue": int(self.state.sum())}
        return self.state, reward, terminated, truncated, info

    def render(self):
        print(f"Step {self.steps} | Queues [N,S,E,W] = {self.state.tolist()}")

def run_fixed_timer(env, total_episodes=20, switch_every=5):
    rewards, avg_queues = [], []
    for _ in range(total_episodes):
        obs, _ = env.reset()
        ep_reward, queues, t = 0.0, [], 0
        while True:
            action = 0 if (t // switch_every) % 2 == 0 else 1
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            queues.append(info["total_queue"])
            t += 1
            if term or trunc:
                break
        rewards.append(ep_reward)
        avg_queues.append(np.mean(queues))
    return float(np.mean(rewards)), float(np.mean(avg_queues))

def discretize_state(state, bin_size=3):
    return tuple((state // bin_size).tolist())

def q_learning(
    env,
    episodes=300,
    alpha=0.1,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    bin_size=3,
):
    from collections import defaultdict
    Q = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float32))
    eps = epsilon_start
    rewards_hist, avg_queue_hist = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        total_r, queues = 0.0, []
        while True:
            s = discretize_state(obs, bin_size)
            a = env.action_space.sample() if np.random.rand() < eps else int(np.argmax(Q[s]))
            next_obs, r, term, trunc, info = env.step(a)
            s2 = discretize_state(next_obs, bin_size)
            td_target = r + gamma * np.max(Q[s2])
            Q[s][a] += alpha * (td_target - Q[s][a])
            obs = next_obs
            total_r += r
            queues.append(info["total_queue"])
            if term or trunc:
                break
        eps = max(epsilon_end, eps * epsilon_decay)
        rewards_hist.append(total_r)
        avg_queue_hist.append(np.mean(queues))
    return Q, rewards_hist, avg_queue_hist


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (
            torch.tensor(s, dtype=torch.float32, device=device),
            torch.tensor(a, dtype=torch.int64, device=device),
            torch.tensor(r, dtype=torch.float32, device=device),
            torch.tensor(s2, dtype=torch.float32, device=device),
            torch.tensor(d, dtype=torch.float32, device=device),
        )

    def __len__(self): return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )
    def forward(self, x): return self.net(x)

def soft_update(target, source, tau=0.05):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

def select_action(policy_net, state, epsilon, action_space):
    if random.random() < epsilon:
        return action_space.sample()
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = policy_net(s)
        return int(torch.argmax(q, dim=1).item())


EPISODES = 600        
STEPS_PER_EP = 120
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.997    
TARGET_UPDATE_EVERY = 100
REPLAY_WARMUP = 1000


def train_dqn():
    env = TrafficEnv(max_steps=STEPS_PER_EP, switch_penalty=0.5)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.n
    policy_net, target_net = DQN(obs_dim, act_dim).to(device), DQN(obs_dim, act_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer(capacity=100_000)

    epsilon, global_step = EPS_START, 0
    returns, avg_queues = [], []

    for ep in range(EPISODES):
        obs, _ = env.reset()
        ep_return, q_episode = 0.0, []
        for t in range(STEPS_PER_EP):
            a = select_action(policy_net, obs, epsilon, env.action_space)
            next_obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            replay.push(obs.astype(np.float32), a, r, next_obs.astype(np.float32), float(done))
            obs = next_obs
            ep_return += r
            q_episode.append(info["total_queue"])
            global_step += 1
            epsilon = max(EPS_END, epsilon * EPS_DECAY)

            if len(replay) >= REPLAY_WARMUP:
                s, a_b, r_b, s2, d_b = replay.sample(BATCH_SIZE)
                q_vals = policy_net(s).gather(1, a_b.view(-1,1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(s2).max(1)[0]
                    target = r_b + GAMMA * (1.0 - d_b) * next_q
                loss = nn.MSELoss()(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
                optimizer.step()

                if global_step % TARGET_UPDATE_EVERY == 0:
                    soft_update(target_net, policy_net, tau=0.05)
            if done:
                break
        returns.append(ep_return)
        avg_queues.append(float(np.mean(q_episode)))
        if (ep + 1) % 50 == 0:
            print(f"Ep {ep+1}/{EPISODES} | Return {ep_return:.1f} | AvgQ {avg_queues[-1]:.2f} | eps {epsilon:.3f}")
    return env, policy_net, target_net, returns, avg_queues


def plot_training(returns, avg_queues):
    plt.figure()
    plt.plot(returns)
    plt.xlabel("Episode"); plt.ylabel("Return"); plt.title("DQN Training: Episode Return")
    plt.show()

    plt.figure()
    plt.plot(avg_queues)
    plt.xlabel("Episode"); plt.ylabel("Average Queue Length"); plt.title("DQN Training: Avg Queue Length")
    plt.show()


def evaluate_policy(env, policy_fn, episodes=30):
    rets, queues = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        total_r, q_list, t = 0.0, [], 0
        while True:
            a = policy_fn(obs, t)
            obs, r, term, trunc, info = env.step(a)
            total_r += r
            q_list.append(info["total_queue"])
            t += 1
            if term or trunc:
                break
        rets.append(total_r)
        queues.append(np.mean(q_list))
    return float(np.mean(rets)), float(np.mean(queues))

def policy_fixed_timer(k=5):
    def _pol(obs, t): return 0 if (t // k) % 2 == 0 else 1
    return _pol

def policy_dqn(net):
    def _pol(obs, t):
        with torch.no_grad():
            s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            q = net(s); return int(torch.argmax(q, dim=1).item())
    return _pol

def policy_tabular(Q, bin_size=3):
    def _pol(obs, t):
        s = tuple((obs // bin_size).tolist())
        if s in Q: return int(np.argmax(Q[s]))
        return 0
    return _pol


def save_model(model, path="results/dqn_traffic.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print("Saved ->", path)

def load_model(path, input_dim, output_dim):
    model = DQN(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print("Loaded model from", path)
    return model


if __name__ == "__main__":
    
    fixed_ret, fixed_avgq = run_fixed_timer(TrafficEnv(), total_episodes=10, switch_every=5)
    print(f"[Baseline] Fixed Timer -> Return: {fixed_ret:.1f} | AvgQueue: {fixed_avgq:.2f}")

   
    RUN_Q_LEARNING = True
    if RUN_Q_LEARNING:
        Q_table, ql_rewards, ql_avgq = q_learning(TrafficEnv(), episodes=200)
        print(f"[Tabular QL] Last-20 mean Return: {np.mean(ql_rewards[-20:]):.1f} | AvgQueue: {np.mean(ql_avgq[-20:]):.2f}")
    else:
        Q_table = None

   
    env, policy_net, target_net, returns, avg_queues = train_dqn()
    plot_training(returns, avg_queues)

   
    eval_env = TrafficEnv(max_steps=120)
    ret_fixed, q_fixed = evaluate_policy(eval_env, policy_fixed_timer(k=5))
    ret_dqn, q_dqn = evaluate_policy(eval_env, policy_dqn(policy_net))

    print(f"[Eval] Fixed Timer -> Return: {ret_fixed:.1f} | AvgQueue: {q_fixed:.2f}")
    print(f"[Eval] DQN         -> Return: {ret_dqn:.1f} | AvgQueue: {q_dqn:.2f}")

    if Q_table is not None:
        ret_ql, q_ql = evaluate_policy(eval_env, policy_tabular(Q_table))
        print(f"[Eval] Q-Learning  -> Return: {ret_ql:.1f} | AvgQueue: {q_ql:.2f}")

   
    save_model(policy_net, "results/dqn_traffic.pt")
    reloaded = load_model("results/dqn_traffic.pt", eval_env.observation_space.shape[0], eval_env.action_space.n)
   k
    r_loaded, q_loaded = evaluate_policy(eval_env, policy_dqn(reloaded), episodes=10)
    print(f"[Eval] Reloaded DQN -> Return: {r_loaded:.1f} | AvgQueue: {q_loaded:.2f}")

    print("\n✅ Done. Tweak EPISODES/STEPS_PER_EP for better performance. Consider higher switch_penalty to discourage flips.")
