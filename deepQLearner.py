from constants import *
from MineSweeperEnv import MinesweeperEnv
from qkNetwork import QNetwork, ReplayBuffer
import random
import tensorflow as tf
import numpy as np
import os
import csv
import time

###############################################################
#  SECTION 3: Deep Q-Learning Loop
# - Does all the DQN heavy work such as epsilon-greedy action selection
# - sampling from ReplayBuffer, Update Q-values, episode iteration
# - and logging results into .csv file
# - adding a Target Network
###############################################################

class DeepQLearner:
    def __init__(self,
                 rows=NUM_ROWS,
                 cols=NUM_COLS,
                 num_mines=NUM_MINES,
                 sub_state_size=3,
                 gamma=0.99,
                 lr=1e-3,
                 epsilon_start=1.0,
                 epsilon_min=0.05,
                 epsilon_decay=1e-3,
                 buffer_capacity=5000,
                 batch_size=32,
                 target_update_freq = 1000):
        self.env = MinesweeperEnv(rows, cols, num_mines, sub_state_size)
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Online Q-network
        self.q_network = QNetwork(sub_state_size=sub_state_size)
        self.q_network.build(input_shape=(None, sub_state_size, sub_state_size))

        # Create target network
        self.target_network = QNetwork(sub_state_size=sub_state_size)
        self.target_network.build(input_shape=(None, sub_state_size, sub_state_size))
        self.target_network.set_weights(self.q_network.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.train_step_count = 0

        self.win_count = 0
        self.episode_count = 0

        self.previous_ratio= 0
        self.previous_weights= None

        self.target_update_freq = target_update_freq

    def select_action(self, state):
        """Epsilon-greedy among all covered cells."""
        actions = self.env.get_available_actions()
        if not actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(actions)

        # Evaluate Q-value for each possible action, pick max
        best_action = None
        best_q = float('-inf')
        for (r, c) in actions:
            sub = self.env.extract_sub_state(r, c)
            sub = np.expand_dims(sub, axis=0)  # shape (1, sub_state_size, sub_state_size)
            q_val = self.q_network(sub).numpy()[0][0]
            if q_val > best_q:
                best_q = q_val
                best_action = (r, c)

        return best_action

    def train_step(self):
        """Update Q-network from replay buffer samples."""
        if len(self.replay_buffer) < self.batch_size:
            return

        s_subs, actions, rewards, s_subs_next, dones = self.replay_buffer.sample(self.batch_size)

        # Approximate max Q for next sub-state
        next_q_vals = []
        for i in range(len(s_subs_next)):
            if dones[i]:
                next_q_vals.append(0.0)
            else:
                #nxt_val = self.q_network(np.expand_dims(s_subs_next[i], axis=0)).numpy()[0][0]
                nxt_val = self.target_network(np.expand_dims(s_subs_next[i], axis=0)).numpy()[0][0]
                next_q_vals.append(nxt_val)

        targets = rewards + self.gamma * np.array(next_q_vals, dtype=np.float32)

        with tf.GradientTape() as tape:
            predictions = self.q_network(s_subs)
            loss = self.loss_fn(targets, tf.squeeze(predictions))

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        self.train_step_count += 1

        # Hard Target Network update
        #if self.train_step_count % self.target_update_freq == 0:
        #    self.target_network.set_weights(self.q_network.get_weights())

        # Polyak-averaging Target Network update
        polyak_var = 0.001
        q_weights = self.q_network.get_weights()
        target_weights = self.target_network.get_weights()
        new_target_weights = [polyak_var * q_w + (1 - polyak_var) * t_w
                            for q_w, t_w in zip(q_weights, target_weights)]
        self.target_network.set_weights(new_target_weights)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def run_episode(self, max_steps = 2 * ((NUM_ROWS * NUM_COLS) - NUM_MINES)):
        """
        Play one Minesweeper episode until done or steps exhausted.
        max_steps is a heuristic to prevent algorithm using too many steps in an episode
        """
        state = self.env.reset()
        total_reward = 0.0
        step = 0
        episode_startTime = time.time()

        while True:
            action = self.select_action(state)
            if action is None:
                break

            next_state, reward, done, info = self.env.step(action)

            # Store transition
            s_sub_current = self.env.extract_sub_state(action[0], action[1])
            # next sub-state from same location (approximation)
            s_sub_next = self.env.extract_sub_state(action[0], action[1])

            self.replay_buffer.store(s_sub_current, action, reward, s_sub_next, done)
            total_reward += reward

            self.train_step()

            state = next_state
            step += 1
            if done or step >= max_steps:
                break

        episode_endTime = time.time()
        totalTime = episode_endTime - episode_startTime
        average_per_move = totalTime/step if step > 0 else 0
        
        if self.env.won:
            self.win_count += 1
        self.episode_count += 1

        return total_reward, self.env.done, self.env.won, totalTime, average_per_move

    def train_episodes(self, num_episodes=NUM_EPISODES, max_steps = 2 * ((NUM_ROWS * NUM_COLS) - NUM_MINES), csv_output=None):
        """
        Train over multiple episodes, log results to CSV, and display squares revealed.
        max_steps is a heuristic to prevent algorithm using too many steps in an episode
        """
        if csv_output is None:
            csv_output = f"metrics_output_{NUM_ROWS}x{NUM_COLS}_{NUM_MINES}mines_" \
                         f"{num_episodes}eps_{self.env.sub_state_size}ss_" \
                         f"{self.gamma}gamma_{self.lr}lr_{self.epsilon_min}emin_" \
                         f"{self.epsilon_decay}edecay_{self.replay_buffer.capacity}buf_" \
                         f"{self.batch_size}bs_{self.target_update_freq}tuf.csv"    
    
        # Prepare CSV file
        if os.path.exists(csv_output):
            os.remove(csv_output)

        with open(csv_output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Episode", "TotalReward", "SquaresRevealed", "Epsilon", "WinRatio", "Win",
                "SubStateSize", "Gamma", "LR", "EpsilonStart", "EpsilonMin", "EpsilonDecay",
                "BufferCapacity", "BatchSize", "TargetUpdateFreq", "EpisodeTime", "AverageMoveTime"
            ]) 

        for ep in range(num_episodes):
            ep_reward, done, won, totalTime, average_per_move = self.run_episode(max_steps=max_steps)
            squares_revealed = self.env.revealed_count
            win_ratio = self.win_count / float(self.episode_count)

            print(
                f"Ep {ep+1}/{num_episodes} | "
                f"Reward={ep_reward:.2f} | "
                f"SquaresRevealed={squares_revealed} | "
                f"Won={won} | "
                f"Eps={self.epsilon:.3f} | "
                f"WinRatio={win_ratio:.3f} | "
                f"EpisodeTime={totalTime:.2f}s | "
                f"AverageTime/Move={average_per_move:.4}s"
            )
            if self.previous_ratio < win_ratio:
                self.previous_weights = self.q_network.get_weights().copy()
                self.previous_ratio = win_ratio
            elif (win_ratio > 0.01) and (self.previous_ratio > win_ratio):
                self.q_network.set_weights(self.previous_weights.copy())
    
            # Write to CSV
            with open(csv_output, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    ep+1, ep_reward, squares_revealed, self.epsilon, win_ratio,
                    int(won),
                    self.env.sub_state_size, self.gamma, self.lr, 1.0,  # assuming epsilon_start is always 1.0
                    self.epsilon_min, self.epsilon_decay, 
                    self.replay_buffer.capacity, self.batch_size, self.target_update_freq,
                    totalTime, average_per_move
                ]) 
        print("Training complete.")
        print(f"Final Win Ratio: {self.win_count}/{self.episode_count} = {win_ratio:.3f}")

    @staticmethod
    def calculate_stratified_metrics(metrics_csv="metrics_output_current.csv",
                                     strat_interval=100,
                                     strat_csv=None):
        """
        Reads metrics from metrics_csv and computes both cumulative and rolling averages per strat_interval.
        The output CSV (strat_csv) will include the following columns:
      - EpisodeRange (e.g., "1-100")
          - RollingAvgReward, RollingAvgWinRatio, RollingAvgEpsilon, RollingAvgSquaresRevealed
          - CumulativeAvgReward, CumulativeAvgWinRatio, CumulativeAvgEpsilon, CumulativeAvgSquaresRevealed
          - And also hyperparameters: SubStateSize, Gamma, LR, EpsilonStart, EpsilonMin, EpsilonDecay,
            BufferCapacity, BatchSize, TargetUpdateFreq, Rows, Cols, NumMines, TotalEpisodes.

        Board parameters (from constants) are used to tag the output filename if strat_csv is not provided.
        """
        # Read all rows from metrics_csv. We assume that metrics_output.csv contains columns:
        # "Episode", "TotalReward", "SquaresRevealed", "Epsilon", "WinRatio"
        with open(metrics_csv, "r", newline="") as infile:
            reader = csv.DictReader(infile)
            rows = list(reader)

        total_rows = len(rows)
        strat_data = []

        # Initialize cumulative sums.
        cum_reward = 0.0
        cum_win = 0.0
        cum_epsilon = 0.0
        cum_squares = 0.0

        # Process episodes one by one to update cumulative sums.
        for i, row in enumerate(rows):
            ep_reward = float(row["TotalReward"])
            ep_win = int(row["Win"]) if "Win" in row else 0
            ep_epsilon = float(row["Epsilon"])
            ep_squares = float(row["SquaresRevealed"])
            
            cum_reward += ep_reward
            cum_win += ep_win  # If you had a binary indicator, you'd do: cum_win += (1 if row["Win"]=="True" else 0)
            cum_epsilon += ep_epsilon
            cum_squares += ep_squares

            # When we reach the end of a strat interval, compute both averages.
            if (i + 1) % strat_interval == 0 or (i + 1) == total_rows:
                # Calculate cumulative averages over episodes 1 to i+1.
                block = rows[i - (strat_interval - 1): i + 1] if (i + 1) >= strat_interval else rows[:i + 1]
                block_count = len(block)

                # rolling metrics
                roll_reward = sum(float(r["TotalReward"]) for r in block) / block_count
                roll_win = sum(int(r["Win"]) for r in block) / block_count
                roll_epsilon = sum(float(r["Epsilon"]) for r in block) / block_count
                roll_squares = sum(float(r["SquaresRevealed"]) for r in block) / block_count

                # cumulative metrics
                num_episodes_so_far = i + 1
                cum_avg_reward = cum_reward / num_episodes_so_far
                cum_avg_win = cum_win / num_episodes_so_far
                cum_avg_epsilon = cum_epsilon / num_episodes_so_far
                cum_avg_squares = cum_squares / num_episodes_so_far
    
                ep_start = block[0]["Episode"]
                ep_end = block[-1]["Episode"]

                # Hyperparams
                first_row = block[0]
                strat_data.append({
                    "EpisodeRange": f"{ep_start}-{ep_end}",
                    "RollingAvgReward": f"{roll_reward:.3f}",
                    "RollingAvgWinRatio": f"{roll_win:.3f}",
                    "RollingAvgEpsilon": f"{roll_epsilon:.3f}",
                    "RollingAvgSquaresRevealed": f"{roll_squares:.3f}",
                    "CumulativeAvgReward": f"{cum_avg_reward:.3f}",
                    "CumulativeAvgWinRatio": f"{cum_avg_win:.3f}",
                    "CumulativeAvgEpsilon": f"{cum_avg_epsilon:.3f}",
                    "CumulativeAvgSquaresRevealed": f"{cum_avg_squares:.3f}",
                    "SubStateSize": first_row.get("SubStateSize", ""),
                    "Gamma": first_row.get("Gamma", ""),
                    "LR": first_row.get("LR", ""),
                    "EpsilonStart": first_row.get("EpsilonStart", ""),
                    "EpsilonMin": first_row.get("EpsilonMin", ""),
                    "EpsilonDecay": first_row.get("EpsilonDecay", ""),
                    "BufferCapacity": first_row.get("BufferCapacity", ""),
                    "BatchSize": first_row.get("BatchSize", ""),
                    "TargetUpdateFreq": first_row.get("TargetUpdateFreq", ""),
                    "Rows": NUM_ROWS,
                    "Cols": NUM_COLS,
                    "NumMines": NUM_MINES,
                    "TotalEpisodes": total_rows
                })

        # If strat_csv filename is not provided, build one using board parameters and total episodes.
        if strat_csv is None:
            strat_csv = f"metrics_stratified_{NUM_ROWS}x{NUM_COLS}_{NUM_MINES}mines_{NUM_EPISODES}eps.csv"

        # Remove any existing file
        if os.path.exists(strat_csv):
            os.remove(strat_csv)

        with open(strat_csv, "w", newline="") as outfile:
            fieldnames = [
                "EpisodeRange",
                "RollingAvgReward", "RollingAvgWinRatio", "RollingAvgEpsilon", "RollingAvgSquaresRevealed",
                "CumulativeAvgReward", "CumulativeAvgWinRatio", "CumulativeAvgEpsilon", "CumulativeAvgSquaresRevealed",
                "SubStateSize", "Gamma", "LR", "EpsilonStart", "EpsilonMin", "EpsilonDecay",
                "BufferCapacity", "BatchSize", "TargetUpdateFreq",
                "Rows", "Cols", "NumMines", "TotalEpisodes"
            ]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in strat_data:
                writer.writerow(entry)

        print(f"Stratified metrics written to {strat_csv}")
