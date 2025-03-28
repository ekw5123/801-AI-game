from constants import *
from MineSweeperEnv import MinesweeperEnv
from qkNetwork import QNetwork
from qkNetwork import ReplayBuffer
import random
import tensorflow as tf
import numpy as np

###############################################################
#  SECTION 3: Deep Q-Learning Loop
# - Does all the DQN heavy work such as epsilon-greedy action selection
# - sampling from ReplayBuffer, Update Q-values, episode iteration
# - and logging results into .csv file
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
                 batch_size=32):
        self.env = MinesweeperEnv(rows, cols, num_mines, sub_state_size)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.q_network = QNetwork(sub_state_size=sub_state_size)
        self.q_network.build(input_shape=(None, sub_state_size, sub_state_size))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.train_step_count = 0

        self.win_count = 0
        self.episode_count = 0

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
                nxt_val = self.q_network(np.expand_dims(s_subs_next[i], axis=0)).numpy()[0][0]
                next_q_vals.append(nxt_val)

        targets = rewards + self.gamma * np.array(next_q_vals, dtype=np.float32)

        with tf.GradientTape() as tape:
            predictions = self.q_network(s_subs)
            loss = self.loss_fn(targets, tf.squeeze(predictions))

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        self.train_step_count += 1
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

        while True:
            action = self.select_action(state)
            if action is None:
                break

            next_state, reward, done, info = self.env.step(action)

            # Store transition
            s_sub_current = self.env.extract_sub_state(action[0], action[1])
            # next sub-state from same location (approximation) or you can pick a next-action
            s_sub_next = self.env.extract_sub_state(action[0], action[1])

            self.replay_buffer.store(s_sub_current, action, reward, s_sub_next, done)
            total_reward += reward

            self.train_step()

            state = next_state
            step += 1
            if done or step >= max_steps:
                break

        if self.env.won:
            self.win_count += 1
        self.episode_count += 1

        return total_reward, self.env.done, self.env.won

    def train_episodes(self, num_episodes=NUM_EPISODES, max_steps = 2 * ((NUM_ROWS * NUM_COLS) - NUM_MINES), csv_output="metrics_output.csv"):
        """
        Train over multiple episodes, log results to CSV, and display squares revealed.
        max_steps is a heuristic to prevent algorithm using too many steps in an episode
        """
        import os, csv
    
        # Prepare CSV file
        if os.path.exists(csv_output):
            os.remove(csv_output)
        with open(csv_output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "TotalReward", "SquaresRevealed", "Epsilon", "WinRatio"])
    
        for ep in range(num_episodes):
            ep_reward, done, won = self.run_episode(max_steps=max_steps)
            squares_revealed = self.env.revealed_count
            win_ratio = self.win_count / float(self.episode_count)
            print(
                f"Ep {ep+1}/{num_episodes} | "
                f"Reward={ep_reward:.2f} | "
                f"SquaresRevealed={squares_revealed} | "
                f"Won={won} | "
                f"Eps={self.epsilon:.3f} | "
                f"WinRatio={win_ratio:.3f}"
            )
    
            # Write to CSV
            with open(csv_output, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ep+1, ep_reward, squares_revealed, self.epsilon, win_ratio])
    
        print("Training complete.")
        print(f"Final Win Ratio: {self.win_count}/{self.episode_count} = {win_ratio:.3f}")
