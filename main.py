import torch
import torch.optim as optim
import random
from collections import deque
import numpy as np
from env import DroneDeliveryEnv
from model import DQN


def train(env, num_episodes=1000, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    model = DQN(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    memory = deque(maxlen=2000)
    batch_size = 64

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0

        for t in range(200):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(model(state)).item()

            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)
            total_reward += reward

            memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*batch)

                batch_states = torch.stack(batch_states)
                batch_actions = torch.LongTensor(batch_actions)
                batch_rewards = torch.FloatTensor(batch_rewards)
                batch_next_states = torch.stack(batch_next_states)
                batch_dones = torch.FloatTensor([1.0 if done else 0.0 for done in batch_dones])

                current_q_values = model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                max_next_q_values = model(batch_next_states).max(1)[0]
                expected_q_values = batch_rewards + (gamma * max_next_q_values * (1 - batch_dones))

                loss = criterion(current_q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        print(f"Episode {episode}, Total Reward: {total_reward}")


def evaluate(env, model, num_episodes=100):
    total_rewards = 0
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0

        for t in range(200):
            with torch.no_grad():
                action = torch.argmax(model(state)).item()

            state, reward, done, _ = env.step(action)
            state = torch.FloatTensor(state)
            total_reward += reward

            if done:
                break

        total_rewards += total_reward

    avg_reward = total_rewards / num_episodes
    print(f"Average Reward: {avg_reward}")


if __name__ == "__main__":
    obstacles = []  # Define obstacles
    target_area = np.array([10, 10, 10])  # Define target area
    boundary = [20, 20, 20]  # Define boundary
    env = DroneDeliveryEnv(obstacles, target_area, boundary)

    # Train the model
    train(env)

    # Evaluate the trained model
    model = DQN(env.observation_space.shape[0], env.action_space.n)
    evaluate(env, model)
