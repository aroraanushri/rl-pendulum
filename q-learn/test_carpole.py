# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
import numpy as np
import os
import sys

episodes = 1

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("CartPole-v1")

if os.path.exists("q_table.npy"):
    q_table = np.load("q_table.npy")
else:
    print("no qtable found")
    env.close()
    sys.exit()

# epsilon = 1.0

cart_pos_bin = np.linspace(-4.8, 4.8, 7)[1:-1]
cart_vel_bin = np.linspace(-3.0, 3.0, 7)[1:-1]
pole_angle_bin = np.linspace(-0.418, 0.418, 7)[1:-1]
pole_vel_bin = np.linspace(-4.0, 4.0, 7)[1:-1]

def discretize_state(observation):
    cart_pos = np.digitize(observation[0], cart_pos_bin)
    cart_vel = np.digitize(observation[1], cart_vel_bin)
    pole_angle = np.digitize(observation[2], pole_angle_bin)
    pole_vel = np.digitize(observation[3], pole_vel_bin)
    return (cart_pos, cart_vel,pole_angle,pole_vel)


for episode in range(episodes):
    # Reset environment to start a new episode
    observation, info = env.reset()
    # observation: what the agent can "see" - cart position, velocity, pole angle, etc.
    # info: extra debugging information (usually not needed for basic learning)

    # print(f"Starting observation: {observation}")
    # print(env.action_space)
    # print(env.observation_space)
    # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
    # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

    episode_over = False
    total_reward = 0


    # learning_rate = 0.1
    # discount_factor = 0.99

    # epsilon=max(0.01, epsilon*0.995)



    state = discretize_state(observation)

    while not episode_over:
        # r = np.random.random()

        # if r<epsilon:
        #     action = env.action_space.sample() 
        # else:
        action = np.argmax(q_table[state])
        
        # Take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)

        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)

        # current_q = q_table[state+(action,)]

        next_state = discretize_state(observation)

        # best_future = np.max(q_table[next_state])

        # if terminated or truncated:
        #     target = reward
        # else:
        #     target = reward + discount_factor * best_future

        # new_q = current_q+learning_rate*(target-current_q)
        # q_table[state+(action,)]=new_q
        state=next_state

        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
# np.save("q_table.npy", q_table)
env.close()