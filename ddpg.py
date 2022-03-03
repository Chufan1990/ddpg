import gym
import gym_torcs

from utils.noise import OUActionNoise, OU, AdaptiveParamNoiseSpec
from utils.replaybuffer import Buffer
from utils.models import ActorNetwork, CriticNetwork, learn, update_perturbed_actor

import numpy as np
import tensorflow as tf
import pandas as pd

from baselines.deepq.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


def run(training=True, Experiment=15, start=0):
    BUFFER_SIZE = 10_000
    BATCH_SIZE = 100
    GAMMA = 0.99
    TAU = 0.001
    EXPLORE = 100_000
    EPISODES = 2_000
    MAXIMUM_STEPS = 100_000
    STORE_EVERY = 100
    epsilon = 0.3
    ACTOR_LR = 1e-4
    CRITIC_LR = 1e-3
    NOISE_STDDEV = 2 * 1e-1
    ALPHA = 0.2
    BETA = 0.5

    np.random.seed(42)
    np.set_printoptions(formatter={"float": "{: 0.2f}".format})

    def obs_preprocess_fn(dict_obs):
        return np.hstack(
            (
                dict_obs["angle"],
                dict_obs["track"],
                dict_obs["trackPos"],
                dict_obs["speedX"],
                dict_obs["speedY"],
                dict_obs["speedZ"],
                dict_obs["wheelSpinVel"],
                dict_obs["rpm"],
            )
        )

    env = gym.make(
        "Torcs-v0",
        rendering=not training,
        throttle=True,
        gear_change=False,
        obs_normalization=True,
        obs_preprocess_fn=obs_preprocess_fn,
        race_config_path="/home/chufan1990/Documents/RL/DDPG/practice.xml",
        lap_limiter=2
    )

    num_states = np.squeeze(env.reset()).shape[0]
    num_actions = env.action_space.shape[0]

    param_noise = AdaptiveParamNoiseSpec(
        initial_stddev=float(NOISE_STDDEV), desired_action_stddev=float(NOISE_STDDEV)
    )

    # param_noise=None

    actor = ActorNetwork(num_states, num_actions, param_noise)
    critic = CriticNetwork(num_states, num_actions)

    actor_optimizer = tf.keras.optimizers.Adam(lr=ACTOR_LR)
    critic_optimizer = tf.keras.optimizers.Adam(lr=CRITIC_LR)

    lower_bound = env.action_space.low
    upper_bound = env.action_space.high

    replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=ALPHA)

    # replay_buffer = ReplayBuffer(BUFFER_SIZE)

    beta = BETA

    noise = OU()

    reward_buffer = np.zeros((500, 1))
    global_iteration = 0
    global_rewards = []

    print("TORCS Experiment Start.")
    for episode in range(1, EPISODES + 1):
        done = False
        step = 0

        state = np.squeeze(env.reset())

        if global_iteration > MAXIMUM_STEPS:
            break

        for _ in range(MAXIMUM_STEPS):

            global_iteration += 1

            epsilon -= 1.0 / EXPLORE

            epsilon = max(epsilon, 0.0)

            beta += 1.0 / EXPLORE if beta < 1.0 else 0

            noise_t = np.zeros(3)
            action = np.zeros(3)

            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

            action_o = actor.policy(tf_state, apply_noise=True)

            if training and param_noise is None:
                noise_t[0] = max(epsilon, 0.1) * noise.function(
                    action_o[0], 0.0, 1.00, 0.30
                )
                noise_t[1] = max(epsilon, 0.1) * noise.function(
                    action_o[1], 0.5, 1.00, 0.10
                )
                noise_t[2] = max(epsilon, 0.1) * noise.function(
                    action_o[2], -0.1, 1.00, 0.05
                )

                if np.random.random() <= 0.3:
                    # print("********Now we apply the brake***********")
                    noise_t[2] = max(epsilon, 0.1) * noise.function(
                        action_o[2], 0.9, 1.00, 0.01
                    )

            action = action_o + noise_t

            if np.random.random() > (1 - 3.0 * epsilon):
                action[2] = 0.0

            action = np.clip(action, lower_bound, upper_bound)

            if action[2] < 1e-2:
                action[2] = 0.0

            next_state, reward, done, _ = env.step(np.squeeze(action))
            next_state = np.squeeze(next_state)

            reward_buffer[global_iteration % 500] = reward

            window_mean_reward = np.mean(reward_buffer)

            global_rewards.append(window_mean_reward)

            reward -= 50.0
            reward /= 50.0

            if training:
                replay_buffer.add(state, action, reward, next_state, int(not done))

                learn(
                    actor=actor,
                    critic=critic,
                    buffer=replay_buffer,
                    gamma=GAMMA,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    batch_size=BATCH_SIZE,
                    beta=beta,
                )

                actor.update_target_network(tau=TAU)
                critic.update_target_network(tau=TAU)

            state = next_state

            step += 1

            print(
                "Episode {} Step {} Action {} Noise {} Reward {} Std {} Iteration {}".format(
                    episode,
                    step,
                    action,
                    noise_t,
                    np.array([reward]),
                    np.array([param_noise.current_stddev]),
                    # 0,
                    global_iteration
                )
            )
            if window_mean_reward > 100:
                print("Save promising model")
                actor.model.save_weights(
                    "models/{}/actor_{}.h5".format(Experiment, global_iteration),
                    overwrite=True,
                )

                critic.model.save_weights(
                    "models/{}/critic_{}.h5".format(Experiment, global_iteration),
                    overwrite=True,
                )

            if done:
                actor.reset()
                break


        if np.mod(episode, STORE_EVERY) == 0:
            if training:
                print("Now we save model")
                actor.model.save_weights(
                    "models/{}/actormodel_{}.h5".format(Experiment, episode),
                    overwrite=True,
                )
                critic.model.save_weights(
                    "models/{}/criticmodel_{}.h5".format(Experiment, episode),
                    overwrite=True,
                )

            df = pd.DataFrame(global_rewards)
            df.to_csv("{}.csv".format(Experiment))

        print(
            "\n Episode {} Total Step {} Epsilon {} Beta {}\n".format(
                episode,
                step,
                np.array([epsilon]),
                np.array([beta]),
            )
        )
    env.end()
    print("Task done.")


if __name__ == "__main__":
    run()
