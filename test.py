import gym
import gym_torcs

# from gym_torcs import TorcsEnv

from utils.noise import OUActionNoise, OU, AdaptiveParamNoiseSpec
from utils.replaybuffer import Buffer
from utils.models import ActorNetwork, CriticNetwork, learn
import numpy as np
import tensorflow as tf

import json
import pickle

import matplotlib.pyplot as plt


def run(video=True):

    # EXPERIMENT = 10

    # for i in range(EXPERIMENT):
    #     MAXIMUM_STEPS = 100_000

    #     np.random.seed(42)

    #     vision = False
    #     rewards = []
    #     try:
    #         store_file = open("Data/ddpg_test{}".format(i), "wb")
    #         # rewards = pickle.load(store_file)
    #         # store_file.close()
    #     except:
    #         # rewards = []
    #         return

    #     # plt.plot(rewards)
    #     # plt.show()

    #     if video:

    #         def obs_preprocess_fn(dict_obs):
    #             return np.hstack(
    #                 (
    #                     dict_obs["angle"],
    #                     dict_obs["track"],
    #                     dict_obs["trackPos"],
    #                     dict_obs["speedX"],
    #                     dict_obs["speedY"],
    #                     dict_obs["speedZ"],
    #                     dict_obs["wheelSpinVel"],
    #                     dict_obs["rpm"],
    #                 )
    #             )

    #         # env = TorcsEnv(vision=vision, throttle=True, gear_change=False)
    #         env = gym.make(
    #             "Torcs-v0",
    #             vision=vision,
    #             rendering=True,
    #             throttle=True,
    #             gear_change=False,
    #             obs_normalization=True,
    #             obs_preprocess_fn=obs_preprocess_fn,
    #             race_config_path="/home/chufan1990/Documents/RL/DDPG/practice.xml",
    #             lap_limiter=2,

    #         )

    #         num_states = np.squeeze(env.reset()).shape[0]
    #         num_actions = env.action_space.shape[0]

    #         actor = ActorNetwork(num_states, num_actions)
    #         critic = CriticNetwork(num_states, num_actions)
    #         for episode in range(1, 20):
    #             actor.model(tf.expand_dims(tf.convert_to_tensor(np.squeeze(env.reset())), 0))

    #             print("Now we load the weight")
    #             try:
    #                 actor.model.load_weights("models/{}/actormodel_{}.h5".format(i, episode*100))
    #                 critic.model.load_weights("models/{}/criticmodel_{}.h5".format(i, episode*100))
    #                 print("Weight load successfully")
    #             except:
    #                 print("Cannot find the weight: models/{}/actormodel_{}.h5".format(i, episode*100))
    #                 break

    #             print("TORCS Experiment Start.")
    #             done = False
    #             step = 0
    #             episodic_reward = 0

    #             state = np.squeeze(env.reset())

    #             for _ in range(MAXIMUM_STEPS):

    #                 tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

    #                 action_o = actor.policy(tf_state)

    #                 action = np.zeros_like(action_o)

    #                 action[0] = action_o[0]
    #                 action[1] = action_o[1]
    #                 action[2] = action_o[2] if action_o[2] >= 1e-2 else 0.0

    #                 next_state, reward, done, _ = env.step(np.squeeze(action))
    #                 next_state = np.squeeze(next_state)

    #                 value = critic.model([tf.convert_to_tensor(np.array(state).reshape(1, -1)), tf.convert_to_tensor(np.array(action).reshape(1, -1))])

    #                 state = next_state

    #                 step += 1

    #                 print(
    #                     "Experiment {} Model {} Action {} Reward {} ".format(
    #                         i, episode*100, action, reward, value
    #                     )
    #                 )
    #                 if done:
    #                     break

    #             rewards.append(episodic_reward)

    #             print("Episode {} Step {} Reward {}".format(episode, step, episodic_reward))

    #         env.end()
    #         pickle.dump(rewards, store_file)
    #         store_file.close()
    #         print("Task done.")
    EXPERIMENT = 14

    MAXIMUM_STEPS = 100_000

    np.random.seed(42)

    MODEL = 2000

    vision = False
    rewards = []
    try:
        store_file = open("Data/global_reward_{}".format(EXPERIMENT), "rb")
        rewards = pickle.load(store_file)
        store_file.close()
    except:
        # rewards = []
        print("????")
        return

    plt.plot(rewards)
    plt.xlabel("Number of iterations")
    plt.ylabel("Average reward")
    plt.show()

    number = np.argmax(rewards)

    print(number)

    number = 7896

    if video:

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

            # env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

        env = gym.make(
            "Torcs-v0",
            vision=vision,
            rendering=True,
            throttle=True,
            gear_change=False,
            obs_normalization=True,
            obs_preprocess_fn=obs_preprocess_fn,
            race_config_path="/home/chufan1990/Documents/RL/DDPG/practice.xml", 
            lap_limiter=2,
        )

        num_states = np.squeeze(env.reset()).shape[0]
        num_actions = env.action_space.shape[0]

        actor = ActorNetwork(num_states, num_actions)
        critic = CriticNetwork(num_states, num_actions)

        actor.model(tf.expand_dims(tf.convert_to_tensor(np.squeeze(env.reset())), 0))

        print("Now we load the weight")
        try:
            actor.model.load_weights(
                "models/{}/actor_{}.h5".format(EXPERIMENT, number)
            )
            tf.keras.utils.plot_model(actor.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
            critic.model.load_weights(
                "models/{}/critic_{}.h5".format(EXPERIMENT, number)
            )

            print("Weight load successfully")
        except:
            print(
                "Cannot find the weight: Experiement {} Model {}.h5".format(
                    EXPERIMENT, number
                )
            )
            return

        print("TORCS Experiment Start.")
        done = False
        step = 0
        episodic_reward = 0

        state = np.squeeze(env.reset())

        for _ in range(MAXIMUM_STEPS):

            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

            action = actor.policy(tf_state)

            next_state, reward, done, _ = env.step(np.squeeze(action))
            next_state = np.squeeze(next_state)

            value = critic.model(
                [
                    tf.convert_to_tensor(np.array(state).reshape(1, -1)),
                    tf.convert_to_tensor(np.array(action).reshape(1, -1)),
                ]
            )

            state = next_state

            step += 1

            print(
                "Experiment {} Model {} Action {} Reward {} ".format(
                    EXPERIMENT, MODEL, action, reward, value
                )
            )
            if done:
                break

            rewards.append(episodic_reward)

            # print(
            #     "Episode {} Step {} Reward {}".format(
            #         episode, step, episodic_reward
            #     )
            # )

        env.end()
    # pickle.dump(rewards, store_file)
    # store_file.close()
    print("Task done.")


if __name__ == "__main__":
    run()
