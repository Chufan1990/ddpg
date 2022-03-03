## 


1. Experiment 1:

    - Hyperparameters:
        |---------------|:-------:|
        | BUFFER_SIZE   | 100_000 |
        | BATCH_SIZE    | 32      |
        | GAMMA         | 0.99    |
        | TAU           | 0.001   |
        | EXPLORE       | 100_000 |
        | EPISODES      | 10_000  |
        | MAXIMUM_STEPS | 100_000 |
        | STORE_EVERY   | 200     |
        | epsilon       | 1.0     |
        | ACTOR_LR      | 1e-4    |
        | CRITIC_LR     | 1e-3    |

    - Algorithm:  DDPG
    - Noise: Ornstein-Uhlenbeck process

    - Results: Learning outcoumes were promising in the first 2000 episodes (the cumulative rewards increasing) but stucked in a poor local minimum after 3000 episodes. It recovered from 10000 episodes but stucked again after 14000 episodes and doen't recover.


    - Phonenmon: The vehicle kept turning a right until hit the wall and ended the game.


2. Experiment 2:

    - Change environment reward from cumulative reward to single reward

3. Experiment 3:

    - Change critic network structure. Change critic gradient function at final step.

4. Experiment 4:


5. Experiment 5:


6. Experiment 6:
    - Normalize replay buffer

7. Experiment 7:
    - Prioritized Replay Buffer, param noise, normalized reward (mean=50, std = 10)

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
    NOISE_STDDEV = 0.5
    ALPHA = 0.2
    BETA = 0.5

8. Experiment 7:
    - Prioritized Replay Buffer, param noise, normalized reward (mean=50, std = 10)

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
    NOISE_STDDEV = 0.35
    ALPHA = 0.2
    BETA = 0.5



9. Experiment 7:
    - Prioritized Replay Buffer, param noise, normalized reward (mean=150, std = 20)

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
    NOISE_STDDEV = 0.35
    ALPHA = 0.2
    BETA = 0.5


12. Every on broad. Stop at 950 episode around.

13. Everything on brouad. Record average reward of 500 sliding window. Record every 100 episode.