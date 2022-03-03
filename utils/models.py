import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class Model(tf.keras.Model):
    def __init__(self, name, network="mlp", **network_kwargs):
        super(Model, self).__init__(name=name)
        self.network = network
        self.network_kwargs = network_kwargs

    @property
    def perturbable_vars(self):
        return [
            var
            for var in self.trainable_variables
            if "layer_normalization" not in var.name
        ]


class Actor(Model):
    def __init__(
        self, num_states, num_actions, name="actor", network="mlp", **network_kwargs
    ):
        super().__init__(name=name, network=network, **network_kwargs)
        steer_init = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.003, seed=None
        )

        self.inputs = layers.InputLayer(input_shape=(num_states,))
        self.h1 = layers.Dense(512, activation="relu")
        self.n1 = layers.BatchNormalization()
        self.h2 = layers.Dense(512, activation="relu")
        self.n2 = layers.BatchNormalization()

        self.steer = layers.Dense(
            1, activation="tanh", kernel_initializer=steer_init, dtype=tf.float64
        )
        self.accel = layers.Dense(
            1, activation="sigmoid", dtype=tf.float64
        )
        self.brake = layers.Dense(
            1, activation="sigmoid", kernel_initializer=steer_init, dtype=tf.float64
        )

    @tf.function
    def call(self, obs):
        x = self.inputs(obs)
        x = self.h1(x)
        x = self.n1(x)
        x = self.h2(x)
        x = self.n2(x)
        s = self.steer(x)
        a = self.accel(x)
        b = self.brake(x)
        return [s, a, b]


class ActorNetwork:
    def __init__(self, num_states, num_actions, param_noise=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.param_noise = param_noise
        self.model = Actor(self.num_states, self.num_actions)
        self.target = Actor(self.num_states, self.num_actions)
        self.target.set_weights(self.model.get_weights())

        if self.param_noise is not None:
            self.setup_param_noise()

    def setup_param_noise(self):
        assert self.param_noise is not None

        # # Configure perturbed actor.
        self.perturbed_actor = Actor(self.num_states, self.num_actions)

        # Configure separate copy for stddev adoption.
        self.perturbed_adaptive_actor = Actor(self.num_states, self.num_actions)

    def update_target_network(self, tau):
        new_weights = []
        target_variables = self.target.weights
        for i, variable in enumerate(self.model.weights):
            new_weights.append(variable * tau + target_variables[i] * (1 - tau))
        self.target.set_weights(new_weights)

    def policy(self, state, apply_noise=False):
        if self.param_noise is not None and apply_noise:
            action = tf.squeeze(self.perturbed_actor(state))
        else:
            action = tf.squeeze(self.model(state))
        return np.squeeze(action)

    def adapt_param_noise(self, obs):
        if self.param_noise is None:
            return 0.0

        mean_distance = self.get_mean_distance(obs).numpy()

        self.param_noise.adapt(mean_distance)
        return mean_distance

    def get_mean_distance(self, state):
        update_perturbed_actor(
            self.model, self.perturbed_adaptive_actor, self.param_noise.current_stddev
        )
        action = tf.squeeze(self.model(state))
        adaptive_action = tf.squeeze(self.perturbed_adaptive_actor(state))
        mean_distance = tf.sqrt(tf.reduce_mean(tf.square(action - adaptive_action)))
        return mean_distance

    def reset(self):
        if self.param_noise is not None:
            update_perturbed_actor(
                self.model, self.perturbed_actor, self.param_noise.current_stddev
            )


class CriticNetwork:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.model = self._build_model()
        self.target = tf.keras.models.clone_model(self.model)

    def _build_model(self):
        state_input = layers.Input(shape=(self.num_states))
        state_out = layers.Dense(512, activation="relu")(state_input)
        state_out = layers.BatchNormalization()(state_out)
        state_out = layers.Dense(521, activation="relu")(state_out)
        state_out = layers.BatchNormalization()(state_out)

        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(521, activation="relu")(action_input)
        action_out = layers.BatchNormalization()(action_out)

        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(512, activation="relu")(concat)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(512, activation="tanh")(out)
        out = layers.BatchNormalization()(out)
        outputs = layers.Dense(1, dtype=tf.double)(out)

        model = tf.keras.Model([state_input, action_input], outputs)
        return model

    def update_target_network(self, tau):
        new_weights = []
        target_variables = self.target.weights
        for i, variable in enumerate(self.model.weights):
            new_weights.append(variable * tau + target_variables[i] * (1 - tau))
        self.target.set_weights(new_weights)


def learn(
    actor,
    critic,
    buffer,
    gamma,
    actor_optimizer=tf.keras.optimizers.Adam(0.001),
    critic_optimizer=tf.keras.optimizers.Adam(0.001),
    batch_size=32,
    beta=0.5,
):
    (
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        done_batch,
        weight_batch,
        index_batch,
    ) = buffer.sample(batch_size, beta)

    # (
    #     state_batch,
    #     action_batch,
    #     reward_batch,
    #     next_state_batch,
    #     done_batch,
    # ) = buffer.sample(batch_size)

    state_batch = state_batch.reshape(batch_size, -1)
    next_state_batch = next_state_batch.reshape(batch_size, -1)
    action_batch = action_batch.reshape(batch_size, -1)
    reward_batch = reward_batch.reshape(-1, 1)
    done_batch = done_batch.reshape(-1, 1)

    with tf.GradientTape() as tape:
        target_actions = np.squeeze(tf.squeeze(actor.target(next_state_batch)).numpy().T)
        y = reward_batch + gamma * done_batch * critic.target(
            [next_state_batch, target_actions]
        )

        critic_value = critic.model([state_batch, action_batch])
        td_errors = tf.math.square(y - critic_value)
        critic_loss = tf.math.reduce_mean(td_errors * weight_batch)
        # critic_loss = tf.math.reduce_mean(td_errors)
    critic_grad = tape.gradient(critic_loss, critic.model.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grad, critic.model.trainable_variables))


    with tf.GradientTape() as tape:
        actions = tf.transpose(tf.squeeze(actor.model(state_batch)))
        critic_value = critic.model([state_batch, actions])
        actor_loss = -tf.math.reduce_mean(critic_value)

    actor_grad = tape.gradient(actor_loss, actor.model.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grad, actor.model.trainable_variables))
    
    actor.adapt_param_noise(state_batch)
    new_priorities = np.abs(td_errors) + 1e-6
    buffer.update_priorities(index_batch, new_priorities)

def update_perturbed_actor(actor, perturbed_actor, param_noise_stddev):
    for var, perturbed_var in zip(actor.variables, perturbed_actor.variables):
        perturbed_var.assign(var)

    for var, perturbed_var in zip(
        actor.perturbable_vars, perturbed_actor.perturbable_vars
    ):

        perturbed_var.assign(
            var
            + tf.random.normal(
                shape=tf.shape(var),
                mean=0.0,
                stddev=param_noise_stddev,
                dtype=var.dtype,
            )
        )

