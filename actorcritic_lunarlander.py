"""This is a policy gradient learning training algotrithm for the gym environment cartpole"""
import logging, os
import tensorflow_probability as tfp

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gym
import Box2D
import tensorflow as tf
import numpy as np
import ray
from really import SampleManager
from really.utils import (
    dict_to_dict_of_datasets,
)

# Action is two real values vector from -1 to +1.
# First controls main engine, -1..0 off,
#                              0..+1 throttle from 50% to 100% power
# Engine can't work with less than 50% power.
# Second value -1.0..-0.5 fire left engine,
#              +0.5..+1.0 fire right engine, -0.5..0.5 off.
""" Actor-critic network: computes state value estimates (baselines) and distribution parameters (mu and log(sigma)) for action values across action dimensions"""
class ActorCritic(tf.keras.Model):

    def __init__(self, state_dim=8, num_actions=2, hdim1=32, hdim2=32, init=tf.keras.initializers.HeUniform()):
        super(ActorCritic, self).__init__()
        self.critic_net = [
            tf.keras.layers.Dense(hdim1, activation=tf.nn.leaky_relu, kernel_initializer=init, name="critic1"),
            tf.keras.layers.Dense(hdim2, activation=tf.nn.leaky_relu, kernel_initializer=init, name="critic2"),
            tf.keras.layers.Dense(1, kernel_initializer=init, name="critic_out")
        ]
        self.actor_net = [
            tf.keras.layers.Dense(hdim1, activation=tf.nn.leaky_relu, kernel_initializer=init, name="actor1"),
            tf.keras.layers.Dense(hdim2, activation=tf.nn.leaky_relu, kernel_initializer=init, name="actor2")
        ]
        self.action_values_mu = tf.keras.layers.Dense(num_actions, activation = tf.keras.activations.tanh, kernel_initializer=init, name="actor_mu")
        self.action_values_logsigma = tf.keras.layers.Dense(num_actions, kernel_initializer=init, name="actor_sigma")

    @tf.function
    def call(self,x):
        output = {}
        v = x
        for layer in self.critic_net:
            v = layer(v)
        # later filter model.trainable_variables according to the layers' names to compute and apply gradients separately
        # compute the state values again when optimizing because returns from manager cannot be backpropagated
        for layer in self.actor_net:
            x = layer(x)
        mus = self.action_values_mu(x)
        sigmas = self.action_values_logsigma(x)
        sigmas = tf.clip_by_value(sigmas, -20, 2)

        output["value_estimate"] = v
        output["mu"] = mus
        output["sigma"] = tf.exp(sigmas)

        return output

if __name__ == "__main__":

    reshape_and_cast = lambda x: tf.cast(tf.reshape(x, (optim_batch_size, -1)), tf.float32)

    gamma = 0.95
    entropy_coeff = 0.1
    learning_rate = 0.001
    optimizer_actor = tf.keras.optimizers.Adam(learning_rate)
    optimizer_critic = tf.keras.optimizers.Adam(learning_rate)
    loss_function = tf.keras.losses.MSE
    num_episodes = 1

    kwargs = {
        "model": ActorCritic,
        "environment": "LunarLanderContinuous-v2",
        "num_parallel": 1,
        "total_steps": 1000,
        "action_sampling_type": "continuous_normal_diagonal",
        "num_steps": 1000,
        "returns": ['value_estimate', 'log_prob', 'monte_carlo']
    }

    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)
    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "\progress_a2c"

    buffer_size = 1000 # not used
    test_steps = 500
    epochs = 20
    sample_size = 1000
    optim_batch_size = 1000
    saving_after = 10

    # keys for replay buffer needed for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done", "monte_carlo"]

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss_actor", "loss_critic", "time_steps"]
    )

    # initial testing:
    print("test before training: ")
    #manager.test(test_steps, do_print=True)

    # get initial agent
    agent = manager.get_agent()
    target_agent = agent

    for epoch in range(epochs):

        # training core
        print("collecting experience..")
        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size, from_buffer=False)
        #breakpoint()
        print(f"collected data for: {sample_dict.keys()}")
        sample_dict = {k:v[:1000] for k,v in sample_dict.items()}
        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)
        print("optimizing...")

        print(data_dict.keys())

        data_dict = {k:v.map(reshape_and_cast) for k,v in data_dict.items()}

        # whole batch update (just 1 iteration)
        for a, s, r, s_new, nd, v_e, log_p, mc_r in zip(*[data_dict[k] for k in data_dict.keys()]):
            # get critic's state value estimates (baseline)
            v_snew = agent.v(s_new)

            # SARSA estimate to use as regression target for V(s)
            TD_target = r + nd * gamma * v_snew
            # MC estimate to use as regression target for V(s)
            MC_target = mc_r

            with tf.GradientTape() as tape:
                v_pred = agent.v(s)
                loss_critic = tf.keras.losses.MSE(MC_target, v_pred)
                gradients_critic = tape.gradient(loss_critic, agent.model.trainable_variables)
                optimizer_critic.apply_gradients(zip(gradients_critic,agent.model.trainable_variables))

            # Advantage estimate to use as regression target
            advantage = mc_r - v_pred
            #print(np.around(action_probs, 2), np.around(v_snew - v_s, 3), 'Advantage:', np.around(advantage, 2))

            with tf.GradientTape() as tape:
                action_probs, entropy = agent.flowing_log_prob(s, a, return_entropy=True)
                loss_actor = - action_probs * advantage - entropy_coeff * entropy
                gradients_actor = tape.gradient(loss_actor, agent.model.trainable_variables)
                optimizer_actor.apply_gradients(zip(gradients_actor, agent.model.trainable_variables))

        # set new weights
        manager.set_agent(agent.get_weights())
        # get new agent
        agent = manager.get_agent()
        # update aggregator
        if epoch%5!=0:
            time_steps = manager.test(test_steps, test_episodes=10)
        else:
            time_steps = manager.test(test_steps,test_episodes=10, render=True)
        manager.update_aggregator(loss_critic=np.mean(loss_critic), loss_actor=np.mean(loss_actor), time_steps=time_steps)
        # print progress
        print(
            f"epoch ::: {epoch}  critic loss ::: {np.mean(loss_critic)} actor loss ::: {np.mean(loss_actor)} avg env steps ::: {np.mean(time_steps)}"
        )

    # and load models
    #manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True, do_print=True)
