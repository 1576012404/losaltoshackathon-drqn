# Los Altos Hackathon 2018.
# The following code describes the training of a Deep Recurrent Q-Network within an environment with partial observability.
# The environment used to train this is CIG scenario from the ViZDoom API.
# The network is trained over 10,000 episodes with 300 frame episodes, for a total of 3,000,000 frames.
# The network is updated every five frames.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from vizdoom import *

import timeit
import math
import os
import sys

from DRQN import DRQN_agent, ExperienceReplay

def train(num_episodes, episode_length, initial_learning_rate, scenario = "/Users/Lex/anaconda3/lib/python3.6/site-packages/vizdoom/scenarios/cig.cfg", map_path = 'map02', render = "False", print_bool = "True", delta_bool = "True"):
    # Discount Parameter for Q-value Computation
    discount_factor = .99

    # Random Chance
    greedy_choice_chance = 0.2
    greedy_choice_chance_end =  0.01
    greedy_decay = (greedy_choice_chance - greedy_choice_chance_end) / (num_episodes)

    # Frequency Variables
    update_frequency = 5
    store_frequency = 50
    print_frequency = 1000

    # Totals
    total_reward = 0
    total_loss = 0

    old_q_value = 0

    # Lists
    rewards = []
    losses = []

    # ViZDoom Setup
    game = DoomGame()
    game.set_doom_scenario_path(scenario)
    game.set_doom_map(map_path)

    game.set_screen_resolution(ScreenResolution.RES_256X160)    # (256, 160, 3) frame of information
    game.set_screen_format(ScreenFormat.RGB24)

    # Particles and Effects
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)
    game.set_render_messages(False)
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)

    # Sets Avaliable Buttons
    game.add_available_button(Button.MOVE_LEFT)
    game.add_available_button(Button.MOVE_RIGHT)
    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.MOVE_FORWARD)
    game.add_available_button(Button.MOVE_BACKWARD)
    game.add_available_button(Button.ATTACK)

    # Create an array of actions
    actions = np.zeros((game.get_available_buttons_size(), game.get_available_buttons_size()))
    count = 0
    for i in actions:
        i[count] = 1
        count += 1
    actions = actions.astype(int).tolist()

    # Sets Avaliable Delta Buttons
    game.add_available_button(Button.TURN_LEFT_RIGHT_DELTA, 90)
    game.add_available_button(Button.LOOK_UP_DOWN_DELTA, 90)

    # Set avaliable Game Variables to Ammo, Health, Amount of Kills
    game.add_available_game_variable(GameVariable.AMMO0)
    game.add_available_game_variable(GameVariable.HEALTH)
    game.add_available_game_variable(GameVariable.KILLCOUNT)

    # Make sure environment stops at maximum number of steps
    game.set_episode_timeout(6 * episode_length)
    game.set_episode_start_time(10)

    # Visual Features
    game.set_window_visible(render)
    game.set_sound_enabled(False)

    # Set Reward for Living
    game.set_living_reward(0)

    # Set mode to player
    game.set_mode(Mode.PLAYER)

    # Initialize the game environment
    game.init()

    # Models
    actionDRQN = DRQN_agent((160, 256, 3), game.get_available_buttons_size() - 2, 2, initial_learning_rate, "actionDRQN")
    targetDRQN = DRQN_agent((160, 256, 3), game.get_available_buttons_size() - 2, 2, initial_learning_rate, "targetDRQN", delta_bool = False)
    experiences = ExperienceReplay(1000)

    # Storage for models
    saver = tf.train.Saver({v.name: v for v in actionDRQN.parameters}, max_to_keep = 1)

    print("Training.")
    start_time = timeit.default_timer()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())     # Initialize all tensorflow variables
        for episode in range(num_episodes):
            game.new_episode()
            for frame in range(episode_length):
                state = game.get_state()
                s = state.screen_buffer

                a = actionDRQN.prediction.eval(feed_dict = {actionDRQN.input: s})[0]
                action = actions[a]

                reward = game.make_action(action)
                total_reward += reward

                if delta_bool:
                    delta = actionDRQN.dv.eval(feed_dict = {actionDRQN.input: s})

                if game.is_episode_finished():
                    break

                if (frame % store_frequency) == 0:
                    experiences.appendToBuffer((s, action, reward))

                if (frame % update_frequency) == 0:
                    memory = experiences.sample(1)
                    mem_frame = memory[0][0]
                    mem_reward = memory[0][2]

                    Q1 = actionDRQN.output.eval(feed_dict = {actionDRQN.input: mem_frame})
                    Q2 = targetDRQN.output.eval(feed_dict = {targetDRQN.input: mem_frame})

                    learning_rate = actionDRQN.learning_rate.eval()

                    Qtarget = old_q_value + learning_rate * (mem_reward + discount_factor * Q2 - old_q_value)     # Discounted Q-value
                    old_q_value = Qtarget

                    # Compute Loss
                    loss = actionDRQN.loss.eval(feed_dict = {actionDRQN.target_vector: Qtarget, actionDRQN.input: mem_frame})
                    total_loss += loss

                    # Update both networks
                    actionDRQN.update.run(feed_dict = {actionDRQN.target_vector: Qtarget, actionDRQN.input: mem_frame})
                    targetDRQN.update.run(feed_dict = {targetDRQN.target_vector: Qtarget, targetDRQN.input: mem_frame})

            rewards.append((episode, total_reward))
            losses.append((episode, total_loss))

            if (episode % print_frequency) == 0:
                print("At Episode %d, Reward = %.3f and Loss = %.3f." % (episode, total_reward, total_loss))

            if greedy_choice_chance > greedy_choice_chance_end:
                greedy_choice_chance -= greedy_decay

            total_reward = 0
            total_loss = 0

        end_time = timeit.default_timer()

        print("Training finished after %.2f hours." % ((end_time - start_time) / 3600))
        print("Recording model reward and loss.")

        if record_model:
            sp = saver.save(sess, 'DRQN_best_params.ckpt')
            print("DRQN model stored at %s" % sp)

            with open('DRQN_rewards_and_losses.txt', 'a+') as g:
                g.write("Rewards (Step, Reward):\n")
                for r in rewards:
                    g.write("(" + str(r[0]) + ", " + str(r[1]) + ")\n")

                g.write("\n")
                g.write("-------------------------------------------------------\n")
                g.write("\n")

                g.write("Losses (Step, Loss):\n")
                for l in losses:
                    g.write("(" + str(l[0]) + ", " + str(l[1]) + ")\n")
                g.close()

if __name__ == '__main__':
    train(num_episodes = 10000, episode_length = 300, initial_learning_rate = 0.01, render = False, delta_bool = False)
