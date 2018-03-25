# Los Altos Hackathon 2018.
# The following code describes the testing of a Deep Recurrent Q-Network within an environment with partial observability.
# The environment used to test this is CIG scenario from the ViZDoom API.
# The network is tested over 10 episodes with 500 frame episodes, for a total of 5,000 frames.
# The network is NOT updated.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from vizdoom import *

import timeit
import math
import os
import sys

from DRQN import DRQN_agent, ExperienceReplay

def train(num_episodes, episode_length, scenario = "/Users/Lex/anaconda3/lib/python3.6/site-packages/vizdoom/scenarios/cig.cfg", map_path = 'map01', render = "False", print_bool = "True", delta_bool = "True"):
    # Totals
    total_reward = 0
    total_kills = 0

    sum_reward = 0
    sum_kills = 0

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
    actionDRQN = DRQN_agent((160, 256, 3), game.gget_available_buttons_size() - 2, 2, inital_learning_rate, "actionDRQN")

    session = tf.Session()

    meta = tf.train.import_meta_graph(meta_path)
    meta.restore(session, tf.train.latest_checkpoint(checkpoint))

    session.run(tf.global_variables_initializer())

    print("Testing.")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())     # Initialize all tensorflow variables
        for episode in range(num_episodes):
            game.new_episode()
            for frame in range(episode_length):
                state = game.get_state()
                s = state.screen_buffer
                v = state.game_variables()

                a = actionDRQN.prediction.eval(feed_dict = {actionDRQN.input: s})
                action = actions[a]

                reward = game.make_action(action)
                total_reward += reward

                if delta_bool:
                    delta = actionDRQN.dv.eval(feed_dict = {actionDRQN.input: s})

                if game.is_episode_finished():
                    break

                total_kills += v[2][0]

            print("At Episode %d, Reward = %.3f and Loss = %.3f." % (episode, total_reward, total_loss))

            sum_reward += total_reward
            sum_kills += total_kills

            total_reward = 0
            total_kills = 0

        print("Testing complete with total reward of %.3f and total killcount of %.3f" % (sum_reward, sum_kills))

if __name__ == '__main__':
    test(num_episodes = 10, episode_length = 500, render = True, delta_bool = False)
