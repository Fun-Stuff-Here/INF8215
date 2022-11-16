"""
Avalam agent.
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal

Uilisation d'une base de connaissance pour l'agent logique
"""

from signal import signal, alarm, SIGALRM
from numpy import array, int64
from avalam import Agent, agent_main
from njitavalam import Board, PLAYER1
from random_actions import random_action, random_actions_alarm_handler
from logic import get_action

class LogicAgent(Agent):

    """My Avalam agent."""

    def play(self, percepts, player, step, time_left):
        """
        Play a move
        :param percepts: dictionary representing the current board
        :param player: the player to control in this step (-1 or 1)
        :param step: the current step
        :param time_left: the time left for the agent to play
        :return: the action to play
        """
        board_array = array(percepts['m'], dtype=int64)
        board_copy = Board(board_array, percepts['max_height'])
        try:
            if time_left < 2.0:
                raise Exception("not enough time left")
            signal(SIGALRM, random_actions_alarm_handler(board_copy))
            alarm(int(time_left - 2.0)) # set an alarm for 2 seconds before the time limit
            action = get_action(percepts, player, step, time_left)
            alarm(0) # cancel alarm
            if board_copy.is_action_valid(action):
                return action
            raise Exception("Invalid action")
        except Exception as err:  # pylint: disable=broad-except
            print(err)
            return random_action(board_copy)

if __name__ == "__main__":
    my_agent = LogicAgent()
    percepts = { "m":[ [ 0,  0,  1, -1,  0,  0,  0,  0,  0],
                                [ 0,  1, -1,  1, -1,  0,  0,  0,  0],
                                [ 0, -1,  1, -1,  1, -1,  1,  0,  0],
                                [ 0,  1, -1,  1, -1,  1, -1,  1, -1],
                                [ 1, -1,  1, -1,  0, -1,  1, -1,  1],
                                [-1,  1, -1,  1, -1,  1, -1,  1,  0],
                                [ 0,  0,  1, -1,  1, -1,  1, -1,  0],
                                [ 0,  0,  0,  0, -1,  1, -1,  1,  0],
                                [ 0,  0,  0,  0,  0, -1,  1,  0,  0] ]
                , "max_height": 5 }
    my_agent.play(percepts=percepts, player=PLAYER1, step=1, time_left=900)
    try:
        agent_main(my_agent)
    except Exception as error:
        print(error)
        print("Error in agent_main")
