"""
Abstract Avalam agent that timeouts after a given time limit. and returns a random action.

Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal
"""

import abc
from signal import signal, alarm, SIGALRM, getsignal
from numpy import array, int64
from avalam import Agent
from njitavalam import Board
from graveyard.random_actions import random_action, random_actions_alarm_handler

class TimeSafeAgent(Agent):
    """
    Abstract Avalam agent that timeouts after a given time limit. and returns a random action.
    """
    __metaclass__ = abc.ABCMeta

    def play(self, percepts, player, step, time_left):
        """
        Play a move using the get_action method
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
            original_handler = getsignal(SIGALRM)
            signal(SIGALRM, random_actions_alarm_handler(board_copy))
            alarm(int(time_left - 2.0)) # set an alarm for 2 seconds before the time limit
            action = self.get_action(Board(board_array, percepts['max_height']), player, step, time_left)
            signal(SIGALRM, original_handler)
            alarm(0) # cancel alarm
            if board_copy.is_action_valid(action):
                return action
            raise Exception("Invalid action")
        except Exception as err:  # pylint: disable=broad-except
            print(err)
            return random_action(board_copy)

    @abc.abstractmethod
    def get_action(self, percepts, player, step, time_left):
        """
        Get an action
        :param percepts: dictionary representing the current board
        :param player: the player to control in this step (-1 or 1)
        :param step: the current step
        :param time_left: the time left for the agent to play
        :return: the action to play
        """
        raise NotImplementedError("get_action must be implemented by a subclass")
