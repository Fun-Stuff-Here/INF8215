#!/usr/bin/env python3
"""
Avalam agent.
Copyright (C) 2022, <<<<<<<<<<< YOUR NAMES HERE >>>>>>>>>>>
Polytechnique Montréal

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

"""
import random

from avalam import *


def random_action(percepts):
    return random.choice(list(dict_to_board(percepts).get_actions()))


class MyAgent(Agent):

    """My Avalam agent."""

    """Start game command: python3 game.py "http://$(hostname).local:8000" human --time 900"""

    def play(self, percepts, player, step, time_left):
        """
        This function is used to play a move according
        to the percepts, player and time left provided as input.
        It must return an action representing the move the player
        will perform.
        :param percepts: dictionary representing the current board
            in a form that can be fed to `dict_to_board()` in avalam.py.
        :param player: the player to control in this step (-1 or 1)
        :param step: the current step number, starting from 1
        :param time_left: a float giving the number of seconds left from the time
            credit. If the game is not time-limited, time_left is None.
        :return: an action
            eg; (1, 4, 1 , 3) to move tower on cell (1,4) to cell (1,3)
        """

        # avoid loosing by forfeit, backup plan
        if time_left < 2:
            return random_action(percepts=percepts)

        """
        Rule 1: si il est possible de faire une tour de 5
        Rule 2: Isoler une tour de notre couleur
        Rule 3: Bouger le pion qui a le plus de move available
        Rule 4: Bouger le pion qui est le plus loin du centre
        Rule 5: Bouger le pion de la couleur adverse sur un autre pion adverse
        Rule 6: Bouger le pion de sa couleur sur un pion de la couleur adverse
        """


        """
        print("percept:", percepts)
        print("player:", player)
        print("step:", step)
        print("time left:", time_left if time_left else '+inf')

        # TODO: implement your agent and return an action for the current step.
        board = dict_to_board(percepts)
        actions = list(board.get_actions())
        print('step', step, 'player', player, 'actions', len(actions))
        """
        from time import sleep
        sleep(1)
        return random_action(percepts=percepts)


if __name__ == "__main__":
    agent_main(MyAgent())
