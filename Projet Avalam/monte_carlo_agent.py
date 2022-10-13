"""
Avalam agent.
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
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

Recherche d'arbre de Monte Carlo
"""
from numpy import array, int64
from numba import njit
from avalam import Agent, agent_main
from njitavalam import Board, PLAYER1
from random_actions import random_action
from monte_carlo_tree_search import monte_carlo_tree_search

def play(board, player:int, step:int, time_left:int):
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
        return random_action(board)

    # Monte Carlo Tree Search
    try:
        return monte_carlo_tree_search(board, player, step, time_left)
    except Exception as error: #pylint: disable=broad-except
        print(error)
        return random_action(board)

class MyAgent(Agent):

    """My Avalam agent."""

    def play(self, percepts, player, step, time_left):
        board_array = array(percepts['m'], dtype=int64)
        return play(Board(board_array, percepts['max_height']), player, step, time_left)

if __name__ == "__main__":
    my_agent = MyAgent()
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
    action = my_agent.play(percepts=percepts, player=PLAYER1, step=1, time_left=900)
    agent_main(my_agent)
