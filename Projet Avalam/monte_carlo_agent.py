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
from avalam import Agent, agent_main
from njitavalam import Board, PLAYER1
from random_actions import random_action
from monte_carlo_tree_search import monte_carlo_tree_search
import traceback

class MyAgent(Agent):

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
            action = monte_carlo_tree_search(Board(board_array, percepts['max_height']), player, step, time_left)
            if board_copy.is_action_valid(action):
                return action
            raise Exception("Invalid action")
        except Exception as error:  # pylint: disable=broad-except
            traceback.print_exc()
            return random_action(board_copy)

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
    my_agent.play(percepts=percepts, player=PLAYER1, step=1, time_left=900)
    try:
        agent_main(my_agent)
    except Exception as error:
        print(error)
        print("Error in agent_main")
