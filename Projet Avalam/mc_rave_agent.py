"""
Avalam agent.
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal

Recherche d'arbre de Monte Carlo avec RAVE
"""

from avalam import agent_main
from njitavalam import Board, PLAYER1
from random_actions import random_action
from mc_rave_algo import monte_carlo_tree_search
from time_safe_agent import TimeSafeAgent

class MonteCarloRaveAgent(TimeSafeAgent):
    """
     Agent based on monte carlo tree search
    """

    def get_action(self, board:Board, player:int, step:int, time_left:int)->tuple[int,int,int,int]:
        """
        Get an action
        :param percepts: dictionary representing the current board
        :param player: the player to control in this step (-1 or 1)
        :param step: the current step
        :param time_left: the time left for the agent to play
        :return: the action to play
        """
        return monte_carlo_tree_search(board, player, step, time_left)

if __name__ == "__main__":
    my_agent = MonteCarloRaveAgent()
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
