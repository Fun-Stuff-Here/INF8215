"""
Avalam agent using minimax.
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal
"""

from numpy import array, int64
from minimax_search import alpha_beta_pruning_search
from random_actions import random_action
from avalam import Agent, agent_main
from njitavalam import PLAYER1, Board
import traceback

class AlphaBetaPruningAgent(Agent):
    """
     Alpha-Beta Pruning agent
    """

    def __init__(self):
        self.cutoff_depth = 1

    def update_cutoff_depth(self, step:int):
        """
        Update cutoff depth
        """
        return
        if step <= 10:
            self.cutoff_depth = 4
        elif step <= 22:
            self.cutoff_depth = 6
        elif step <= 34:
            self.cutoff_depth = 8

    def play(self, percepts:dict, player:int, step:int, time_left:int):
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
            self.update_cutoff_depth(step)
            action = alpha_beta_pruning_search(percepts, player, self.cutoff_depth)
            if board_copy.is_action_valid(action):
                return action
            raise Exception("Invalid action")
        except Exception as error:  # pylint: disable=broad-except
            traceback.print_exc()
            return random_action(board_copy)

if __name__ == "__main__":
    my_agent = AlphaBetaPruningAgent()
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
    
    # import cProfile
    # cProfile.run('my_agent.play(percepts=percepts, player=PLAYER1, step=1, time_left=900)', 'profile_results')
    # import pstats
    # file = open('formatted_profile.txt', 'w')
    # profile = pstats.Stats('profile_results', stream=file)
    # profile.sort_stats('cumulative') # Sorts the result according to the supplied criteria
    # profile.print_stats() # Prints the first 15 lines of the sorted report
    # file.close()
    try:
        agent_main(my_agent)
    except Exception as error:
        print(error)
        print("Error in agent_main")
