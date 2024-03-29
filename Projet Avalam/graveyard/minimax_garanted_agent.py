"""
Avalam agent using minimax.
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal
"""

from numpy import array, int64
from graveyard.minimax_garanted_search import alpha_beta_pruning_search
from graveyard.random_actions import random_action
from avalam import Agent, agent_main
from njitavalam import PLAYER1, Board

class AlphaBetaPruningAgent(Agent):
    """
     Alpha-Beta Pruning agent
    """

    def __init__(self):
        self.cutoff_depth = 2

    def update_cutoff_depth(self, step:int):
        """
        Update cutoff depth
        """
        if step <= 8:
            self.cutoff_depth = 1
        elif step <= 20:
            self.cutoff_depth = 2
        elif step <= 30:
            self.cutoff_depth = 3
        elif step <= 36:
            self.cutoff_depth = 4
        else:
            self.cutoff_depth = 5

    def play(self, percepts:dict, player:int, step:int, time_left:int):
        """
        Play a move
        :param percepts: dictionary representing the current board
        :param player: the player to control in this step (-1 or 1)
        :param step: the current step
        :param time_left: the time left for the agent to play
        :return: the action to play
        """
        print("step : ", step, "\n Time left : ", time_left)
        board_array = array(percepts['m'], dtype=int64)
        board_copy = Board(board_array, percepts['max_height'])
        try:
            if time_left < 2.0:
                raise Exception("not enough time left")
            self.update_cutoff_depth(step)
            action = alpha_beta_pruning_search(percepts, player, self.cutoff_depth, step)
            if board_copy.is_action_valid(action):
                return action
            raise Exception("Invalid action")
        except Exception as error:  # pylint: disable=broad-except
            print(error)
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
    my_agent.play(percepts=percepts, player=PLAYER1, step=1, time_left=900)
    try:
        agent_main(my_agent)
    except Exception as error:
        print(error)
        print("Error in agent_main")