"""
Random actions for the agent
"""
from random import choice
from numpy.random import uniform
from numba import njit
from numba.types import int64
from avalam import Board

@njit(locals={'index': int64})
def choose_random_actions(actions):
    """
    Returns a random action from the board
    """
    if len(actions) == 0:
        return (0, 0, 0, 0)
    index = int(uniform(0, len(actions)-1, 1)[0])
    return actions[index]

def random_action(board: Board):
    """
    Returns a random action from the board
    """
    return choice(board.get_actions())

def random_actions_alarm_handler(board: Board):
    """
    Handler for the alarm signal
    """
    def handler(_, __):
        return choice(board.get_actions())
    return handler
