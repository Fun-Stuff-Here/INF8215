"""
Random actions for the agent
"""
from numpy.random import uniform
from numba import njit
from numba.types import int64
from avalam import Board

#@njit(locals={'index': int64}, cache = True)
def choose_random_actions(actions: list[tuple[int, int, int, int]]):
    """
    Returns a random action from the board
    """
    actions = list(actions)
    lenght = len(actions)
    if lenght == 0:
        return (0, 0, 0, 0)
    index = uniform(0, lenght-1, 1)[0]
    return actions[int(index)]

def random_action(board: Board):
    """
    Returns a random action from the board
    """
    actions = board.get_actions()
    return choose_random_actions(actions)

