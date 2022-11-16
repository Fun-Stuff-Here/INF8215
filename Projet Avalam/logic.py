"""
Logic for logic agent
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal
"""

from numpy import array, int64
from njitavalam import Board

def get_action(percepts:dict, player:int, step:int, time_left:int)->tuple[int, int, int, int]:
    """
    Returns an action from the board
    """
    board_string = ''.join(''.join(map(str,i)) for i in percepts['m'])
    # flatten index of the board  i = x * 9 + y
    print(board_string)
    return (0,0,0,0)
