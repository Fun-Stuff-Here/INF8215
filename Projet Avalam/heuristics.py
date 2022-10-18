"""
Heuristics functions for the Avalam algorithm.
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal
"""

from numba import njit
from njitavalam import Board as AvalamState
from numpy import array, absolute


def heuristic_isolation(state:AvalamState, player:int)->int:
    """
    Heuristic function
    :param state: the current state
    :param player: the player to control in this step (-1 or 1)
    :return: the heuristic value
    """

    # If the game is finished, return the score
    if state.is_finished():
        return state.get_score()

    # If the game is not finished, return the number of isolated cells
    f = lambda x: 5 - absolute(x)
    isolation_factor = f(array(state.m))
    towers = state.get_towers()
    isolated_towers_heights = []
    for tower in towers:
        i, j, h = tower
        h_abs = absolute(h)
        if i != 0 and j !=0:
            if isolation_factor[i-1, j-1] > h_abs:
                isolated_towers_heights.append(h)
                continue
        if i != 0:
            if isolation_factor[i-1, j] > h_abs:
                isolated_towers_heights.append(h)
                continue
        if i != 0 and j != 8:
            if isolation_factor[i-1, j+1] > h_abs:
                isolated_towers_heights.append(h)
                continue
        if j != 0:
            if isolation_factor[i, j-1] > h_abs:
                isolated_towers_heights.append(h)
                continue
        if j != 8:
            if isolation_factor[i, j+1] > h_abs:
                isolated_towers_heights.append(h)
                continue
        if i != 8 and j != 0:
            if isolation_factor[i+1, j-1] > h_abs:
                isolated_towers_heights.append(h)
                continue
        if i != 8:
            if isolation_factor[i+1, j] > h_abs:
                isolated_towers_heights.append(h)
                continue
        if i != 8 and j != 8:
            if isolation_factor[i+1, j+1] > h_abs:
                isolated_towers_heights.append(h)
                continue

    approximative_score = 0
    for h in isolated_towers_heights:
        if h < 0:
            approximative_score -= 1
        else:
            approximative_score += 1
    return approximative_score
