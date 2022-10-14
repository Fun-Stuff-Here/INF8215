"""
Minimax with alpha beta pruning.
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal
"""

from numpy import inf, array, int64
from numba import njit
from njitavalam import Board as AvalamState, RED

def alpha_beta_pruning_search(percepts:dict, player:int, cutoff_depth:int):
    """
    Alpha-Beta Pruning search
    :param percepts: dictionary representing the current board
    :param player: the player to control in this step (-1 or 1)
    :param cutoff_depth: the depth at which the search will be cutoff
    :return: the best move
    """
    board_array = array(percepts['m'], dtype=int64)
    return alpha_beta_pruning_algo(AvalamState(board_array, percepts['max_height']), player, cutoff_depth)

@njit()
def heuristic(state:AvalamState, player:int):
    """
    Heuristic function
    :param state: the current state
    :param player: the player to control in this step (-1 or 1)
    :return: the heuristic value
    """
    return state.get_score()

@njit()
def max_value(state:AvalamState, player:int, alpha:int, beta:int, depth:int, cutoff_depth:int):
    """
    Max value function for alpha beta pruning yellow percpective
    """
    if state.is_finished():
        return state.get_score(), None
    if depth > cutoff_depth:
        return heuristic(state, player), None
    depth += 1
    best_score = -inf
    best_move = None

    for action in state.get_actions():
        new_state = state.clone().play_action(action)
        score, _ = min_value(new_state, player, alpha, beta, depth, cutoff_depth)
        if score > best_score:
            best_score = score
            best_move = action
            alpha = max(alpha, best_score)
        if best_score >= beta:
            return best_score, best_move
    return best_score, best_move

@njit()
def min_value(state:AvalamState, player:int, alpha:int, beta:int, depth:int, cutoff_depth:int):
    """
    Min value function for alpha beta pruning red percepctive
    """
    if state.is_finished():
        return state.get_score(), None
    if depth > cutoff_depth:
        return heuristic(state, player), None
    depth += 1
    best_score = inf
    best_move = None

    for action in state.get_actions():
        new_state = state.clone().play_action(action)
        score, _ = max_value(new_state, player, alpha, beta, depth, cutoff_depth)
        if score < best_score:
            best_score = score
            best_move = action
            beta = min(beta, best_score)
        if best_score <= alpha:
            return best_score, best_move
    return best_score, best_move

@njit()
def alpha_beta_pruning_algo(state:AvalamState, player:int, cutoff_depth:int):
    """
    Alpha-Beta Pruning search
    :param state: avalam board
    :param player: the player to control in this step (-1 or 1)
    :param cutoff_depth: the depth at which the search will be cutoff
    :return: the best move
    """
    if player == RED:
        return min_value(state, player, -inf, inf, 0, cutoff_depth)[1]
    return max_value(state, player, -inf, inf, 0, cutoff_depth)[1]
