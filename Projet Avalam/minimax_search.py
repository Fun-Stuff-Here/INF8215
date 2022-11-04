"""
Minimax with alpha beta pruning.
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal
"""

from numpy import inf, array, int64, absolute
# from numba import njit
from njitavalam import Board as AvalamState, RED
from heuristics import heuristic_1, heuristic_2, heuristic_isolation
from functools import lru_cache
from joblib import Memory
from random import random
from minimax_optimization import IncrementalBoard, neighbor_cells_indexes

memory = Memory("cachedir", verbose=0, bytes_limit=1e9)
@memory.cache
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

def heuristic(state, player:int):
    """
    Heuristic function
    :param state: the current state
    :param player: the player to control in this step (-1 or 1)
    :return: the heuristic value
    """
    return heuristic_isolation(state, player)

def max_value(state, player:int, alpha:int, beta:int, depth:int, cutoff_depth:int):
    """
    Max value function for alpha beta pruning yellow percpective
    """
    if state.is_finished():
        return state.get_score(), None
    if depth > cutoff_depth and is_quiescent(player, state):
        return heuristic(state, player), None
    depth += 1
    best_score = -inf
    best_move = None

    for action in get_actions(player, state, depth):
        new_state = state.append(action)
        score, _ = min_value(new_state, player, alpha, beta, depth, cutoff_depth)
        state.pop()
        if score > best_score:
            best_score = score
            best_move = action
            alpha = max(alpha, best_score)
        if best_score >= beta:
            return best_score, best_move
    return best_score, best_move

def min_value(state, player:int, alpha:int, beta:int, depth:int, cutoff_depth:int):
    """
    Min value function for alpha beta pruning red percepctive
    """
    if state.is_finished():
        return state.get_score(), None
    if depth > cutoff_depth and is_quiescent(player, state):
        return heuristic(state, player), None
    depth += 1
    best_score = inf
    best_move = None

    for action in get_actions(player, state, depth):
        new_state = state.append(action)
        score, _ = max_value(new_state, player, alpha, beta, depth, cutoff_depth)
        state.pop()
        if score < best_score:
            best_score = score
            best_move = action
            beta = min(beta, best_score)
        if best_score <= alpha:
            return best_score, best_move
    return best_score, best_move

def alpha_beta_pruning_algo(state:AvalamState, player:int, cutoff_depth:int):
    """
    Alpha-Beta Pruning search
    :param state: avalam board
    :param player: the player to control in this step (-1 or 1)
    :param cutoff_depth: the depth at which the search will be cutoff
    :return: the best move
    """
    incremtal_board = IncrementalBoard(state)
    if player == RED:
        return min_value(incremtal_board, player, -inf, inf, 0, cutoff_depth)[1]
    return max_value(incremtal_board, player, -inf, inf, 0, cutoff_depth)[1]

def is_quiescent(player: int, state) -> bool:
    """
    Check if the state is quiescent
    """
    if state.last_action is None:
        return True

    x_from, y_from, x_to, y_to = state.last_action
    h = state.cell(x_to, y_to)
    h_abs = absolute(h)
    max_height = state.max_height

    # if the last action creates a tower of 4 it is not quiescent
    if h_abs == max_height - 1:
        return False
    return True

def get_actions(player:int, state:AvalamState, depth:int):
    """
    Get all the possible actions for a player with a given rule
    :param player: the player to control in this step (-1 or 1)
    :param state: the current state
    :return: the list of all the possible actions
    """
    #rule : don't consider the action that is only surrended by ones
    for x_from, y_from, x_to, y_to in state.get_actions():

        if absolute(state.cell(x_to, y_to)) > 1:
            yield (x_from, y_from, x_to, y_to)
            continue
        for x, y in neighbor_cells_indexes(x_to, y_to):
            if absolute(state.cell(x, y)) > 1:
                # reject 20% of the actions times depth
                if(random() > 0.30*depth):
                    yield (x_from, y_from, x_to, y_to)
                    break
        else:
            # accept action with only ones as neighboors with probability of 0.08
            if(random() < 0.08):
                yield (x_from, y_from, x_to, y_to)
