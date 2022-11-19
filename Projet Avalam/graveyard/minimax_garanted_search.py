from numpy import inf, array, int64, absolute
from numba import njit
from njitavalam import Board as AvalamState, RED
from graveyard.heuristics import heuristic_isolation
from joblib import Memory

memory = Memory("cachedir", verbose=1, bytes_limit=1e9)
@memory.cache
def alpha_beta_pruning_search(percepts:dict, player:int, cutoff_depth:int, step: int):
    """
    Alpha-Beta Pruning search
    :param percepts: dictionary representing the current board
    :param player: the player to control in this step (-1 or 1)
    :param cutoff_depth: the depth at which the search will be cutoff
    :return: the best move
    """
    board_array = array(percepts['m'], dtype=int64)
    return alpha_beta_pruning_algo(AvalamState(board_array, percepts['max_height']), player, cutoff_depth, step)

def heuristic(state:AvalamState, player:int, step: int):
    """
    Heuristic function
    :param state: the current state
    :param player: the player to control in this step (-1 or 1)
    :return: the heuristic value
    """
    return heuristic_isolation(state, player)

def max_value(state:AvalamState, player:int, alpha:int, beta:int, depth:int, cutoff_depth:int, step:int):
    """
    Max value function for alpha beta pruning yellow percpective
    """
    if state.is_finished():
        return state.get_score(), None
    if depth > cutoff_depth and is_quiescent(player, state):
        return heuristic(state, player, step), None
    depth += 1
    best_score = -inf
    best_move = None

    for action in state.get_actions():
        new_state = state.clone().play_action(action)
        score, _ = min_value(new_state, player, alpha, beta, depth, cutoff_depth, step + 1)
        if score > best_score:
            best_score = score
            best_move = action
            alpha = max(alpha, best_score)
        if best_score >= beta:
            return best_score, best_move
    return best_score, best_move

def min_value(state:AvalamState, player:int, alpha:int, beta:int, depth:int, cutoff_depth:int, step:int):
    """
    Min value function for alpha beta pruning red percepctive
    """
    if state.is_finished():
        return state.get_score(), None
    if depth > cutoff_depth and is_quiescent(player, state):
        return heuristic(state, player, step), None
    depth += 1
    best_score = inf
    best_move = None

    for action in state.get_actions():
        new_state = state.clone().play_action(action)
        score, _ = max_value(new_state, player, alpha, beta, depth, cutoff_depth, step + 1)
        if score < best_score:
            best_score = score
            best_move = action
            beta = min(beta, best_score)
        if best_score <= alpha:
            return best_score, best_move
    return best_score, best_move

def alpha_beta_pruning_algo(state:AvalamState, player:int, cutoff_depth:int, step:int):
    """
    Alpha-Beta Pruning search
    :param state: avalam board
    :param player: the player to control in this step (-1 or 1)
    :param cutoff_depth: the depth at which the search will be cutoff
    :return: the best move
    """
    if player == RED:
        return min_value(state, player, -inf, inf, 0, cutoff_depth, step)[1]
    return max_value(state, player, -inf, inf, 0, cutoff_depth, step)[1]

def is_quiescent(player: int, state: AvalamState) -> bool:
    """
    Check if the state is quiescent
    """
    return len(state.get_actions()) > 10