# Code from https://github.com/aimacode/aima-python

import math
import random

def random_player(game, state): 
    """A game player who plays randomly"""
    return random.choice(list(game.actions(state)))

def player(search_algorithm):
    """A game player who uses the specified search algorithm"""
    return lambda game, state: search_algorithm(game, state)[1]

def minimax_search(game, state):
    """Search game tree to determine best move; return (value, move) pair."""

    player = state.to_move

    def max_value(state):
        # TODO: include a recursive call to min_value function
        raise Exception("Function not implemented")

    def min_value(state):
        # TODO: include a recursive call to max_value function
        raise Exception("Function not implemented")
    

    return max_value(state)

infinity = math.inf

def alphabeta_search(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = state.to_move

    def max_value(state, alpha, beta):
        # TODO: include a recursive call to min_value function
        raise Exception("Function not implemented")

    def min_value(state, alpha, beta):
        # TODO: include a recursive call to max_value function
        raise Exception("Function not implemented")

    return max_value(state, -infinity, +infinity)


def cutoff_depth(d):
    """A cutoff function that searches to depth d."""
    return lambda game, state, depth: depth > d

def h_alphabeta_search(game, state, cutoff=cutoff_depth(6), h=lambda s , p: 0):
    """Search game to determine best action; use alpha-beta pruning.
    This version searches all the way to the leaves."""

    player = state.to_move

    def max_value(state, alpha, beta, depth):
        # TODO: include a recursive call to min_value function
        raise Exception("Function not implemented")

    def min_value(state, alpha, beta, depth):
        # TODO: include a recursive call to min_value function
        raise Exception("Function not implemented")

    
    return max_value(state, -infinity, +infinity, 0)


def your_nice_agent(game, state):
    return h_alphabeta_search(game, state, cutoff=cutoff_depth(1), h=your_nice_heuristic)

def your_nice_heuristic(state, player):
    # TODO: write your own heuristic
    return 0