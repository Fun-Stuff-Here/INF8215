"""
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal

Recherche d'arbre de Monte Carlo
https://www.analyticsvidhya.com/blog/2019/01/monte-carlo-tree-search-introduction-algorithm-deepmind-alphago/
"""

from time import time as python_time
# from numba import njit, cfunc, objmode
# from numba.types import boolean, float64
from numpy import abs as np_abs, inf
from monte_carlo_tree_node import MCTS_Node as Node

# @njit()
def time():
    """
    Returns the time in seconds
    """
    # with objmode(current_time=float64):
    #     current_time = python_time()
    return python_time()

# @cfunc(boolean(float64, float64))
def time_condition(start_time:float, current_time:float):
    """
    This function is used to stop the monte carlo tree search
    Returns true if the time is up
    """
    return np_abs(current_time - start_time) > 6.0 #seconds

# @njit()
def monte_carlo_tree_search(board, player:int, step:int, time_left:int):
    """
    Returns best action from monte-carlo tree search
    """
    root = Node(board, None, player)
    return monte_carlo_algo(root, player, time_condition, step)

# @njit()
def tree_policy(node:Node, player:int):
    """
    select the node the maximize the UCB score
    """
    children:list[Node] = node.children
    best_child_found:Node = None
    upper_confidence_bound = -inf
    for child in children:
        uct = child.UCT(player)
        if uct > upper_confidence_bound:
            upper_confidence_bound = uct
            best_child_found = child

    if best_child_found is None:
        return node

    return best_child_found

# @njit()
def best_action(root: Node, player:int, step:int):
    """
    returns the best action to take
    """

    print("n_simulations", root.n_simulations)

    best_child:Node = root.best_child()
    if best_child is None:
        return root.rollout_policy(root.state, player, step)
    return best_child.state.last_action

# @njit()
def monte_carlo_algo(root:Node, player: int, stop_condition, step:int):
    """
    Hold the algorithm of monte-carlo tree search
    """
    start_time = time()
    root.expand()

    while not stop_condition(start_time, time()):
        current_node = root
        while not current_node.is_leaf:
            current_node = tree_policy(current_node, player)

        if current_node.n_simulations != 0:
            current_node = current_node.expand()

        utility = current_node.rollout(step)
        current_node.backpropagate(utility)

    return best_action(root, player, step)
