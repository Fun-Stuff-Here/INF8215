"""
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal

Recherche d'arbre de Monte Carlo
https://www.analyticsvidhya.com/blog/2019/01/monte-carlo-tree-search-introduction-algorithm-deepmind-alphago/
"""

from time import time
from numpy import abs as np_abs, inf, floor
from monte_carlo_tree_node import MCTS_Node as Node

TURN_REPARTITION = {
    1: 5,
    2: 35,
    3: 120,
    4: 100,
    5: 100,
    6: 100,
    7: 100,
    8: 75,
    9: 75,
    10: 60,
    11: 30,
    12: 30,
    13: 30,
    14: 15,
    15: 10,
    16: 5,
    17: 3,
    18: 3,
    19: 2,
    20: 2
}

def time_condition(start_time:float, current_time:float, time_left:int):
    """
    This function is used to stop the monte carlo tree search
    Returns true if the time is up
    """
    return np_abs(current_time - start_time) > (time_left - 1)

def monte_carlo_tree_search(board, player:int, step:int, time_left:int):
    """
    Returns best action from monte-carlo tree search
    """
    turn_number = floor(step/2) + step%2
    if turn_number in TURN_REPARTITION:
        time_left = TURN_REPARTITION[turn_number]

    root = Node(board, None, player)
    return monte_carlo_algo(root, player, time_condition, step, time_left)

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

def best_action(root: Node, player:int, step:int):
    """
    returns the best action to take
    """

    best_child:Node = root.best_child()
    if best_child is None:
        return root.rollout_policy(root.state, player, step)
    return best_child.state.last_action

def monte_carlo_algo(root:Node, player: int, stop_condition, step:int, time_left:int):
    """
    Hold the algorithm of monte-carlo tree search
    """
    start_time = time()
    root.expand()

    while not stop_condition(start_time, time(), time_left):
        current_node = root
        while not current_node.is_leaf:
            current_node = tree_policy(current_node, player)

        if current_node.n_simulations != 0:
            current_node = current_node.expand()

        utility = current_node.rollout(step)
        current_node.backpropagate(utility)

    return best_action(root, player, step)
