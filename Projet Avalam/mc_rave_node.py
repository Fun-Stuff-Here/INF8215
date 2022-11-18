"""
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal

Noeud de monte carlo tree search

https://github.com/int8/monte-carlo-tree-search/blob/master/mctspy/tree/nodes.py
https://stackoverflow.com/questions/60189128/n-ary-trees-in-numba
https://www.geeksforgeeks.org/left-child-right-sibling-representation-tree/
https://en.wikipedia.org/wiki/Left-child_right-sibling_binary_tree
"""
from numba import int64, deferred_type, optional, objmode
from numba.types import Tuple, DictType
from numba.experimental import jitclass
from numba.typed import Dict, List
from numpy import sqrt, inf, log
from avalam import Board
from njitavalam import YELLOW, RED, Board as AvalamState
from random_actions import random_action
from greedy_alog import greedy_action
from heuristics import heuristic_isolation

#node_type = deferred_type()
action_type = Tuple([int64, int64, int64, int64])
dict_item_type = Tuple([int64, int64])
dict_type = DictType(action_type, dict_item_type)

# @jitclass([
#     ('state', AvalamState.class_type.instance_type), # pylint: disable=no-member
#     ('parent', optional(node_type)),
#     ('child', optional(node_type)),
#     ('next', optional(node_type)),
#     ('utility', int64),
#     ('n_simulations', int64),
#     ('player', int64),
#     ('action_dict', dict_type),
#     ('action', action_type)
# ])
class MCTS_Rave_Node: # pylint: disable=invalid-name
    """
    Noeud de monte carlo tree search
    """
    def __init__(self, state: AvalamState, parent, player:int, action_played:action_type, action_dict:dict_type):
        self.state = state
        self.parent = parent
        self.child = None
        self.next = None
        self.utility = 0
        self.n_simulations = 0
        self.player = player
        self.action_dict = action_dict
        self.action = action_played

    @property
    def children(self):
        """
        Returns a list of all children of this node
        """
        children = []
        child = self.child
        while child is not None:
            children.append(child)
            child = child.next
        return children

    def expand(self):
        """
        expand the node by adding all possible children
        """
        if self.child is None:
            actions = self.state.get_actions()
            if len(actions) > 0:
                next_state = self.state.clone().play_action(actions[0])
                sibling = MCTS_Rave_Node(next_state, self, self.player * -1,actions[0], self.action_dict)
                self.child = sibling
                for action in actions[1:]:
                    next_state = self.state.clone().play_action(action)
                    sibling.next = MCTS_Rave_Node(next_state, self, self.player * -1, action, self.action_dict)
                    sibling = sibling.next
                return self.child
        return self

    @property
    def is_leaf(self):
        """
        Returns True if the node is a leaf
        """
        return self.child is None

    @property
    def is_root(self):
        """
        Returns True if the node is the root
        """
        return self.parent is None

    def UCT(self, player: RED|YELLOW): # pylint: disable=invalid-name
        """
        Returns the UCT value of the node
        On multiplie par le player pour que si on est les rouges ont
        cherche la plus grande moyene (<0: red, >0: yellow, 0: draw)
        (negative for red, positive for yellow)
        """
        if self.n_simulations == 0 or self.is_root:
            return inf
        # biais
        b = sqrt(2)
        c = sqrt(2)

        N_action = 1
        U_action = 0
        if self.action in self.action_dict:
            action_info = self.action_dict[self.action]
            N_action = action_info[0]
            U_action = action_info[1]
        N_simulation = self.n_simulations
        U_simulation = self.utility

        # beta that minimize MSE
        beta = (N_action)/(N_simulation + N_action + 4*N_action*N_simulation*(b**2))
        mean_action = U_action/N_action
        mean_simulation = U_simulation/N_simulation
        combined_mean = (1-beta)*mean_action + beta*mean_simulation
        exploration_factor = c*sqrt(2 * log(self.parent.n_simulations) / self.n_simulations)
        return ( player * combined_mean ) + exploration_factor

    def backpropagate(self, utility):
        """
        backpropagate the utility of the simulation
        """
        self.increment(utility)

        if self.parent is not None:
            current_node = self.parent
            while current_node is not None:
                current_node.increment(utility)
                current_node = current_node.parent

    def increment(self, utility):
        """
        Increment the number of simulations of the node
        """
        self.n_simulations += 1
        self.utility += utility

    def best_child(self):
        """
        Returns the best child
        """
        if self.child is None:
            return None

        best_child_found = self.child
        max_n_simulations = best_child_found.n_simulations
        for child in self.children:
            if child.n_simulations > max_n_simulations:
                best_child_found = child
                max_n_simulations = child.n_simulations

        return best_child_found

    def rollout_policy(self, state:AvalamState, player:int, step:int):
        """
        policy used for the rollout
        """
        actions = state.get_actions()
        best_action = actions[0]
        best_value = -30
        for action in actions:
            next_state = state.clone().play_action(action)
            isolation_value = heuristic_isolation(next_state, player)
            value = player * isolation_value
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def rollout(self, step:int):
        """
        Simulate a playout from this node
        """
        current_rollout_state: Board = self.state.clone()
        current_player = self.player
        current_step = step
        actions = List()
        score = 0
        for _ in range(6):
            if current_rollout_state.is_finished():
                score = current_rollout_state.get_score()
                break
            action = self.rollout_policy(current_rollout_state, current_player, current_step)
            actions.append(action)
            current_rollout_state = current_rollout_state.play_action(action)
            current_player = current_player * -1
            current_step += 1
        else:
            score = heuristic_isolation(current_rollout_state, self.player)
        # update action_dict
        score = current_rollout_state.get_score()
        for action in actions:
            if action in self.action_dict:
                action_info = self.action_dict[action]
                self.action_dict[action] = (action_info[0] + 1, action_info[1] + score)
            else:
                self.action_dict[action] = (1, score)
        return score

#node_type.define(MCTS_Rave_Node.class_type.instance_type) # pylint: disable=no-member

if __name__ == "__main__":
    state1 = AvalamState()
    d = Dict.empty(key_type=Tuple([int64, int64, int64, int64]), value_type=Tuple([int64, int64]))
    n = MCTS_Rave_Node(state1, None, 1, (0,0,0,0), d)
