"""
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal

Noeud de monte carlo tree search

https://github.com/int8/monte-carlo-tree-search/blob/master/mctspy/tree/nodes.py
https://stackoverflow.com/questions/60189128/n-ary-trees-in-numba
https://www.geeksforgeeks.org/left-child-right-sibling-representation-tree/
https://en.wikipedia.org/wiki/Left-child_right-sibling_binary_tree
"""
from numba import int64, deferred_type, optional
from numba.experimental import jitclass
from numpy import sqrt, inf, log
from avalam import Board
from njitavalam import YELLOW, RED, Board as AvalamState
from random_actions import choose_random_actions

node_type = deferred_type()

@jitclass([
    ('state', AvalamState.class_type.instance_type), # pylint: disable=no-member
    ('parent', optional(node_type)),
    ('child', optional(node_type)),
    ('next', optional(node_type)),
    ('utility', int64),
    ('n_simulations', int64),
])
class MCTS_Node: # pylint: disable=invalid-name
    """
    Noeud de monte carlo tree search
    """
    def __init__(self, state: AvalamState, parent):
        self.state = state
        self.parent = parent
        self.child = None
        self.next = None
        self.utility = 0
        self.n_simulations = 0

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
                sibling = MCTS_Node(next_state, self)
                self.child = sibling
                for action in actions[1:]:
                    next_state = self.state.clone().play_action(action)
                    sibling.next = MCTS_Node(next_state, self)
                    sibling = sibling.next
        return self.child

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
        return ( (self.utility * player) / self.n_simulations +
                sqrt(2) * sqrt(2 * log(self.parent.n_simulations) / self.n_simulations))

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

    def rollout_policy(self, possible_moves):
        """
        Random policy used for the rollout
        """
        return choose_random_actions(possible_moves)

    def rollout(self):
        """
        Simulate a playout from this node
        """
        current_rollout_state: Board = self.state.clone()
        while not current_rollout_state.is_finished():
            possible_moves = current_rollout_state.get_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.play_action(action)
        return current_rollout_state.get_score()

node_type.define(MCTS_Node.class_type.instance_type) # pylint: disable=no-member

if __name__ == "__main__":
    state1 = AvalamState()
    n = MCTS_Node(state1, None)
