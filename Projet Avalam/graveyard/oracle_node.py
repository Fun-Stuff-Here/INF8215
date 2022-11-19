"""
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal

Noeud de l'oracle

"""

from njitavalam import Board as AvalamState

class OracleNode: # pylint: disable=invalid-name
    """
    Noeud de monte carlo tree search
    """
    def __init__(self, state: AvalamState, parent):
        self.state = state
        self.parent = parent
        self.children: dict = dict()
        self.utility_distribution = dict()

    def expand(self):
        """
        expand the node by adding all possible children
        """
        self.children = {action: OracleNode(self.state.clone().play_action(action), self) for action in self.state.get_actions()}
        return self.children

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
