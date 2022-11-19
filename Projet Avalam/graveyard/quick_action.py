"""
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal

action la plus prometteuse selon une heuristique
"""

from numba import njit
from numba.typed import List # pylint: disable=no-name-in-module
from numpy import absolute as np_absolute
from njitavalam import Board as AvalamState
from graveyard.random_actions import choose_random_actions

@njit()
def is_tower_isolated(state:AvalamState, action:tuple[int,int,int,int]) -> bool:
    """
    Returns true if the tower is isolated
    """
    x_from, y_from, x_to, y_to = action
    tower_actions = List(state.get_tower_actions(x_to, y_to))
    if len(tower_actions) > 1:
        return False
    return tower_actions[0] == (x_to, y_to, x_from, y_from)

@njit()
def quick_action(state:AvalamState, player:int):
    """
    Returns an action by following heuristic
    Rule 1: si il est possible de faire une tour de 5
    Rule 2: Isoler une tour de notre couleur
    Rule 3: Bouger le pion de la couleur adverse sur un autre pion adverse si c'est le debut de partie

    Une action est un tuple contenant 4 éléments (i1, j1, i2, j2), définis comme suit :
        — i1 (int) : coordonnée en ligne (de gauche à droite) de la tour à déplacer.
        — j1 (int) : coordonnée en colonne (de haut en bas) de la tour à déplacer.
        — i2 (int) : coordonnée en ligne (de gauche à droite) de la tour à recouvrir.
        — j2 (int) : coordonnée en colonne (de haut en bas) de la tour à recouvrir
    ( x_from, y_from, x_to, y_to )
    """

    actions_statisfy_rule2 = []
    actions_statisfy_rule3 = []
    actions = List(state.get_actions())

    for action in actions:
        x_from, y_from, x_to, y_to = action
        # Rule 1: si il est possible de faire une tour de 5
        if (state.m[x_from][y_from] == player and
            np_absolute(state.m[x_from][y_from] + state.m[x_to][y_to]) == state.max_height):
            return action

        # Rule 2: Isoler une tour de notre couleur
        if state.m[x_from][y_from] == player and is_tower_isolated(state, action):
            actions_statisfy_rule2.append(action)

        # Rule 3: Bouger le pion de la couleur adverse sur un autre pion adverse si c'est le debut de partie
        if ( len(actions) < 254 and
            state.m[x_from][y_from] * player == 1 and state.m[x_to][y_to] * player == 1 ):
            actions_statisfy_rule3.append(action)

    if len(actions_statisfy_rule2) > 0:
        return actions_statisfy_rule2[0]

    if len(actions_statisfy_rule3) > 0:
        return actions_statisfy_rule3[0]

    return choose_random_actions(actions)
