"""
Heuristics functions for the Avalam algorithm.
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal
"""

from numba import njit
from njitavalam import Board as AvalamState
from numpy import array, absolute


def heuristic_isolation(state:AvalamState, player:int)->int:
    """
    Heuristic function
    :param state: the current state
    :param player: the player to control in this step (-1 or 1)
    :return: the heuristic value
    """

    # If the game is finished, return the score
    if state.is_finished():
        return state.get_score()

    # If the game is not finished, return the number of isolated cells
    f = lambda x: 5 - absolute(x)
    isolation_factor = f(array(state.m))
    towers = state.get_towers()
    isolated_towers_heights = []
    for tower in towers:
        i, j, h = tower
        h_abs = absolute(h)
        if i != 0 and j !=0:
            if isolation_factor[i-1, j-1] > h_abs:
                isolated_towers_heights.append(h)
                continue
        if i != 0:
            if isolation_factor[i-1, j] > h_abs:
                isolated_towers_heights.append(h)
                continue
        if i != 0 and j != 8:
            if isolation_factor[i-1, j+1] > h_abs:
                isolated_towers_heights.append(h)
                continue
        if j != 0:
            if isolation_factor[i, j-1] > h_abs:
                isolated_towers_heights.append(h)
                continue
        if j != 8:
            if isolation_factor[i, j+1] > h_abs:
                isolated_towers_heights.append(h)
                continue
        if i != 8 and j != 0:
            if isolation_factor[i+1, j-1] > h_abs:
                isolated_towers_heights.append(h)
                continue
        if i != 8:
            if isolation_factor[i+1, j] > h_abs:
                isolated_towers_heights.append(h)
                continue
        if i != 8 and j != 8:
            if isolation_factor[i+1, j+1] > h_abs:
                isolated_towers_heights.append(h)
                continue

    approximative_score = 0
    for h in isolated_towers_heights:
        if h < 0:
            approximative_score -= 1
        else:
            approximative_score += 1
    return approximative_score

def get_all_moves_involving_tower(i: int, j: int, state: AvalamState):
    tower_actions = state.get_tower_actions(i, j)
    opposite_actions = []
    for action in tower_actions:
        opposite_action = (action[2], action[3], action[0], action[1])
        # No need to validate action technically
        opposite_actions.append(opposite_action)
    return tower_actions + opposite_actions

def get_all_player_towers(state: AvalamState, player: int):
    towers = state.get_towers()
    player_towers = []
    for tower in towers:
        if tower[2] > 0 and player == 1:
            player_towers.append(tower)
        elif tower[2] < 0 and player == -1:
            player_towers.append(tower)
    return player_towers

def is_player_tower_buried_by_action(state: AvalamState, player: int, action):
    if state.m[action[2][3]] > 0 and player == 1:
        return True
    if state.m[action[2][3]] < 0 and player == -1:
        return True

def heuristic_1(state: AvalamState, player: int)->float:
    TOWER_HEIGHT_INDEX = 2
    SCORE_FOR_MAX_TOWER = 1.2
    estimated_score = 0
    for player_tower in get_all_player_towers(state, player):
        if player_tower[TOWER_HEIGHT_INDEX] == player * state.max_height:
            estimated_score += SCORE_FOR_MAX_TOWER
        elif player * player_tower[TOWER_HEIGHT_INDEX] < 3 and not state.is_tower_movable(player_tower[0], player_tower[1]):
            estimated_score += 1.7
        else:
            nb_moves_burying_player_tower = 0
            nb_moves_burying_opponent_tower = 0
            for move in get_all_moves_involving_tower(player_tower[0], player_tower[1], state):
                #if player tower is buried

                # Player 1
                # -1 over -1 -> -1 TECHNICALLY NOT POSSIBLE
                # -1 over 1 -> -1 true
                # 1 over -1 -> 1 false
                # 1 over 1 -> 1 true

                # Player -1
                # -1 over -1 -> -1 true
                # -1 over 1 -> -1 false
                # 1 over -1 -> 1 true
                # 1 over 1 -> 1 TECHNICALLY NOT POSSIBLE
                if is_player_tower_buried_by_action(state, player, move):
                    nb_moves_burying_player_tower += 1
                else:
                    nb_moves_burying_opponent_tower += 1
               
            estimated_score += 0.8 if nb_moves_burying_player_tower >= nb_moves_burying_opponent_tower else 1.4
    return estimated_score

def heuristic_2(state: AvalamState, player: int)->float:
    player_estimated_score = heuristic_1(state, player)
    opponent_estimated_score = heuristic_1(state, -1 * player)
    real_score = state.get_score()
    return real_score + player * 2 if player_estimated_score > opponent_estimated_score else real_score
"""
def heuristic_3(state: AvalamState, player: int, step: int)->float:
    TOWER_HEIGHT_INDEX = 2
    SCORE_FOR_MAX_TOWER = 1.2
    estimated_score = 0
    for player_tower in get_all_player_towers(state, player):
        if player_tower[TOWER_HEIGHT_INDEX] == player * state.max_height:
            print("Youpi")
            # Check if there's a potential Tie, if yes give more importance to tower of 5
        elif player_tower[TOWER_HEIGHT_INDEX] == player * 4 and :
            print("Houra")
            # 
"""    
