"""
Heuristics functions for the Avalam algorithm.
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal
"""

from numba import njit
from njitavalam import Board as AvalamState
from numpy import array, absolute


def basic_heuristic(state: AvalamState):
    return state.get_score()

def heuristic_isolation(state:AvalamState, player:int)->int:
    """
    Heuristic function
    :param state: the current state
    :param player: the player to control in this step (-1 or 1)
    :return: the heuristic value
    """

    # If the game is not finished, return the number of isolated cells
    estimated_score = 0
    for x, y, h in state.get_towers():
        if not state.is_tower_movable(x, y):
            if h > 0:
                estimated_score += 1
            else:
                estimated_score -= 1
    return estimated_score

def get_all_moves_involving_tower(i: int, j: int, state: AvalamState):
    actions = []
    for action in state.get_tower_actions(i, j):
        actions.append(action)
        opposite_action = (action[2], action[3], action[0], action[1])
        # No need to validate action technically
        actions.append(opposite_action)
    return actions

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
    buried_tower = state.m[action[2]][action[3]]
    if buried_tower > 0 and player == 1:
        return True
    if buried_tower < 0 and player == -1:
        return True
    return False

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

def heuristic_3(state: AvalamState, player: int, step: int)->float:
    TOWER_HEIGHT_INDEX = 2
    SCORE_FOR_MAX_TOWER = 1.2
    estimated_score = 0
    for player_tower in get_all_player_towers(state, player):

        if player_tower[TOWER_HEIGHT_INDEX] == player * state.max_height:
            abs_real_score = abs(state.get_score())
            # Check if there's potential for a tie (absolute real score is low and we are in midlle-end game)
            #   if yes, give more importance to tower of 5 of height
            if abs_real_score < 3 and step > 25:
                estimated_score += 1.5
            # Check if there is a small score difference and if we are in beginning of the game
            #   if yes, give importance to tower of 5
            elif abs_real_score < 3 and step < 15:
                estimated_score += 1.2
            # For any other tower of 5, just add 1 point to the estimated score
            else:
                estimated_score += 1
            continue

        # For non movable tower under 5 of heigh, add at leadt 1 to score since it's a garantied point
        # If the tower heigh 1, add more point to the estimated score
        elif player_tower[TOWER_HEIGHT_INDEX] <= player * 4 and not state.is_tower_movable(player_tower[0], player_tower[1]) :
            estimated_score += 1
            if player_tower[TOWER_HEIGHT_INDEX] == 1:
                estimated_score += 0.2
            continue
        
        # In other cases, evaluate the surroundings of the tower. 
        # Add points to the estimated score according to the tower in function of the number of moves involving this tower that burries 
        # one of the player's tower and in function of the number of moves involving this tower that burries 
        # one of the opponent's tower
        else:
            nb_moves_burying_player_tower = 0
            nb_moves_burying_opponent_tower = 0
            for move in get_all_moves_involving_tower(player_tower[0], player_tower[1], state):
                if is_player_tower_buried_by_action(state, player, move):
                    nb_moves_burying_player_tower += 1
                else:
                    nb_moves_burying_opponent_tower += 1

            # Could be evaluate in another way 
            diff = nb_moves_burying_opponent_tower - nb_moves_burying_player_tower
            estimated_score += 1 + diff / 10
                
    return estimated_score
    
