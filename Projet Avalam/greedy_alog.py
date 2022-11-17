"""
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal

greedy_alog.py
"""
from numba import njit, objmode, int64
from numba.types import Tuple, ListType, DictType, float64
from numba.typed import Dict
from random import random
from numpy import inf
from njitavalam import Board

action_type = Tuple([int64, int64, int64, int64])

@njit()
def predict_score(board:Board, action:tuple[int,int,int,int], player:int):
    x_from, y_from, x_to, y_to = action
    h_from = board.m[x_from][y_from]
    h_to = board.m[x_to][y_to]
    h = abs(h_from) + abs(h_to)
    if h==5 and h_from * player > 0:
        return inf
    if h_from < 0:
        h = -h
    if h_to * player < 0:
        h = h + 1 *player
    return h

@njit(cache=True)
def srt_player(index:int, player:int):
    srt_player1 = {b: i for i, b in enumerate([5,4,3,2,-2,-3,-4,-5])}
    srt_player2 = {b: i for i, b in enumerate([-5,-4,-3,-2,2,3,4,5])}
    return srt_player1[index] if player == 1 else srt_player2[index]

@njit(cache=True)
def greedy_action(board: Board, player:int):
    """
    Returns a random action from the board
    """
    actions = board.get_actions()
    action_max_found = actions[0]
    max_value = -inf
    if random() < 0.5:
        for action in actions:
            h = predict_score(board, action, player) * player
            if h > max_value:
                max_value = h
                action_max_found = action
    return action_max_found
