from avalam import Board
from my_player import MyAgent
from random_actions import random_action
import math

class AlphaBetaPruningAgent(MyAgent):

    def __init__(self):
        self.cutoff_depth = 6

    def play(self, percepts, player, step, time_left):
        if time_left < 2.0:
            return random_action(percepts)
        try:
            return self.alpha_beta_pruning_search(Board(percepts['m'], percepts['max_height']), player)[1]
        except Exception as error:  # pylint: disable=broad-except
            print(error)
            return random_action(percepts)

    def alpha_beta_pruning_search(self, board, player):
        return self.max_value(board, player, -math.inf, math.inf, 0)

    def max_value(self, board, player, alpha, beta, depth):
        if board.is_finished():
            return self.get_player_absolute_score(board, player), None
        if depth > self.cutoff_depth:
            return self.heuristic(board, player), None
        depth += 1
        best_score = -math.inf
        best_move = None

        for action in board.get_actions():
            new_board = board.clone()
            new_board.play_action(action)
            score, _ = self.min_value(new_board, player, alpha, beta, depth)
            if score > best_score:
                best_score = score
                best_move = action
                alpha = max(alpha, best_score)
            if best_score >= beta:
                return best_score, best_move
        return best_score, best_move

    def min_value(self, board, player, alpha, beta, depth):
        if board.is_finished():
            return self.get_player_absolute_score(board, player), None
        if depth > self.cutoff_depth:
            return self.heuristic(board, player), None
        depth += 1
        best_score = math.inf
        best_move = None

        for action in board.get_actions():
            new_board = board.clone()
            new_board.play_action(action)
            score, _ = self.min_value(new_board, player, alpha, beta, depth)
            if score < best_score:
                best_score = score
                best_move = action
                beta = min(beta, best_score)
            if best_score <= alpha:
                return best_score, best_move
        return best_score, best_move


    def heuristic(self, board, player):
        return self.get_player_absolute_score(board, player)

    #Rouge = 1 Jaune = -1
    def get_player_absolute_score(self, board, player):
        return board.get_score() if player == 1 else -board.get_score()
        
