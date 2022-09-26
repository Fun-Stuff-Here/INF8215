import sys
import os
import argparse
import time

from games import *
from players import *

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--scenario', type=str, default="")

    return parser.parse_args()
if __name__ == '__main__':
    
    args = parse_arguments()

    start_time = time.time()

    if args.scenario == "test":
        print("Random (X) vs Random (O) on tic-tac-toe")
        print("Only used to check that the code is working in your system")
        play_game(TicTacToe(), dict(X=random_player, O=random_player), verbose=True).utility
    
    elif args.scenario == "S1":
        print("Random (X) vs Minimax (O) on tic-tac-toe")
        print("You should observe that Minimax never loses to game (a draw is possible)")
        play_game(TicTacToe(), dict(X=random_player, O=player(minimax_search)), verbose=True).utility

    elif args.scenario == "S2":
        print("Random (X) vs Minimax (O) on Connect Four")
        print("You should observe that Minimax is very slow")
        print("It is because the search space is too large")
        print("You can stop the execution")
        play_game(ConnectFour(), dict(X=random_player, O=player(minimax_search)), verbose=True).utility

    elif args.scenario == "S3":
        print("Random (X) vs AlphaBeta (O) on tic-tac-toe")
        print("You should observe that AlphaBeta never loses to game (a draw is possible)")
        play_game(TicTacToe(), dict(X=random_player, O=player(alphabeta_search)), verbose=True).utility

    elif args.scenario == "S4":
        print("Random (X) vs AlphaBeta (O) on Connect Four")
        print("You should observe that AlphabetaBeta is very slow")
        print("It is because the search space is too large, even with pruning")
        print("You can stop the execution")
        play_game(ConnectFour(), dict(X=random_player, O=player(alphabeta_search)), verbose=True).utility

    elif args.scenario == "S5":
        print("minimax (X) vs AlphaBeta (O) on tic-tac-toe")
        play_game(TicTacToe(), dict(X=player(minimax_search), O=player(alphabeta_search)), verbose=True).utility

    elif args.scenario == "S6":
        print("Random (X) vs AlphaBeta with a cutoff (O) on Connect Four ")
        print("Cutoff is done at depth 6")
        play_game(ConnectFour(), dict(X=random_player,O=player(h_alphabeta_search)), verbose=True).utility

    elif args.scenario == "S7":
        print("Your agent (X) vs AlphaBeta with a cutoff (O) on Connect Four ")
        print("Cutoff is done at depth 6")
        play_game(ConnectFour(), dict(X=player(your_nice_agent),O=player(h_alphabeta_search)), verbose=True).utility

    end_time = time.time() - time.time()

    print("Elapsed: %.2f seconds" % (time.time() - start_time))
