import pickle
import numpy as np
from njitavalam import Board as AvalamState
from random_actions import choose_random_actions

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def main():
    board = np.array([  [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  1, -1,  1,  0,  0],
                        [ 0,  0,  0,  0, -1,  1, -1,  0,  0],
                        [ 0,  0,  0,  0,  1, -1,  1,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0,  0]])
    state = AvalamState(board)
    tree_size = 1
    while not state.is_finished():
        actions = state.get_actions()
        tree_size *= len(actions)
        state.play_action(choose_random_actions(actions))
    print(f"{tree_size}")

if __name__ == '__main__':
    main()
