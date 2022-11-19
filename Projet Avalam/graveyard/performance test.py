"""
Performance tests
"""
import timeit

if __name__ == "__main__":
    import_module_compiled = "from njitavalam import Board"
    import_module_normal = "from avalam import Board"
    testcode = '''
def test():
    #test njit avalam
    board = Board()
    print(board.__str__())
    towers = board.get_towers()
    is_action_valid = board.is_action_valid((2,3,4,2))
    towerAction = board.get_tower_actions(4,4)
    actions = board.get_actions()
    is_movable = board.is_tower_movable(3,4)
    towers_action = board.get_actions()
    board.play_action((2,3,4,2))
    board2 = board.clone()
    percepts2 = board.get_percepts()
    is_finish = board.is_finished()
    score = board.get_score()
'''
    NUMBER = int(1e6 + 1e7/2)
    print(f" Not Compiled avalam = {max(timeit.repeat(stmt=testcode, setup=import_module_normal, repeat=30, number=NUMBER))} s")
    print(f" Compiled avalam = {max(timeit.repeat(stmt=testcode, setup=import_module_compiled, repeat=30, number=NUMBER))} s")