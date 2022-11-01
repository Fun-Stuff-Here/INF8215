# first line: 14
@memory.cache
def alpha_beta_pruning_search(percepts:dict, player:int, cutoff_depth:int):
    """
    Alpha-Beta Pruning search
    :param percepts: dictionary representing the current board
    :param player: the player to control in this step (-1 or 1)
    :param cutoff_depth: the depth at which the search will be cutoff
    :return: the best move
    """
    board_array = array(percepts['m'], dtype=int64)
    return alpha_beta_pruning_algo(AvalamState(board_array, percepts['max_height']), player, cutoff_depth)
