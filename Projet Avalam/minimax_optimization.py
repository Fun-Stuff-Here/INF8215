"""
Optimization for minimax search with incremental board
Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
Polytechnique Montréal
"""

from numpy import zeros
from njitavalam import Board as AvalamState
from functools import lru_cache

NOT_IN_REAL_BOARD = {
    (0, 0), (0,1), (0, 4), (0, 5),(0, 6), (0, 7), (0, 8),
    (1, 0), (1, 5), (1, 6), (1, 7), (1, 8),
    (2,0), (2, 7), (2,8),
    (3, 0),
    (4, 4),
    (5, 8),
    (6, 0), (6, 1), (6, 8),
    (7, 0), (7, 1), (7, 2), (7, 3), (7, 8),
    (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 7), (8, 8)
}

@lru_cache(maxsize=1_000_000)
def neighbor_cells_indexes(i:int, j:int) -> list[tuple[int, int]]:
    """
    Return the neighbors indexes of tower (i,j)
    :param i: row index
    :param j: column index
    :return: list of neighbors indexes
    """
    neighbors = []
    i_min = max(0, i-1)
    j_min = max(0, j-1)
    i_max = min(8, i+1)
    j_max = min(8, j+1)
    for i in range(i_min, i_max+1):
        for j in range(j_min, j_max+1):
            if (i, j) not in NOT_IN_REAL_BOARD:
                neighbors.append((i, j))
    return set(neighbors)

class IncrementalBoard:
    """
    Incremental board to compute applied actions and undo them
    """

    def __init__(self, root_state:AvalamState)->None:
        """
        Incremental board to compute applied actions and undo them
        :param root_state: the current state
        """
        self.root_state = root_state
        self.root_actions = list(root_state.get_actions())
        self.root_towers_positions = [ (i, j) for i, j, h in root_state.get_towers()]
        self.root_board = root_state.m
        self.max_height = root_state.max_height
        self.actions_played = []
        self.head = 0
        self.rows = root_state.rows
        self.columns = root_state.columns
        self.modified_neighborhoods = []
        self.modified_cells_history = dict()
        self.modified_cells = dict()

    def cell(self, i:int, j:int)->int:
        """
        Get the value of a cell
        :param i: row index
        :param j: column index
        :return: the value of the cell
        """
        if (i, j) in self.modified_cells:
            return self.modified_cells[(i, j)]
        return self.root_board[i][j]

    def append(self, action:tuple[int,int,int,int]):
        """
        Apply an action to the board
        :param action: the action to apply
        """

        # apply action
        x_from, y_from, x_to, y_to = action
        new_h = abs(self.cell(x_to, y_to)) + abs(self.cell(x_from, y_from))
        if self.cell(x_from, y_from) < 0:
            new_h = -new_h
        self.actions_played.append(action)

        modified_neighborhood = neighbor_cells_indexes(x_to, y_to).union(neighbor_cells_indexes(x_from, y_from))

        #update modified neighborhood
        modified_neighborhood.remove((x_from, y_from))
        if self.head != 0:
            modified_neighborhood = self.modified_neighborhoods[-1].union(modified_neighborhood)
        self.modified_neighborhoods.append(modified_neighborhood)

        #update modified cells history
        self.modified_cells_history[(x_from, y_from, self.head)] = self.cell(x_from, y_from)
        self.modified_cells_history[(x_to, y_to, self.head)] = self.cell(x_to, y_to)
        self.modified_cells[(x_from, y_from)] = 0
        self.modified_cells[(x_to, y_to)] = new_h

        self.head += 1
        return self

    def pop(self)-> tuple[int, int, int, int]:
        """
        Undo the last action
        """
        if self.head == 0:
            return None
        self.head -= 1
        action = self.actions_played.pop()
        self.modified_neighborhoods.pop()

        #update modified cells
        x_from, y_from, x_to, y_to = action
        self.modified_cells[(x_from, y_from)] = self.modified_cells_history[(x_from, y_from, self.head)]
        self.modified_cells[(x_to, y_to)] = self.modified_cells_history[(x_to, y_to, self.head)]
        del self.modified_cells_history[(x_from, y_from, self.head)]
        del self.modified_cells_history[(x_to, y_to, self.head)]
        return action

    @property
    def modified_neighborhood(self)->set[tuple[int, int]]:
        """
        Return the modified neighborhood of the head
        :return: the modified neighborhood of the head
        """
        if self.head == 0:
            return set()
        return self.modified_neighborhoods[-1]

    @property
    def last_action(self) -> tuple[int, int, int, int]:
        """
        Return the last action played
        :return: the last action played
        """
        if len(self.actions_played) == 0:
            return None
        return self.actions_played[-1]

    def get_towers(self)-> list[tuple[int, int, int]]:
        """
        Return the towers positions
        :return: the towers positions
        """
        for (i, j) in self.root_towers_positions:
            h = self.cell(i, j)
            if h != 0:
                yield (i, j, h)

    def get_actions(self) -> list[tuple[int, int, int, int]]:
        """
        Return all possible actions from the current state
        :return: the actions
        """
        for action in self.root_actions:
            if self.is_action_still_valid(action):
                yield action

    def is_action_still_valid(self, action):
        """
        Check if an action still valid
        :param action: the action to check
        :return: True if the action is valid
        """
        x_from, y_from, x_to, y_to = action
        if self.cell(x_to, y_to) == 0:
            return False
        if self.cell(x_from, y_from) == 0:
            return False
        if abs(self.cell(x_from, y_from)) + abs(self.cell(x_to, y_to)) > self.max_height:
            return False
        return True

    def get_tower_actions(self, i, j):
        """
        Yield all actions with moving tower (i,j)
        """
        for (x, y) in neighbor_cells_indexes(i, j):
            if self.is_action_still_valid((i, j, x, y)):
                yield (i, j, x, y)

    def get_neighbors(self, i:int, j:int)-> list[tuple[int, int]]:
        """
        Return the neighbors of tower (i,j)
        :param i: row index
        :param j: column index
        :return: the neighbors
        """
        for i, j in neighbor_cells_indexes(i, j):
            if self.cell(i, j) != 0:
                yield (i, j)

    def is_finished(self):
        """
        Return whether no more moves can be made (i.e., game finished).
        """
        for _ in self.get_actions():
            return False
        return True

    def is_tower_movable(self, i, j):
        """
        Return wether tower (i,j) is movable
        """
        h = abs(self.cell(i, j))
        for neighbor in neighbor_cells_indexes(i, j):
            if abs(self.cell(*neighbor)) + h <= self.max_height:
                return True
        return False

    def get_score(self):
        """Return a score for this board.

        The score is the difference between the number of towers of each
        player. In case of ties, it is the difference between the maximal
        height towers of each player. If self.is_finished() returns True,
        this score represents the winner (<0: red, >0: yellow, 0: draw).

        """
        score = 0
        for i in range(self.rows):
            for j in range(self.columns):
                if self.cell(i, j) < 0:
                    score -= 1
                elif self.cell(i, j) > 0:
                    score += 1
        if score == 0:
            for i in range(self.rows):
                for j in range(self.columns):
                    if self.cell(i, j) == -self.max_height:
                        score -= 1
                    elif self.cell(i, j) == self.max_height:
                        score += 1
        return score
