"""
  Avalam numba compatible
"""
from numba.experimental import jitclass
from numba.types import int64, optional, Tuple
from numba.typed import List # pylint: disable=no-name-in-module
import numpy as np

# (negative for red, positive for yellow)
# this score represents the winner (<0: red, >0: yellow, 0: draw).
PLAYER1 = 1 # yellow
PLAYER2 = -1 # red
YELLOW = PLAYER1
RED = PLAYER2

@jitclass([
    ('max_height', int64),
    ('initial_board', int64[:,:]),
    ('m', int64[:,:]),
    ('rows', int64),
    ('columns', int64),
    ('max_height', int64),
    ('last_action', optional(Tuple([int64, int64, int64, int64]))),
])
class Board:

    """Representation of an Avalam Board.

    self.m is a self.rows by self.columns bi-dimensional array representing the
    board.  The absolute value of a cell is the height of the tower.  The sign
    is the color of the top-most counter (negative for red, positive for
    yellow).
    """

    def __init__(self, percepts=np.array([ [ 0,  0,  1, -1,  0,  0,  0,  0,  0],
                                [ 0,  1, -1,  1, -1,  0,  0,  0,  0],
                                [ 0, -1,  1, -1,  1, -1,  1,  0,  0],
                                [ 0,  1, -1,  1, -1,  1, -1,  1, -1],
                                [ 1, -1,  1, -1,  0, -1,  1, -1,  1],
                                [-1,  1, -1,  1, -1,  1, -1,  1,  0],
                                [ 0,  0,  1, -1,  1, -1,  1, -1,  0],
                                [ 0,  0,  0,  0, -1,  1, -1,  1,  0],
                                [ 0,  0,  0,  0,  0, -1,  1,  0,  0] ]),
                       max_height=5,
                       invert=False) -> None:
        """Initialize the board.

        Arguments:
        percepts -- matrix representing the board
        invert -- whether to invert the sign of all values, inverting the
            players
        max_height -- maximum height of a tower

        """
        # standard avalam
        self.max_height = 5
        self.initial_board =  np.array([ [ 0,  0,  1, -1,  0,  0,  0,  0,  0],
                                [ 0,  1, -1,  1, -1,  0,  0,  0,  0],
                                [ 0, -1,  1, -1,  1, -1,  1,  0,  0],
                                [ 0,  1, -1,  1, -1,  1, -1,  1, -1],
                                [ 1, -1,  1, -1,  0, -1,  1, -1,  1],
                                [-1,  1, -1,  1, -1,  1, -1,  1,  0],
                                [ 0,  0,  1, -1,  1, -1,  1, -1,  0],
                                [ 0,  0,  0,  0, -1,  1, -1,  1,  0],
                                [ 0,  0,  0,  0,  0, -1,  1,  0,  0] ])
        self.m = percepts # pylint: disable=invalid-name
        self.rows = len(self.m)
        self.columns = len(self.m[0])
        self.max_height = max_height
        self.m = self.get_percepts(invert)  # make a copy of the percepts
        self.last_action = None

    def __str__(self) -> str:
        def str_cell(i, j):
            x = self.m[i][j] # pylint: disable=invalid-name
            if x:
                return "%+2d" % x
            else:
                return " ."
        return "\n".join(" ".join(str_cell(i, j) for j in range(self.columns))
                         for i in range(self.rows))

    def clone(self):
        """Return a clone of this object."""
        return Board(self.m.copy(), self.max_height, False)

    def get_percepts(self, invert=False):
        """Return the percepts corresponding to the current state.

        If invert is True, the sign of all values is inverted to get the view
        of the other player.

        """
        mul = PLAYER1
        if invert:
            mul = PLAYER2
        percepts = self.m.copy()
        for i in range(self.rows):
            for j in range(self.columns):
                percepts[i][j] *= mul
        return percepts

    def get_towers(self):
        """Yield all towers.

        Yield the towers as triplets (i, j, h):
        i -- row number of the tower
        j -- column number of the tower
        h -- height of the tower (absolute value) and owner (sign)

        """
        for i in range(self.rows):
            for j in range(self.columns):
                if self.m[i][j]:
                    yield (i, j, self.m[i][j])

    def get_tower_actions(self, i, j):
        """Yield all actions with moving tower (i,j)"""
        h = abs(self.m[i][j]) # pylint: disable=invalid-name
        if 0 < h < self.max_height:
            i_min = max(0, i-1)
            j_min = max(0, j-1)
            i_max = min(8, i+1)
            j_max = min(8, j+1)
            for di in range(i_min, i_max+1):
                for dj in range(j_min, j_max+1):
                    if h < abs(self.m[di][dj]) + h <= self.max_height:
                        if di == i and dj == j:
                            continue
                        yield (i, j, di, dj)

    def get_actions(self) -> list[tuple[int, int, int, int]]:
        """Yield all valid actions on this board."""
        actions = List()
        for i, j, _ in self.get_towers():
            for action in self.get_tower_actions(i, j):
                actions.append(action)
        return actions

    def play_action(self, action):
        """Play an action if it is valid.

        An action is a 4-uple containing the row and column of the tower to
        move and the row and column of the tower to gobble. If the action is
        invalid, raise an InvalidAction exception. Return self.

        """
        if not self.is_action_valid(action):
            return self
        i1, j1, i2, j2 = action # pylint: disable=invalid-name
        h1 = abs(self.m[i1][j1]) # pylint: disable=invalid-name
        h2 = abs(self.m[i2][j2]) # pylint: disable=invalid-name
        if self.m[i1][j1] < 0:
            self.m[i2][j2] = -(h1 + h2)
        else:
            self.m[i2][j2] = h1 + h2
        self.m[i1][j1] = 0
        self.last_action = action
        return self

    def is_finished(self):
        """Return whether no more moves can be made (i.e., game finished)."""
        for _ in self.get_actions():
            return False
        return True

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
                if self.m[i][j] < 0:
                    score -= 1
                elif self.m[i][j] > 0:
                    score += 1
        if score == 0:
            for i in range(self.rows):
                for j in range(self.columns):
                    if self.m[i][j] == -self.max_height:
                        score -= 1
                    elif self.m[i][j] == self.max_height:
                        score += 1
        return score

    def is_tower_movable(self, i, j):
        """Return wether tower (i,j) is movable"""
        for _ in self.get_tower_actions(i, j):
            return True
        return False

    def is_action_valid(self, action):
        """Return whether action is a valid action."""

        i1, j1, i2, j2 = action # pylint: disable=invalid-name
        if i1 < 0 or j1 < 0 or i2 < 0 or j2 < 0 or \
            i1 >= self.rows or j1 >= self.columns or \
            i2 >= self.rows or j2 >= self.columns or \
            (i1 == i2 and j1 == j2) or (abs(i1-i2) > 1) or (abs(j1-j2) > 1):
            return False
        h1 = abs(self.m[i1][j1]) # pylint: disable=invalid-name
        h2 = abs(self.m[i2][j2]) # pylint: disable=invalid-name
        if h1 <= 0 or h1 >= self.max_height or h2 <= 0 or \
                h2 >= self.max_height or h1+h2 > self.max_height:
            return False
        return True
