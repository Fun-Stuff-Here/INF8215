# search.py
# ----------------------------------------------------------------------------------------------------------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu)
# ----------------------------------------------------------------------------------------------------------------------
# Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem: SearchProblem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 1 ICI
    '''
    POSITION_INDEX = 0
    DIRECTION_INDEX = 1
    COST_INDEX = 2

    visitedPositions = dict()
    stack = util.Stack()
    start = (problem.getStartState(), None, 0) #Node format ( (x,y), Direction, cost)
    stack.push(start)

    while not stack.isEmpty():
        currentState = stack.pop()

        if problem.isGoalState(currentState[POSITION_INDEX]):
            solution = []

            while currentState[POSITION_INDEX] is not start[POSITION_INDEX]:
                currentState = visitedPositions[currentState[POSITION_INDEX]]
                solution = [*solution, currentState[DIRECTION_INDEX]]

            solution.reverse()
            return solution

        for successor in problem.getSuccessors(currentState[POSITION_INDEX]):
            if not successor[POSITION_INDEX] in visitedPositions:
                visitedPositions[successor[POSITION_INDEX]] = (currentState[POSITION_INDEX], successor[DIRECTION_INDEX])
                stack.push(successor)

    return None

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""


    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 2 ICI
    '''
    POSITION_INDEX = 0
    DIRECTION_INDEX = 1
    COST_INDEX = 2

    visitedPositions = dict()
    queue = util.Queue()
    start = (problem.getStartState(), None, 0) #Node format ( (x,y), Direction, cost)
    queue.push(start)
    visitedPositions[start[POSITION_INDEX]] = (start[POSITION_INDEX], start[DIRECTION_INDEX]) # avoid enqueue of start

    while not queue.isEmpty():
        currentState = queue.pop()

        if problem.isGoalState(currentState[POSITION_INDEX]):
            solution = []

            while currentState[POSITION_INDEX] is not start[POSITION_INDEX]:
                currentState = visitedPositions[currentState[POSITION_INDEX]]
                solution = [*solution, currentState[DIRECTION_INDEX]]

            solution.reverse()
            return solution

        for successor in problem.getSuccessors(currentState[POSITION_INDEX]):
            if not successor[POSITION_INDEX] in visitedPositions:
                visitedPositions[successor[POSITION_INDEX]] = (currentState[POSITION_INDEX], successor[DIRECTION_INDEX])
                queue.push(successor)

    return None

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""


    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 3 ICI
    '''
    POSITION_INDEX = 0
    DIRECTION_INDEX = 1
    COST_INDEX = 2

    visitedPositions = dict()
    queue = util.PriorityQueue()
    start = (problem.getStartState(), None, 0) #Node format ( (x,y), Direction, cost)
    queue.push(start, 0)
    visitedPositions[start[POSITION_INDEX]] = (start[POSITION_INDEX], start[DIRECTION_INDEX]) # avoid enqueue of start

    while not queue.isEmpty():
        currentState = queue.pop()

        if problem.isGoalState(currentState[POSITION_INDEX]):
            solution = []

            while currentState[POSITION_INDEX] is not start[POSITION_INDEX]:
                currentState = visitedPositions[currentState[POSITION_INDEX]]
                solution = [*solution, currentState[DIRECTION_INDEX]]

            solution.reverse()
            return solution

        for successor in problem.getSuccessors(currentState[POSITION_INDEX]):
            if not successor[POSITION_INDEX] in visitedPositions:
                visitedPositions[successor[POSITION_INDEX]] = (currentState[POSITION_INDEX], successor[DIRECTION_INDEX])
                queue.push(successor, successor[COST_INDEX])

    return None

def nullHeuristic(state, problem: SearchProblem = None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 4 ICI
    '''

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch