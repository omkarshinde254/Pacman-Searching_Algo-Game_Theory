# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def DFSRecursion(problem, visited, return_array, node, action, isgoal):
    # Check if we have reached goal node
    if isgoal:
        return return_array, isgoal

    if problem.isGoalState(node):
        isgoal = True
        return return_array.append(action), isgoal

    # Define visited set for first time
    if not visited:
        visited = set()
        return_array = []

    # Add node in visited set
    visited.add(node)
    if action:
        return_array.append(action)

    # Loop through neighbours using DFS
    neighbour = problem.getSuccessors(node)
    # print(neighbour)
    for value in neighbour:
        if value[0] not in visited:
            array, isgoal = DFSRecursion(problem, visited, return_array, value[0], value[1], isgoal)

    if isgoal == False:
        return_array.pop()
    return return_array, isgoal

def BFSLogic(problem,start):
    from util import Queue

    visited = []
    queue = Queue()
    # push start state to the queue
    queue.push((start, []))

    while(queue):
        # move through the queue iteratively until goal state is found
        # get successors and iterate through them
        # if node is unvisited, append it to visited array and queue and dont visit again
        state, path = queue.pop()
        if (problem.isGoalState(state)):
            break
        neighbours = problem.getSuccessors(state)
        for neighbour in neighbours:
            print(neighbour)
            if(neighbour[0] not in visited):  # if not visited, add to array+path and don't visit again
                visited.append(neighbour[0])
                queue.push((neighbour[0], path + [neighbour[1]]))
    return path



def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from game import Directions

    startnode = problem.getStartState()
    current_path, isGoal = DFSRecursion(problem, None, None, startnode, '', False)
    return current_path


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from game import Directions

    startnode = problem.getStartState()
    current_path = BFSLogic(problem, startnode)
    return current_path

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
