# multiAgents.py
# --------------
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


from cmath import exp
import imp
from operator import ge
from os import access
import re
from time import sleep
from traceback import print_tb
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # print ("Scores-> ", scores)
        # print ("Best Scores-> ", bestScore)
        # print (legalMoves[chosenIndex])
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        import math
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # newPos = successorGameState.getPacmanPosition()
        newFood = list(successorGameState.getFood())[1:-1]
        newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        '''
        Strategy-
          1. Stay alive, avoid ghost for 1 box
            - any action that keeps me near ghost is bad -5
          2. Go towards Food
            - any action that gets me near food is good
          3. Don't stay near walls:
            - Maybe-- anything that keeps me away from borders is good.
        '''

        #print ("New Position is- ", newPos)
        # print("Has Food- ", currentGameState.hasFood(newPos[0], newPos[1]))
        # print ("Score-> ", successorGameState.getScore())

        # If Action is Stop then return default score
        if action == 'Stop':
          return successorGameState.getScore()

        # Logic --
        currScore = successorGameState.getScore()

        # Pacman Zone- check # unit around me
        z_reflex = 1

        # Rewards --
        r_nearGhost = -5
        r_awayGhost = +5
        r_NearWall = 0  # avoid wall reward is set to 0
        r_nearFood = +3
        r_mh_distance_multiplier = -0.01 # manhattan dist multiplier

        # Ghost position
        p_ghosts = []
        for ghostState in newGhostStates:
          p_ghosts.append([int(_) for _ in list(ghostState.getPosition())])
        # print("Ghost States-> ", p_ghosts)

        # Packman position
        p_pacman = successorGameState.getPacmanPosition()
        p_x = p_pacman[0]
        p_y = p_pacman[1]
        p_pacman_new_position = []


        # Get locations here --
        # Avoid Ghost Locations
        for x in range(-z_reflex,z_reflex):
          for y in range(-z_reflex,z_reflex):
            if x == 0 and y == 0:
              continue
            p_pacman_new_position.append([p_x + x, p_y + y])


        # Get Food Locations
        p_food = []
        x_fctr = 1
        for x in newFood:
          y_fctr = 1
          for y in x[1:-1]:
            if y:
              p_food.append([x_fctr, y_fctr])
            y_fctr += 1
          x_fctr += 1


        # Get eucledian distance of those food locations, to prefer nearest food.
        if p_food:
          p_food_nearest = min(p_food, key=lambda x: math.hypot(x[0] - p_pacman[0], x[1] - p_pacman[1]))


        # Edit Score Here --
        for np in p_pacman_new_position:
          # ---  Avoid Ghost ---
          if np in p_ghosts:
            currScore += r_nearGhost
          else:
            currScore += r_awayGhost

          # --- Prefer Food ---
          if successorGameState.hasFood(*np):
            # For food near me
            currScore += r_nearFood
          elif p_food:
            # For food away from me
            mh_distance = util.manhattanDistance(np,p_food_nearest)
            currScore += (mh_distance * r_mh_distance_multiplier )
            # print("calc md")

          # --- Avoid Walls ---
          if successorGameState.hasWall(*np):
            currScore += r_NearWall


        # TODO- comment sleep
        #sleep(0.1)

        # return successorGameState.getScore()
        # print(currScore)
        return currScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):

        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        pacman_moves = gameState.getLegalActions(0)
        bestScore = [self.minState(gameState.generateSuccessor(0,action), 0,1) for action in pacman_moves]
        bestIndex = bestScore.index(max(bestScore))
        return pacman_moves[bestIndex]


    def minState(self, gameState, depth, agentIndex):
        # check if gameState is at final state (win/lose) or proceed
        numAgents = gameState.getNumAgents()
        legal_actions = gameState.getLegalActions(agentIndex)
        if gameState.isWin() or gameState.isLose() or self.depth == depth:
            return scoreEvaluationFunction(gameState)
        if agentIndex == numAgents-1: # call max for pacman's move
            depth += 1
            successors = [self.maxState(gameState.generateSuccessor(agentIndex, action), depth) for action in legal_actions]
        else:  # it is ghost's turn. increment index and call min
            successors = [self.minState(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1) for action in legal_actions]
        return min(successors)

    def maxState(self, gameState, depth):
        legal_actions = gameState.getLegalActions(0)
        if gameState.isWin() or gameState.isLose() or self.depth == depth:
            return scoreEvaluationFunction(gameState)
        successors = [self.minState(gameState.generateSuccessor(0, action), depth, 1) for action in legal_actions]
        return max(successors)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacman_moves = gameState.getLegalActions(0)
        min_val = float('inf')
        min_counter = 0
        bestScore = [self.minState(gameState.generateSuccessor(0,action), 0,1,min_val,min_counter) for action in pacman_moves]
        bestIndex = bestScore.index(max(bestScore))
        return pacman_moves[bestIndex]


    def minState(self, gameState, depth, agentIndex, min_val, min_counter):
        # check if gameState is at final state (win/lose) or proceed
        numAgents = gameState.getNumAgents()
        legal_actions = gameState.getLegalActions(agentIndex)
        if gameState.isWin() or gameState.isLose() or self.depth == depth:
            return scoreEvaluationFunction(gameState)
        if agentIndex == numAgents-1: # call max for pacman's move
            depth += 1
            successors =[]
            max_val = -float('inf')
            counter = 0
            # successors = [self.maxState(gameState.generateSuccessor(agentIndex, action), depth) for action in legal_actions]
            for action in legal_actions:
                value = self.maxState(gameState.generateSuccessor(agentIndex,action),depth, max_val, counter)
                successors.append(value[0])
                max_val = value[1]
                counter = value[2]


        else:  # it is ghost's turn. increment index and call min
            successors = []
            for action in legal_actions:
                value = self.minState(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1, min_val, min_counter)
                if not min_counter:
                    min_val = value
                    min_counter +=1
                elif min_val < value:
                    min_val = value
                    return [min_val, min_val, min_counter]
                successors.append(value)
        return [min(successors),min_val,min_counter]

    def maxState(self, gameState, depth, max_val, counter):
        legal_actions = gameState.getLegalActions(0)
        if gameState.isWin() or gameState.isLose() or self.depth == depth:
            return scoreEvaluationFunction(gameState)
        successors =[]
        min_val = float('inf')
        min_counter = 0
        for action in legal_actions:
            print(min_val)
            print(min_counter)
            value = self.minState(gameState.generateSuccessor(0, action), depth, 1, min_val,min_counter)
            print(value)
            min_val = value[1]
            min_counter = value[2]
            if not counter:
                max_val = value
                counter +=1
            elif max_val < value:
                max_val = value
                return [max_val, max_val, counter]
            successors.append(value)
        return [max(successors), max_val, counter]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        '''
        Strategy-
          1. First move is pacman move
          2. Apply max to it
          3. Then perform ghost moves for thier num of agents
          4. Apply avg to it
        '''

        currAgent = 0
        myDepth = 0
        score = []
        legalMoves = gameState.getLegalActions(currAgent)

        # First Move is always pacman move
        for action in legalMoves:
          score.append( self.expectiMax(gameState.generateSuccessor(currAgent, action), 1, myDepth ))

        bestScoreIdx = score.index( max(score) )
        # print ("Score-- ",score)
        # print ("Move-- ", legalMoves[bestScoreIdx])

        return legalMoves[bestScoreIdx]


    def expectiMax(self, gameState, currAgent, myDepth):
      # Terminal States

      # For win or lose
      if gameState.isWin() or gameState.isLose():
        return scoreEvaluationFunction(gameState)

      # For depth
      if myDepth == self.depth:
        return scoreEvaluationFunction(gameState)


      # Pacman Move
      if currAgent == 0:
        # print ("Max Agent- ", currAgent)
        return max( [ self.expectiMax(gameState.generateSuccessor(currAgent, action), currAgent+1, myDepth) for action in gameState.getLegalActions(currAgent)] )

      else:
        # Ghost Move
        # print("Avg Agent- ", currAgent)

        if currAgent == gameState.getNumAgents() - 1:
          nxtAgent = 0
          myDepth = myDepth + 1
        else:
          nxtAgent = currAgent + 1

        return sum( [ self.expectiMax(gameState.generateSuccessor(currAgent, action), nxtAgent, myDepth) for action in gameState.getLegalActions(currAgent) ] )/ float(len(gameState.getLegalActions(currAgent)))


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
