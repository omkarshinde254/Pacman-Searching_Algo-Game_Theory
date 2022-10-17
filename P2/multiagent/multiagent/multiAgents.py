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


import imp
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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #print ("New Position is- ", newPos)
        # print("Has Food- ", currentGameState.hasFood(newPos[0], newPos[1]))
        # print ("Score-> ", successorGameState.getScore())

        '''
        Strategy- 
          1. Stay alive, avoid ghost for 1 box
            - any action that keeps me near ghost is bad -5
            - any action that gets me away from ghost is good +5
          2. Go towards Food
        '''

        # If Action is Stop then return default score
        if action == 'Stop':
          return successorGameState.getScore()

        # Logic -- 
        currScore = successorGameState.getScore()

        # Pacman Zone
        z_reflex = 1

        # Rewards --
        r_nearGhost = -5
        r_awayGhost = +5
        r_NearWall = 0
        r_nearFood = +3
        r_mh_distance_multiplier = -0.01

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

        # Avoid Ghost Logic:
        for x in range(-z_reflex,z_reflex):
          for y in range(-z_reflex,z_reflex):
            if x == 0 and y == 0:
              continue
            p_pacman_new_position.append([p_x + x, p_y + y])
      
        
        # Food position
        p_food = []
        x_fctr = 1
        for x in newFood:
          y_fctr = 1
          for y in x[1:-1]:
            if y:
              p_food.append([x_fctr, y_fctr])
            y_fctr += 1
          x_fctr += 1
        
        #p_food.sort()
        #print ("Food --> ", p_food)
        # Get nearest food (Euclidean Distance)
        #print ("P Pacman-- ", p_pacman)
        #print ("P Pacman-- ", p_food)
        if p_food:
          p_food_nearest = min(p_food, key=lambda x: math.hypot(x[0] - p_pacman[0], x[1] - p_pacman[1]))

        # print("Pacman Current Position-> ", p_pacman)
        # print("Nearest Food --> ", p_food_nearest)
        # print("Pacman New Position-> ", p_pacman_new_position)

        # Logic for score tracking
        for np in p_pacman_new_position:

          # Ghost check
          if np in p_ghosts:
            currScore += r_nearGhost
          else:
            currScore += r_awayGhost

          # Food Check near me
          if successorGameState.hasFood(*np):
            currScore += r_nearFood
          elif p_food:
            mh_distance = util.manhattanDistance(np,p_food_nearest)
            currScore += (mh_distance * r_mh_distance_multiplier )
            # print("calc md")


          # Wall Check
          if successorGameState.hasWall(*np):
            currScore += r_NearWall


        # TODO
        #sleep(0.1)

        # return successorGameState.getScore()
        #print(currScore)
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
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
          1. Make a tree of scores
          2. apply max and avg to that tree.
          3. root should be max
          4. return action with max score
        '''

        # print("In Expecti Max function")
        legalMoves = gameState.getLegalActions()
        # print("Legal Moves- ", legalMoves)
        # print("Num of Agents- ", gameState.getNumAgents())

        scores = []
        isMax = False
        self.depth = 4
        print ("Depth in Question- ", self.depth)

        for action in legalMoves:
          visited = []
          # print("Checking for action-- ", action)
          successorGameState = gameState.generateSuccessor(0, action)
          scores.append(self.expectiMax(successorGameState, visited, isMax, 0))
        
        print("Score==> ", scores)
        maxidx = scores.index(max(scores))
        print ("Move==> ",legalMoves[maxidx])
        return legalMoves[maxidx]

    
    def expectiMax(self, gameState, visited, isMax, myDepth):
      # sleep(0.5)
      # currPos = self.gameState.getPacmanPosition()
      # currPos = gameState.getPosition()
      # print("Curr Poss-- ", currPos)

      if gameState.isLose() or gameState.isWin():
        return self.evaluationFunction(gameState)
      
      if myDepth == self.depth:
        # print("Depth- ", myDepth)
        print("Value- ", self.evaluationFunction(gameState))
        return self.evaluationFunction(gameState)

      # if currPos in visited:
      #   return self.evaluationFunction(gameState)
      # else:
      #   visited.append(currPos)

      ss_legalMoves = gameState.getLegalActions()
      myDepth += 1
      print("Curr Depth- ", myDepth)

      if isMax:
        isMax = False
        # print("Max")
        arr =  [ self.expectiMax(gameState.generateSuccessor(0, action), visited, isMax, myDepth) for action in ss_legalMoves]
        print ("Max Arr- ", arr)
        return max ( arr )
        # score = []
        # for action in ss_legalMoves:
          # score.append(self.expectiMax(gameState.generatePacmanSuccessor(action), visited, isMax, depth))
        # print("Score- ", score)
        # return max(score)
      else:
        isMax = True
        # print("Avg")
        arr = [ self.expectiMax(gameState.generateSuccessor(0, action), visited, isMax, myDepth) for action in ss_legalMoves ]
        print ("Avg Arr- ", arr)
        return sum(arr)/float(len(ss_legalMoves))
        # score = []
        # for action in ss_legalMoves:
          # score.append(self.expectiMax(gameState.generatePacmanSuccessor(action), visited, isMax, depth))
        # print("Score- ", score)
        # return (sum(score)/float(len(score)))


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

