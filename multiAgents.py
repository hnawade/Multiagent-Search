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
from math import log, exp

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        oldPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        score = successorGameState.getScore()

        if newFood.count() != 0:
            score += min([manhattanDistance(food, oldPos) for food in newFood.asList()]) - min(
                [manhattanDistance(food, newPos) for food in newFood.asList()])

        if len(newGhostStates) != 0:
            score *= 1 - exp(
                -min([manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]))

        return score


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minmax_value(depth, state, agent):
            if depth == 0:
                return self.evaluationFunction(state), None
            if agent == 0:
                v = -float("inf")
                max_action = None
                for action in state.getLegalActions(0):
                    next_state = state.generateSuccessor(agent, action)
                    if next_state.isWin():
                        v0 = self.evaluationFunction(next_state)
                    elif next_state.isLose():
                        v0 = self.evaluationFunction(next_state)
                    else:
                        v0 = minmax_value(depth, next_state, agent + 1)[0]
                    if v0 > v:
                        v = v0
                        max_action = action
                return v, max_action
            elif agent < gameState.getNumAgents() - 1:
                v = float("inf")
                max_action = None
                for action in state.getLegalActions(agent):
                    next_state = state.generateSuccessor(agent, action)
                    if next_state.isWin():
                        v0 = self.evaluationFunction(next_state)
                    elif next_state.isLose():
                        v0 = self.evaluationFunction(next_state)
                    else:
                        v0 = minmax_value(depth, state.generateSuccessor(agent, action), agent + 1)[0]
                    if v0 < v:
                        v = v0
                        max_action = action
                return v, max_action
            else:
                v = float("inf")
                max_action = None
                for action in state.getLegalActions(agent):
                    next_state = state.generateSuccessor(agent, action)
                    if next_state.isWin():
                        v0 = self.evaluationFunction(next_state)
                    elif next_state.isLose():
                        v0 = self.evaluationFunction(next_state)
                    else:
                        v0 = minmax_value(depth - 1, state.generateSuccessor(agent, action), 0)[0]
                    if v0 < v:
                        v = v0
                        max_action = action
                return v, max_action

        return minmax_value(self.depth, gameState, 0)[1]
        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphabet(depth, state, agent, a, b):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            if agent == 0:
                v = -float("inf")
                max_action = None
                for action in state.getLegalActions(agent):
                    next_state = state.generateSuccessor(agent, action)
                    v0 = alphabet(depth, next_state, agent + 1, a, b)[0]
                    if v0 > v:
                        v = v0
                        max_action = action
                    if v > b:
                        return v, max_action
                    a = max(a, v)
            elif agent < gameState.getNumAgents() - 1:
                v = float("inf")
                max_action = None
                for action in state.getLegalActions(agent):
                    next_state = state.generateSuccessor(agent, action)
                    v0 = alphabet(depth, next_state, agent + 1, a, b)[0]
                    if v0 < v:
                        v = v0
                        max_action = action
                    if v < a:
                        return v, max_action
                    b = min(b, v)
            else:
                v = float("inf")
                max_action = None
                for action in state.getLegalActions(agent):
                    next_state = state.generateSuccessor(agent, action)
                    v0 = alphabet(depth - 1, next_state, 0, a, b)[0]
                    if v0 < v:
                        v = v0
                        max_action = action
                    if v < a:
                        return v, max_action
                    b = min(b, v)
            return v, max_action

        return alphabet(self.depth, gameState, 0, -float("inf"), float("inf"))[1]
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

        def expect(depth, state, agent):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            if agent == 0:
                v = -float("inf")
                max_action = None
                for action in state.getLegalActions(agent):
                    next_state = state.generateSuccessor(agent, action)
                    v0 = expect(depth, next_state, 1)[0]
                    if v0 > v:
                        v = v0
                        max_action = action
            elif agent < gameState.getNumAgents() - 1:
                total = 0
                max_action = None
                for action in state.getLegalActions(agent):
                    next_state = state.generateSuccessor(agent, action)
                    total += expect(depth, next_state, agent + 1)[0]
                v = total / len(state.getLegalActions(agent))
            else:
                total = 0
                max_action = None
                for action in state.getLegalActions(agent):
                    next_state = state.generateSuccessor(agent, action)
                    total += expect(depth - 1, next_state, 0)[0]
                v = total / len(state.getLegalActions(agent))
            return v, max_action

        return expect(self.depth, gameState, 0)[1]

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    def alphabet(depth, state, agent, a, b):
        if depth == 0 or state.isWin() or state.isLose():
            return scoreEvaluationFunction(state), None
        if agent == 0:
            v = -float("inf")
            max_action = None
            for action in state.getLegalActions(agent):
                next_state = state.generateSuccessor(agent, action)
                v0 = alphabet(depth, next_state, agent + 1, a, b)[0]
                if v0 > v:
                    v = v0
                    max_action = action
                if v > b:
                    return v, max_action
                a = max(a, v)
        elif agent < state.getNumAgents() - 1:
            v = float("inf")
            max_action = None
            for action in state.getLegalActions(agent):
                next_state = state.generateSuccessor(agent, action)
                v0 = alphabet(depth, next_state, agent + 1, a, b)[0]
                if v0 < v:
                    v = v0
                    max_action = action
                if v < a:
                    return v, max_action
                b = min(b, v)
        else:
            v = float("inf")
            max_action = None
            for action in state.getLegalActions(agent):
                next_state = state.generateSuccessor(agent, action)
                v0 = alphabet(depth - 1, next_state, 0, a, b)[0]
                if v0 < v:
                    v = v0
                    max_action = action
                if v < a:
                    return v, max_action
                b = min(b, v)
        return v, max_action

    return alphabet(1, currentGameState, 0, -float("inf"), float("inf"))[0]
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
