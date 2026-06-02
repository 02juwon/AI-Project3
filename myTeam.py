# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class BaseCaptureAgent(CaptureAgent):
  """
  A feature-based reflex agent base class. Provides shared helpers used by
  both the offensive and defensive agents: a maze-distance cache, home
  boundary computation, eaten-food detection, and the standard
  evaluate/getFeatures/getWeights reflex loop.
  """

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)

    self.walls = gameState.getWalls()
    self.mapWidth = self.walls.width
    self.mapHeight = self.walls.height

    # Cache for maze distance lookups.
    self._distanceCache = {}

    # Pre-compute the non-wall positions along our side of the home boundary.
    self.homeBoundary = self.computeHomeBoundary(gameState)

    # Defensive patrol center: the boundary cell closest to the map's
    # vertical middle (a natural choke region to watch).
    midY = self.mapHeight // 2
    self.defensiveCenter = min(
        self.homeBoundary,
        key=lambda p: abs(p[1] - midY))

    # Remember the food we are defending so we can detect what gets eaten.
    self.lastDefendingFood = self.getFoodYouAreDefending(gameState).asList()
    self.lastEatenFood = None

  def computeHomeBoundary(self, gameState):
    """Return the list of legal (non-wall) cells on our side of the border."""
    if self.red:
      boundaryX = (self.mapWidth // 2) - 1
    else:
      boundaryX = self.mapWidth // 2
    boundary = []
    for y in range(self.mapHeight):
      if not self.walls[boundaryX][y]:
        boundary.append((boundaryX, y))
    return boundary

  def getMazeDistanceCached(self, pos1, pos2):
    """Maze distance with memoization to stay within the per-turn time limit."""
    key = (pos1, pos2)
    if key in self._distanceCache:
      return self._distanceCache[key]
    d = self.getMazeDistance(pos1, pos2)
    self._distanceCache[key] = d
    self._distanceCache[(pos2, pos1)] = d
    return d

  def getSuccessor(self, gameState, action):
    """Finds the next successor (handles half-grid positions)."""
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """Linear combination of features and feature weights."""
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 1.0}

  def updateEatenFood(self, gameState):
    """
    Compare the food we are defending now versus last turn. If a piece
    disappeared, an invader ate it, so remember that location.
    """
    currentFood = self.getFoodYouAreDefending(gameState).asList()
    if len(currentFood) < len(self.lastDefendingFood):
      eaten = [f for f in self.lastDefendingFood if f not in currentFood]
      if eaten:
        self.lastEatenFood = eaten[0]
    self.lastDefendingFood = currentFood

  def getVisibleEnemies(self, gameState):
    """Return enemy AgentState objects whose exact position is known."""
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    return [e for e in enemies if e.getPosition() is not None]

  def chooseAction(self, gameState):
    self.updateEatenFood(gameState)
    actions = gameState.getLegalActions(self.index)

    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)


class OffensiveAgent(BaseCaptureAgent):
  """
  Offensive agent. Eats opponent food, flees nearby (non-scared) ghosts,
  grabs capsules when threatened, and returns home once carrying enough food.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)

    # Distance to the nearest food.
    if len(foodList) > 0:
      minFoodDist = min([self.getMazeDistanceCached(myPos, food) for food in foodList])
      features['distanceToFood'] = minFoodDist

    # Carrying count and distance back to our home boundary.
    carrying = myState.numCarrying
    distHome = min([self.getMazeDistanceCached(myPos, b) for b in self.homeBoundary])
    features['distanceToHome'] = distHome

    # Information about visible enemy ghosts (on their own side, not scared).
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    activeGhosts = [e for e in enemies
                    if not e.isPacman and e.getPosition() is not None
                    and e.scaredTimer <= 1]
    ghostDist = None
    if activeGhosts:
      ghostDist = min([self.getMazeDistanceCached(myPos, g.getPosition())
                       for g in activeGhosts])
      # Strongly penalize being adjacent to / on top of a ghost.
      if ghostDist <= 1:
        features['ghostNearby'] = 1
      # Encourage keeping distance when a ghost is close.
      features['ghostDistance'] = ghostDist if ghostDist < 5 else 5

    # Distance to the nearest capsule (useful escape valve when threatened).
    capsules = self.getCapsules(successor)
    if capsules:
      minCapDist = min([self.getMazeDistanceCached(myPos, c) for c in capsules])
      features['distanceToCapsule'] = minCapDist

    # Decide whether we should be heading home.
    #  - carrying a lot of food, or
    #  - carrying something with a ghost breathing down our neck.
    shouldReturn = False
    if carrying >= 5:
      shouldReturn = True
    elif carrying >= 1 and ghostDist is not None and ghostDist <= 4:
      shouldReturn = True
    elif carrying >= 3:
      shouldReturn = True
    if shouldReturn:
      features['returnHome'] = distHome

    # If almost no food left to win the game, just bring it home.
    if len(foodList) <= 2:
      features['returnHome'] = distHome

    if action == Directions.STOP:
      features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev:
      features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {
        'successorScore': 100,
        'distanceToFood': -2,
        'ghostNearby': -1000,
        'ghostDistance': 20,
        'distanceToCapsule': -3,
        'returnHome': -8,
        'stop': -100,
        'reverse': -2,
    }


class DefensiveAgent(BaseCaptureAgent):
  """
  Defensive agent. Stays on our side, chases visible invaders, heads toward
  the last spot food was eaten when no invader is visible, and otherwise
  patrols the central choke region.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Stay a ghost on defense (penalize crossing into enemy territory).
    features['onDefense'] = 1
    if myState.isPacman:
      features['onDefense'] = 0

    # Visible invaders (enemy Pacmen on our side).
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [e for e in enemies if e.isPacman and e.getPosition() is not None]
    features['numInvaders'] = len(invaders)

    if len(invaders) > 0:
      dists = [self.getMazeDistanceCached(myPos, e.getPosition()) for e in invaders]
      features['invaderDistance'] = min(dists)
    elif self.lastEatenFood is not None:
      # No invader visible: move toward the last spot food vanished.
      features['distanceToEaten'] = self.getMazeDistanceCached(myPos, self.lastEatenFood)
    else:
      # Nothing to chase: patrol the central choke region.
      features['distanceToCenter'] = self.getMazeDistanceCached(myPos, self.defensiveCenter)

    if action == Directions.STOP:
      features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev:
      features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {
        'numInvaders': -1000,
        'onDefense': 100,
        'invaderDistance': -10,
        'distanceToEaten': -3,
        'distanceToCenter': -1,
        'stop': -100,
        'reverse': -2,
    }
