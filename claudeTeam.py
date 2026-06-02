# claudeTeam.py
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

# =============================================================================
# Claude Code edition (v2) - rewritten after finishing last in the league.
#
# KEY INSIGHT (verified in capture.py for THIS variant):
#   - Food is scored the INSTANT it is eaten (capture.py: scoreChange += score
#     on consume) and is NEVER refunded. There is no "carry it home to bank it"
#     mechanic here (DUMP_FOOD_ON_DEATH = False, the return-scoring block is
#     commented out). => Retreating home for safety wastes turns we could spend
#     eating. We only go home as a genuine escape, never to "deposit" food.
#   - KILL_POINTS = 3. Killing an enemy Pacman, or eating a scared ghost, is
#     worth +3 -- as much as three dots. Dying hands the enemy +3 AND a long
#     respawn walk. => Survival and hunting invaders are first-class goals.
#
# Strategy distilled from that:
#   Offense  : relentless but survivable forager. Keep eating, evade active
#              ghosts, never walk into dead-ends while chased, grab capsules,
#              and hunt scared ghosts (+3). Go home ONLY to escape.
#   Defense  : kill-focused hunter. Charge and eat invaders (+3 each), keep a
#              safe gap when scared, investigate vanished food, patrol chokes.
#
# Claude's stability mandate is preserved throughout: every turn is wrapped so
# a feature/exception/time-overrun can never forfeit a move, distances fall
# back to Manhattan, and a soft time budget keeps us well under 1 second.
# =============================================================================

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
  """
  Returns the two agents that form the team. The contest harness looks up this
  exact name (`createTeam`), so it must not be renamed.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

# Soft per-action time budget (seconds). Hard limit is 1.0s; we stop exploring
# well before that and return the best action found so far.
TIME_BUDGET = 0.8

# Discount applied to the one-step lookahead bonus (see chooseAction).
LOOKAHEAD_DISCOUNT = 0.6


class BaseCaptureAgent(CaptureAgent):
  """
  Shared base for the offensive and defensive agents. Provides a maze-distance
  cache, home-boundary and dead-end pre-computation, eaten-food tracking, and a
  hardened, time-budgeted action loop with a shallow lookahead.
  """

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)

    self.walls = gameState.getWalls()
    self.mapWidth = self.walls.width
    self.mapHeight = self.walls.height

    # Memoized maze-distance lookups (keeps us inside the per-turn budget).
    self._distanceCache = {}

    # Non-wall cells along our side of the home border. Used both as escape
    # targets and (for the defender) as patrol gates.
    self.homeBoundary = self.computeHomeBoundary(gameState)
    if not self.homeBoundary:
      self.homeBoundary = [self.start]

    midY = self.mapHeight // 2
    self.defensiveCenter = min(self.homeBoundary, key=lambda p: abs(p[1] - midY))

    # Track defended food so the defender can detect where invaders ate.
    self.lastDefendingFood = self.getFoodYouAreDefending(gameState).asList()
    self.lastEatenFood = None

    # Pre-compute dead-end depth for every cell. Depth 0 == not a dead end;
    # higher == deeper inside a one-way pocket where a ghost can trap us.
    self.deadEndDepth = self.computeDeadEndDepths()

  # --- geometry / precomputation ------------------------------------------

  def computeHomeBoundary(self, gameState):
    """Return the legal (non-wall) cells on our side of the border."""
    if self.red:
      boundaryX = (self.mapWidth // 2) - 1
    else:
      boundaryX = self.mapWidth // 2
    return [(boundaryX, y) for y in range(self.mapHeight)
            if not self.walls[boundaryX][y]]

  def computeDeadEndDepths(self):
    """
    Leaf-pruning ("degree reduction") over the maze graph. Repeatedly remove
    cells with <=1 open neighbor; the order they are removed gives how deep
    into a dead-end pocket they sit. Cells on cycles/through-corridors keep
    degree>=2 and stay at depth 0. Runs once at startup.
    """
    free = set()
    for x in range(self.mapWidth):
      for y in range(self.mapHeight):
        if not self.walls[x][y]:
          free.add((x, y))

    def openNeighbors(p):
      x, y = p
      result = []
      for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
        if (nx, ny) in free:
          result.append((nx, ny))
      return result

    neighbors = {p: openNeighbors(p) for p in free}
    degree = {p: len(neighbors[p]) for p in free}
    depth = {}
    queue = [p for p in free if degree[p] <= 1]

    while queue:
      p = queue.pop(0)
      if p in depth:
        continue
      deeper = [depth[n] for n in neighbors[p] if n in depth]
      depth[p] = 1 + (max(deeper) if deeper else 0)
      for n in neighbors[p]:
        if n not in depth:
          degree[n] -= 1
          if degree[n] <= 1:
            queue.append(n)

    return {p: depth.get(p, 0) for p in free}

  # --- distance helpers ----------------------------------------------------

  def getMazeDistanceCached(self, pos1, pos2):
    """Memoized maze distance with a Manhattan fallback (never throws)."""
    key = (pos1, pos2)
    if key in self._distanceCache:
      return self._distanceCache[key]
    try:
      d = self.getMazeDistance(pos1, pos2)
    except Exception:
      d = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    self._distanceCache[key] = d
    self._distanceCache[(pos2, pos1)] = d
    return d

  def distanceToHome(self, pos):
    """Shortest maze distance from `pos` to any home-boundary cell."""
    return min(self.getMazeDistanceCached(pos, b) for b in self.homeBoundary)

  def getSuccessor(self, gameState, action):
    """Successor state, correcting for half-grid positions."""
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    return successor

  # --- evaluation ----------------------------------------------------------

  def evaluate(self, gameState, action):
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

  def teamScore(self, gameState):
    """Score from our team's perspective (positive == we are winning)."""
    return self.getScore(gameState)

  # --- perception ----------------------------------------------------------

  def updateEatenFood(self, gameState):
    """If a defended dot vanished, an invader ate it; remember the spot. Forget
    it once we reach it (or it reappears) so we stop chasing a stale crumb."""
    currentFood = self.getFoodYouAreDefending(gameState).asList()
    if len(currentFood) < len(self.lastDefendingFood):
      eaten = [f for f in self.lastDefendingFood if f not in currentFood]
      if eaten:
        self.lastEatenFood = eaten[0]
    if self.lastEatenFood is not None:
      myPos = gameState.getAgentPosition(self.index)
      if myPos is not None and \
         self.getMazeDistanceCached(myPos, self.lastEatenFood) <= 1:
        self.lastEatenFood = None
    self.lastDefendingFood = currentFood

  def getTeammate(self, gameState):
    """The AgentState of our teammate (the other agent on our team)."""
    return gameState.getAgentState((self.index + 2) % 4)

  # --- hardened action selection ------------------------------------------

  def chooseAction(self, gameState):
    """
    Pick the highest-value legal action while guaranteeing that we (1) always
    return a legal action, (2) never let one bad evaluation forfeit the turn,
    and (3) stop before the time budget. A shallow, discounted one-step
    lookahead refines the choice when time allows; the immediate value always
    dominates so the first move's safety is never traded away.
    """
    start = time.time()

    try:
      self.updateEatenFood(gameState)
    except Exception:
      pass

    actions = gameState.getLegalActions(self.index)
    if not actions:
      return Directions.STOP
    # Prefer to keep moving: drop STOP when any real move exists.
    if len(actions) > 1 and Directions.STOP in actions:
      actions = [a for a in actions if a != Directions.STOP]

    bestValue = float('-inf')
    bestTies = [actions[0]]

    for action in actions:
      if time.time() - start > TIME_BUDGET:
        break
      try:
        value = self.evaluate(gameState, action)
        # Discounted lookahead: best value reachable one of our moves later.
        if time.time() - start <= TIME_BUDGET:
          successor = self.getSuccessor(gameState, action)
          nextActions = successor.getLegalActions(self.index)
          if len(nextActions) > 1 and Directions.STOP in nextActions:
            nextActions = [a for a in nextActions if a != Directions.STOP]
          bestNext = float('-inf')
          for nextAction in nextActions:
            if time.time() - start > TIME_BUDGET:
              break
            bestNext = max(bestNext, self.evaluate(successor, nextAction))
          if bestNext != float('-inf'):
            value += LOOKAHEAD_DISCOUNT * bestNext
      except Exception:
        # Skip an action that blew up rather than crashing the turn.
        continue

      if value > bestValue:
        bestValue = value
        bestTies = [action]
      elif value == bestValue:
        bestTies.append(action)

    # Random tie-break keeps us unpredictable to adaptive opponents.
    return random.choice(bestTies)


class OffensiveAgent(BaseCaptureAgent):
  """
  Relentless-but-survivable forager. Because food scores immediately, this
  agent keeps eating instead of banking food at home; it only heads home as an
  escape. It evades active ghosts, refuses to get trapped in dead-ends while
  chased, uses capsules under pressure, and hunts scared ghosts for +3.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    currentState = gameState.getAgentState(self.index)
    currentPos = currentState.getPosition()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)

    # --- target assignment: split food with the teammate to avoid doubling up.
    teammate = self.getTeammate(successor)
    teammatePos = teammate.getPosition()
    assignedFood = foodList
    if foodList and teammate.isPacman:
      midY = self.mapHeight // 2
      if self.index % 2 == 0:
        half = [f for f in foodList if f[1] >= midY]
      else:
        half = [f for f in foodList if f[1] < midY]
      assignedFood = half if half else foodList
    if assignedFood:
      features['distanceToFood'] = min(
          self.getMazeDistanceCached(myPos, f) for f in assignedFood)

    distHome = self.distanceToHome(myPos)

    # --- classify visible enemy ghosts on their own turf.
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    activeGhosts, scaredGhosts = [], []
    for e in enemies:
      if e.isPacman or e.getPosition() is None:
        continue
      (scaredGhosts if e.scaredTimer > 1 else activeGhosts).append(e)

    ghostDist = None
    if activeGhosts:
      ghostDist = min(self.getMazeDistanceCached(myPos, g.getPosition())
                      for g in activeGhosts)
      if ghostDist <= 1:
        features['ghostNearby'] = 1            # adjacent: about to die
      features['ghostDistance'] = min(ghostDist, 5)

      # Dead-end danger: avoid pockets we could be trapped in while chased.
      depth = self.deadEndDepth.get(myPos, 0)
      if depth > 0 and ghostDist <= depth + 2:
        features['deadEndDanger'] = depth

    # Detect an actual death: the move bounced us back to our spawn.
    if currentState.isPacman and currentPos != self.start and myPos == self.start:
      features['gotKilled'] = 1

    # --- capsules: a reset button that turns ghosts into +3 prey.
    capsules = self.getCapsules(successor)
    if capsules:
      minCapDist = min(self.getMazeDistanceCached(myPos, c) for c in capsules)
      features['distanceToCapsule'] = minCapDist
      if ghostDist is not None and ghostDist <= 5:
        features['capsuleWhenChased'] = minCapDist

    # --- hunt scared ghosts only if we can reach them before they un-scare.
    if scaredGhosts:
      closest = min(scaredGhosts,
                    key=lambda g: self.getMazeDistanceCached(myPos, g.getPosition()))
      sgDist = self.getMazeDistanceCached(myPos, closest.getPosition())
      if closest.scaredTimer > sgDist + 1:
        features['scaredGhostDistance'] = sgDist

    # --- spread out from the teammate so we cover more food.
    if teammatePos is not None and self.getMazeDistanceCached(myPos, teammatePos) <= 1:
      features['teammateNearby'] = 1

    # --- going home: ONLY as an escape (food is already banked when eaten),
    # or to play safe once almost all food is gone.
    threatened = ghostDist is not None and ghostDist <= 4 and myState.numCarrying >= 1
    if threatened or len(foodList) <= 2:
      features['returnHome'] = distHome

    if action == Directions.STOP:
      features['stop'] = 1
    # Penalize reversing only when safe; doubling back can be the escape.
    rev = Directions.REVERSE[currentState.configuration.direction]
    if action == rev and (ghostDist is None or ghostDist > 4):
      features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {
        'successorScore': 100,
        'distanceToFood': -2,
        'ghostNearby': -2000,        # death = -3 score plus a long respawn
        'ghostDistance': 35,
        'deadEndDanger': -160,
        'gotKilled': -1500,
        'scaredGhostDistance': -8,   # eating a scared ghost is worth +3
        'distanceToCapsule': -2,
        'capsuleWhenChased': -12,
        'teammateNearby': -25,
        'returnHome': -10,
        'stop': -100,
        'reverse': -3,
    }


class DefensiveAgent(BaseCaptureAgent):
  """
  Kill-focused hunter. Each invader killed is +3, so this agent charges and
  eats invaders when it can, keeps a safe gap while scared (touching an invader
  then would hand THEM +3), investigates where food vanished, and patrols the
  central chokepoint otherwise. If badly behind late with nothing to defend, it
  joins the attack.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [e for e in enemies if e.isPacman and e.getPosition() is not None]

    myScore = self.teamScore(successor)
    timeleft = successor.data.timeleft
    losingLate = myScore < 0 and timeleft < 200

    # --- hybrid: badly behind, late, and nothing to chase -> go score food.
    if losingLate and not invaders:
      foodList = self.getFood(successor).asList()
      features['successorScore'] = -len(foodList)
      if foodList:
        features['distanceToFood'] = min(
            self.getMazeDistanceCached(myPos, f) for f in foodList)
      activeGhosts = [e for e in enemies if not e.isPacman
                      and e.getPosition() is not None and e.scaredTimer <= 1]
      if activeGhosts:
        gd = min(self.getMazeDistanceCached(myPos, g.getPosition())
                 for g in activeGhosts)
        if gd <= 1:
          features['ghostNearby'] = 1
        features['ghostDistance'] = min(gd, 4)
      if action == Directions.STOP:
        features['stop'] = 1
      return features

    # --- standard defense.
    # Reward kills directly: moving onto an invader bumps our team score by +3.
    features['killScore'] = myScore
    features['onDefense'] = 0 if myState.isPacman else 1
    features['numInvaders'] = len(invaders)

    if invaders:
      minDist = min(self.getMazeDistanceCached(myPos, e.getPosition())
                    for e in invaders)
      if myState.scaredTimer > 1:
        # Scared: shadow at distance ~2, ready to pounce when the timer drops.
        features['scaredKeepDistance'] = abs(minDist - 2)
      else:
        features['invaderDistance'] = minDist
    elif self.lastEatenFood is not None:
      features['distanceToEaten'] = self.getMazeDistanceCached(myPos, self.lastEatenFood)
    else:
      # Patrol: split the boundary gates with the teammate for wider coverage.
      teammate = self.getTeammate(successor)
      midY = self.mapHeight // 2
      gates = self.homeBoundary
      if not teammate.isPacman:
        if self.index % 2 == 0:
          half = [g for g in self.homeBoundary if g[1] >= midY]
        else:
          half = [g for g in self.homeBoundary if g[1] < midY]
        if half:
          gates = half
      patrolTarget = min(gates, key=lambda g: abs(g[1] - midY))
      features['distanceToCenter'] = self.getMazeDistanceCached(myPos, patrolTarget)

    # Don't cluster with the teammate.
    teammatePos = self.getTeammate(successor).getPosition()
    if teammatePos is not None and self.getMazeDistanceCached(myPos, teammatePos) <= 1:
      features['teammateNearby'] = 1

    if action == Directions.STOP:
      features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev:
      features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [e for e in enemies if e.isPacman and e.getPosition() is not None]
    myScore = self.teamScore(gameState)
    if myScore < 0 and gameState.data.timeleft < 200 and not invaders:
      # Hybrid offensive weights (defender temporarily attacks).
      return {
          'successorScore': 100,
          'distanceToFood': -2.5,
          'ghostNearby': -1500,
          'ghostDistance': 20,
          'stop': -100,
      }
    return {
        'killScore': 100,            # +3 per kill makes hunting worth it
        'numInvaders': -1000,
        'onDefense': 200,
        'invaderDistance': -15,
        'scaredKeepDistance': -10,
        'distanceToEaten': -4,
        'distanceToCenter': -1.5,
        'teammateNearby': -25,
        'stop': -100,
        'reverse': -2,
    }
