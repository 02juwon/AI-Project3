# geminiTeam.py
# ---------
# Berkeley Pacman Capture the Flag - Gemini Adaptive Strategic Team (Champion Version)
#
# This file implements the Gemini variant team, adhering strictly to the Project 3 guidelines.
# Optimized after league success:
#   - Pre-computes dead-ends using a mathematical degree-reduction algorithm to avoid getting trapped.
#   - Dynamic target grid partitioning: only splits food grids if BOTH agents are actively attacking.
#   - Teammate collision avoidance to prevent clustering.
#   - Dynamic boundary gate patrol division to cover maximum choke points.
#   - Refined scared defender shadowing: charges at the invader when scaredTimer is exactly 1 step.
#   - Hardened with lookahead search and strict time management (0.8s budget).
# ---------

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'GeminiOffensiveAgent', second = 'GeminiDefensiveAgent'):
  """
  Returns a list of two agents that form the team.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

# Soft per-action time budget to keep well within the 1-second limit.
TIME_BUDGET = 0.8

class BaseCaptureAgent(CaptureAgent):
  """
  A feature-based reflex agent base class shared by Gemini agents.
  Provides a maze-distance cache, home-boundary pre-computation,
  dead-end mapping, eaten-food detection, and a robust search execution loop.
  """

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)

    self.walls = gameState.getWalls()
    self.mapWidth = self.walls.width
    self.mapHeight = self.walls.height

    # Cache for maze-distance lookups (keeps us inside the per-turn budget).
    self._distanceCache = {}

    # Pre-compute the non-wall cells along our side of the home boundary.
    self.homeBoundary = self.computeHomeBoundary(gameState)

    # Defensive patrol center: the boundary cell nearest the vertical middle.
    if self.homeBoundary:
      midY = self.mapHeight // 2
      self.defensiveCenter = min(self.homeBoundary,
                                 key=lambda p: abs(p[1] - midY))
    else:
      self.defensiveCenter = self.start

    # Track defended food to detect where invaders eat.
    self.lastDefendingFood = self.getFoodYouAreDefending(gameState).asList()
    self.lastEatenFood = None

    # --- ADVANCED: Pre-compute Dead-End Depths using Degree-Reduction ---
    self.deadEndDepths = {}
    all_positions = []
    for x in range(self.mapWidth):
      for y in range(self.mapHeight):
        if not self.walls[x][y]:
          all_positions.append((x, y))

    # Build adjacency mapping
    neighbors = {}
    for p in all_positions:
      neighbors[p] = []
      for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        np_pos = (p[0]+dx, p[1]+dy)
        if np_pos in all_positions:
          neighbors[p].append(np_pos)

    # Iteratively prune leaves (nodes with degree <= 1)
    degrees = {p: len(neighbors[p]) for p in all_positions}
    queue = [p for p in all_positions if degrees[p] <= 1]
    dead_ends = {}

    while queue:
      p = queue.pop(0)
      # Depth is 1 + maximum depth of its dead-end neighbors
      d_neighs = [dead_ends[n] for n in neighbors[p] if n in dead_ends]
      depth = 1 + (max(d_neighs) if d_neighs else 0)
      dead_ends[p] = depth

      for n in neighbors[p]:
        if n not in dead_ends:
          degrees[n] -= 1
          if degrees[n] <= 1:
            queue.append(n)

    # Populate final mapping: non-deadends map to depth 0
    self.deadEndDepths = {p: dead_ends.get(p, 0) for p in all_positions}

  def computeHomeBoundary(self, gameState):
    """Return the legal (non-wall) cells on our side of the border."""
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
    """Memoized maze distance. Falls back to Manhattan distance if BFS fails."""
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
    if not self.homeBoundary:
      return self.getMazeDistanceCached(pos, self.start)
    return min(self.getMazeDistanceCached(pos, b) for b in self.homeBoundary)

  def getSuccessor(self, gameState, action):
    """Find the successor state, correcting for half-grid positions."""
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    return successor

  def evaluate(self, gameState, action):
    """Linear combination of features and weights."""
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
    """Compare defending food to detect eating location of invisible Pacmen."""
    currentFood = self.getFoodYouAreDefending(gameState).asList()
    if len(currentFood) < len(self.lastDefendingFood):
      eaten = [f for f in self.lastDefendingFood if f not in currentFood]
      if eaten:
        self.lastEatenFood = eaten[0]
    # Reset tracking if we reach the eaten food spot.
    if self.lastEatenFood is not None and \
       gameState.getAgentPosition(self.index) == self.lastEatenFood:
      self.lastEatenFood = None
    self.lastDefendingFood = currentFood

  def getVisibleEnemies(self, gameState):
    """Enemy AgentState objects whose exact position is currently known."""
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    return [e for e in enemies if e.getPosition() is not None]

  def chooseAction(self, gameState):
    """
    Looks ahead up to 2 steps using a soft time budget (0.8s) and handles exceptions.
    """
    start_time = time.time()
    
    # Track defended food.
    try:
      self.updateEatenFood(gameState)
    except Exception:
      pass

    actions = gameState.getLegalActions(self.index)
    if not actions:
      return Directions.STOP

    # Filter out STOP action if possible to keep agent moving.
    if len(actions) > 1 and Directions.STOP in actions:
      actions = [a for a in actions if a != Directions.STOP]

    bestAction = actions[0]
    bestValue = float('-inf')
    bestTies = []

    for a1 in actions:
      if time.time() - start_time > TIME_BUDGET:
        break
      try:
        # Step 1 evaluation
        val1 = self.evaluate(gameState, a1)
        
        # Step 2 lookahead
        s1 = self.getSuccessor(gameState, a1)
        a2s = s1.getLegalActions(self.index)
        if len(a2s) > 1 and Directions.STOP in a2s:
          a2s = [a for a in a2s if a != Directions.STOP]
          
        val2 = float('-inf')
        for a2 in a2s:
          if time.time() - start_time > TIME_BUDGET:
            break
          val2 = max(val2, self.evaluate(s1, a2))
        
        # Combined score (current step + lookahead)
        score = val2 if val2 != float('-inf') else val1
      except Exception:
        # Exception fallback
        try:
          score = self.evaluate(gameState, a1)
        except Exception:
          continue

      if score > bestValue:
        bestValue = score
        bestTies = [a1]
      elif score == bestValue:
        bestTies.append(a1)

    if bestTies:
      return random.choice(bestTies)
    return bestAction


class GeminiOffensiveAgent(BaseCaptureAgent):
  """
  Gemini Offensive Agent. 
  Implements smart target splitting (conditional on teammate role), dead-end danger avoidance,
  score-dependent returning thresholds, capsule threat evasion, and scared ghost hunting.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Base Score is negated food remaining count
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)

    # Teammate state tracking for target partitioning & collision avoidance
    teammate_index = (self.index + 2) % 4
    teammate_state = successor.getAgentState(teammate_index)
    teammate_pos = teammate_state.getPosition()

    # Calculate carrying and score states
    carrying = myState.numCarrying
    my_score = self.getScore(successor) if self.red else -self.getScore(successor)
    timeleft = successor.data.timeleft

    is_winning = (my_score > 0)
    is_losing = (my_score < 0)
    is_endgame = (timeleft < 200)

    # Check if teammate is also acting offensively (meaning both are attacking)
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [e for e in enemies if e.isPacman and e.getPosition() is not None]
    teammate_is_offensive = teammate_state.isPacman or (is_losing and is_endgame and len(invaders) == 0)

    # 1. Target Grid Partitioning
    if foodList:
      midY = self.mapHeight // 2
      if teammate_is_offensive:
        # Split target food to avoid redundant paths
        if self.index % 2 == 0:
          assigned_food = [f for f in foodList if f[1] >= midY]
        else:
          assigned_food = [f for f in foodList if f[1] < midY]
        if not assigned_food:
          assigned_food = foodList
      else:
        # Teammate is defending; this agent can target ALL food freely
        assigned_food = foodList
        
      minFoodDist = min(self.getMazeDistanceCached(myPos, f) for f in assigned_food)
      features['distanceToFood'] = minFoodDist

    # Get distance back to home border
    distHome = self.distanceToHome(myPos)
    features['distanceToHome'] = distHome

    # Adaptive Returning Threshold & Safety margins
    if is_winning and is_endgame:
      returning_threshold = 1   # Lock in score immediately when winning in endgame
      ghost_safety_margin = 6
    elif is_winning:
      returning_threshold = 3   # Play safe when winning
      ghost_safety_margin = 5
    elif is_losing and is_endgame:
      returning_threshold = 7   # High risk, carry as much as possible to secure lead
      ghost_safety_margin = 3
    elif is_losing:
      returning_threshold = 5   # Carry more when losing
      ghost_safety_margin = 4
    else: # Tied
      returning_threshold = 4
      ghost_safety_margin = 4

    # 2. Ghost Threat Classification
    activeGhosts = []
    scaredGhosts = []
    
    for e in enemies:
      if e.isPacman or e.getPosition() is None:
        continue
      if e.scaredTimer <= 1:
        activeGhosts.append(e)
      else:
        scaredGhosts.append(e)

    # Active ghost fleeing
    ghostDist = None
    if activeGhosts:
      ghostDist = min(self.getMazeDistanceCached(myPos, g.getPosition()) for g in activeGhosts)
      if ghostDist <= 1:
        features['ghostNearby'] = 1 # Adjacent: major threat
      features['ghostDistance'] = min(ghostDist, ghost_safety_margin)

    # 3. ADVANCED: Dead-End Danger Avoidance
    my_dead_end_depth = self.deadEndDepths.get(myPos, 0)
    if my_dead_end_depth > 0 and ghostDist is not None and ghostDist <= 5:
      # Heavily penalize entering/staying deep in a dead end when a ghost is nearby
      features['deadEndDanger'] = my_dead_end_depth

    # 4. Capsule Centric Tactics
    capsules = self.getCapsules(successor)
    if capsules:
      minCapDist = min(self.getMazeDistanceCached(myPos, c) for c in capsules)
      features['distanceToCapsule'] = minCapDist
      # Actively target capsule if chased by a ghost
      if ghostDist is not None and ghostDist <= 5:
        features['capsuleWhenChased'] = minCapDist

    # 5. Scared Ghost Exploitation
    if scaredGhosts:
      closestScared = min(scaredGhosts, key=lambda g: self.getMazeDistanceCached(myPos, g.getPosition()))
      sgDist = self.getMazeDistanceCached(myPos, closestScared.getPosition())
      # Chase scared ghost only if we can definitely reach it
      if closestScared.scaredTimer > sgDist + 1:
        features['scaredGhostDistance'] = sgDist

    # 6. Teammate Collision Avoidance
    if teammate_pos is not None:
      t_dist = self.getMazeDistanceCached(myPos, teammate_pos)
      if t_dist <= 1:
        features['teammateNearby'] = 1

    # 7. Return Home Triggers
    shouldReturn = False
    if carrying >= returning_threshold:
      shouldReturn = True
    elif carrying >= 1 and ghostDist is not None and ghostDist <= 4:
      shouldReturn = True
    elif len(foodList) <= 2: # Clean up and return
      shouldReturn = True
      
    if shouldReturn:
      features['returnHome'] = distHome

    # Penalties
    if action == Directions.STOP:
      features['stop'] = 1

    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev and (ghostDist is None or ghostDist > 4):
      features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {
        'successorScore': 100,
        'distanceToFood': -2,
        'ghostNearby': -2000,      # Increased penalty to enforce survival
        'ghostDistance': 35,
        'deadEndDanger': -150,     # Strong penalty to avoid dead ends when ghosts are close
        'scaredGhostDistance': -6,
        'distanceToCapsule': -2,
        'capsuleWhenChased': -10,
        'teammateNearby': -20,     # Prevent clustering with teammate
        'returnHome': -15,         # Sharper homing behavior
        'stop': -100,
        'reverse': -3,
    }


class GeminiDefensiveAgent(BaseCaptureAgent):
  """
  Gemini Defensive Agent.
  Patrols boundary choke points, estimates invader positions, shadows invaders
  when scared (maintaining distance 2, but attacks when scaredTimer reaches 1),
  avoids teammate collision, and transitions to attack mode when losing in endgame.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Check game state
    my_score = self.getScore(successor) if self.red else -self.getScore(successor)
    timeleft = successor.data.timeleft
    is_losing = (my_score < 0)
    is_endgame = (timeleft < 200)

    # Identify Invaders
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [e for e in enemies if e.isPacman and e.getPosition() is not None]
    features['numInvaders'] = len(invaders)

    # Teammate state tracking
    teammate_index = (self.index + 2) % 4
    teammate_state = successor.getAgentState(teammate_index)
    teammate_pos = teammate_state.getPosition()

    # Teammate defensive status
    teammate_is_defending = not teammate_state.isPacman

    # 1. Hybrid Role Transition: If losing in endgame and no invaders, act offensively!
    if is_losing and is_endgame and len(invaders) == 0:
      foodList = self.getFood(successor).asList()
      features['successorScore'] = -len(foodList)
      if foodList:
        midY = self.mapHeight // 2
        # Target opposite half of the primary offensive agent
        if self.index % 2 == 0:
          assigned_food = [f for f in foodList if f[1] >= midY]
        else:
          assigned_food = [f for f in foodList if f[1] < midY]
        if not assigned_food:
          assigned_food = foodList
        minFoodDist = min(self.getMazeDistanceCached(myPos, f) for f in assigned_food)
        features['distanceToFood'] = minFoodDist

      distHome = self.distanceToHome(myPos)
      features['distanceToHome'] = distHome

      # Evasion of active ghosts
      activeGhosts = [e for e in enemies if not e.isPacman and e.getPosition() is not None and e.scaredTimer <= 1]
      if activeGhosts:
        ghostDist = min(self.getMazeDistanceCached(myPos, g.getPosition()) for g in activeGhosts)
        if ghostDist <= 1:
          features['ghostNearby'] = 1
        features['ghostDistance'] = min(ghostDist, 3)

      carrying = myState.numCarrying
      if carrying >= 2 or len(foodList) <= 2:
        features['returnHome'] = distHome

      if teammate_pos is not None:
        t_dist = self.getMazeDistanceCached(myPos, teammate_pos)
        if t_dist <= 1:
          features['teammateNearby'] = 1

      if action == Directions.STOP:
        features['stop'] = 1
      return features

    # --- Standard Defensive Behavior ---
    features['onDefense'] = 0 if myState.isPacman else 1

    if invaders:
      dists = [self.getMazeDistanceCached(myPos, e.getPosition()) for e in invaders]
      minDist = min(dists)

      # 2. Refined Scared Shadowing: Maintain distance 2, but attack if timer is exactly 1 step
      if myState.scaredTimer > 1:
        features['scaredKeepDistance'] = abs(minDist - 2)
      else:
        # Active chase when not scared, or about to un-scare in 1 step
        features['invaderDistance'] = minDist
    else:
      # 3. Dynamic Boundary Patrol Split
      midY = self.mapHeight // 2
      if teammate_is_defending:
        # Split boundary gates with teammate to maximize coverage
        if self.index % 2 == 0:
          my_gates = [g for g in self.homeBoundary if g[1] >= midY]
        else:
          my_gates = [g for g in self.homeBoundary if g[1] < midY]
        if not my_gates:
          my_gates = self.homeBoundary
      else:
        # Only defender: has full boundary access
        my_gates = self.homeBoundary

      if self.lastEatenFood is not None:
        closestGate = min(my_gates, key=lambda g: self.getMazeDistanceCached(g, self.lastEatenFood))
        features['distanceToPatrol'] = self.getMazeDistanceCached(myPos, closestGate)
      else:
        my_center = min(my_gates, key=lambda g: abs(g[1] - midY))
        features['distanceToCenter'] = self.getMazeDistanceCached(myPos, my_center)

    # Teammate proximity penalty to avoid blocking
    if teammate_pos is not None:
      t_dist = self.getMazeDistanceCached(myPos, teammate_pos)
      if t_dist <= 1:
        features['teammateNearby'] = 1

    if action == Directions.STOP:
      features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev:
      features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    my_score = self.getScore(gameState) if self.red else -self.getScore(gameState)
    timeleft = gameState.data.timeleft
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [e for e in enemies if e.isPacman and e.getPosition() is not None]

    if my_score < 0 and timeleft < 200 and len(invaders) == 0:
      # Hybrid Offensive Weights
      return {
          'successorScore': 100,
          'distanceToFood': -2.5,
          'ghostNearby': -1000,
          'ghostDistance': 15,
          'teammateNearby': -20,
          'returnHome': -8,
          'stop': -100,
      }

    # Standard Defensive Weights
    return {
        'numInvaders': -1000,
        'onDefense': 100,
        'invaderDistance': -15,    # Increased priority to hunt down invaders
        'scaredKeepDistance': -10,
        'distanceToPatrol': -4,
        'distanceToCenter': -1.5,
        'teammateNearby': -20,     # Prevent blocking the other defender
        'stop': -100,
        'reverse': -2,
    }
