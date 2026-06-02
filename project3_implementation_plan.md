# Project 3 구현 방안 추천: Pacman Competition

## 목표

이 문서는 `Project#3: Pacman Competition`에서 `myTeam.py`를 어떻게 설계하면 좋을지 정리한 구현 전략이다.

과제의 핵심 조건은 다음과 같다.

- 제출 파일은 `myTeam.py` 하나만 가능하다.
- 별도의 모델 파일, Q-table 파일, weight 파일을 불러올 수 없다.
- Python standard library와 NumPy만 사용할 수 있다.
- 매 action은 1초 안에 반환해야 한다.
- baselineTeam 상대 20게임에서 65% 이상 승률을 달성해야 본선에 진출한다.

따라서 추천 방향은 **복잡한 딥러닝/RL 모델**이 아니라, 제한 시간 안에서 안정적으로 동작하는 **feature-based reflex agent** 또는 **approximate Q-style agent**이다.

---

# 1. baselineTeam을 이기는 방법

## 1.1 기본 전략

baselineTeam을 이기는 데 가장 중요한 것은 화려한 알고리즘보다 **안정성**이다.

baselineTeam은 대체로 단순한 공격/수비 행동을 하므로, 다음 세 가지를 잘하면 충분히 이길 가능성이 높다.

1. **공격 agent가 food를 꾸준히 먹는다.**
2. **먹은 food를 너무 욕심내지 않고 안전하게 돌아온다.**
3. **수비 agent가 우리 진영에 들어온 상대 Pacman을 빠르게 잡는다.**

즉, 처음 목표는 “최강 agent”가 아니라 **baselineTeam을 안정적으로 이기는 agent**를 만드는 것이다.

---

## 1.2 팀 구성 추천

가장 추천하는 구조는 다음과 같다.

```text
Agent 1: OffensiveAgent
Agent 2: DefensiveAgent
```

또는 조금 더 강하게 만들려면 다음 구조를 사용한다.

```text
Agent 1: HybridAgent, 기본은 공격
Agent 2: HybridAgent, 기본은 수비
```

초기 구현에서는 역할을 명확히 나누는 것이 좋다.

| Agent | 기본 역할 | 핵심 목표 |
|---|---|---|
| OffensiveAgent | 공격 | 상대 food 먹기, 위험하면 귀환 |
| DefensiveAgent | 수비 | 우리 food 지키기, 침입자 추적 |

---

## 1.3 OffensiveAgent 설계

공격 agent는 매 턴 가능한 action들을 평가해서 가장 높은 점수를 주는 action을 선택한다.

### 핵심 feature

공격 agent에서 사용할 feature는 다음 정도면 충분하다.

| Feature | 의미 | 방향 |
|---|---|---|
| `successorScore` | action 후 점수 | 클수록 좋음 |
| `distanceToFood` | 가장 가까운 food까지의 거리 | 작을수록 좋음 |
| `distanceToHome` | 우리 진영까지의 거리 | 상황에 따라 작을수록 좋음 |
| `numCarrying` | 현재 들고 있는 food 개수 | 많으면 귀환 고려 |
| `distanceToGhost` | 상대 ghost까지의 거리 | 가까우면 위험 |
| `distanceToCapsule` | capsule까지의 거리 | 위험할 때 작을수록 좋음 |
| `stop` | STOP action 여부 | 가능하면 피함 |
| `reverse` | 직전 방향 반대 여부 | 불필요한 왕복 방지 |

---

## 1.4 공격 agent의 행동 규칙

공격 agent는 상황에 따라 mode를 나누는 것이 좋다.

```text
1. Safe Attack Mode
2. Escape Mode
3. Return Home Mode
```

### Safe Attack Mode

상대 ghost가 멀리 있고, 들고 있는 food가 적을 때는 가장 가까운 food를 향해 간다.

```text
조건 예시:
- visible enemy ghost가 없거나 충분히 멀다.
- carrying food 개수가 0~2개 정도다.
```

이때는 `distanceToFood`를 강하게 줄이면 된다.

```text
좋은 action = food에 가까워지는 action
```

---

### Escape Mode

상대 ghost가 가까우면 food보다 생존이 우선이다.

```text
조건 예시:
- enemy ghost와의 거리가 1~3 이하
- 상대 ghost가 scared 상태가 아님
```

이때는 다음 우선순위를 적용한다.

```text
1. ghost와 멀어지기
2. capsule이 가까우면 capsule로 이동
3. 우리 진영으로 복귀
```

단순히 food만 따라가면 baselineTeam에게 잡히면서 점수를 잃는다. 공격 agent는 “먹고 살아서 돌아오는 것”이 핵심이다.

---

### Return Home Mode

food를 어느 정도 먹었거나 시간이 부족하면 귀환해야 한다.

```text
조건 예시:
- carrying food 개수 >= 3
- 남은 시간이 적음
- score가 이미 앞서고 있음
- 주변에 ghost가 보임
```

이때는 `distanceToHome`을 줄이는 action에 높은 점수를 준다.

추천 기준은 다음과 같다.

```text
carrying >= 3이면 귀환 고려
carrying >= 5이면 강제 귀환
게임 후반이면 carrying >= 1이어도 귀환 고려
```

baselineTeam을 이기는 목적이라면 지나친 욕심보다 안정적인 귀환이 훨씬 중요하다.

---

## 1.5 DefensiveAgent 설계

수비 agent는 우리 진영을 지키는 agent이다.

### 핵심 feature

| Feature | 의미 | 방향 |
|---|---|---|
| `onDefense` | 현재 ghost 상태인지 | ghost 상태 유지 |
| `numInvaders` | 우리 진영에 들어온 상대 Pacman 수 | 작을수록 좋음 |
| `invaderDistance` | 가장 가까운 invader까지 거리 | 작을수록 좋음 |
| `distanceToDefensiveCenter` | 수비 중심 위치까지 거리 | 작을수록 좋음 |
| `stop` | STOP 여부 | 피함 |
| `reverse` | 방향 반전 여부 | 약하게 피함 |

---

## 1.6 수비 agent의 행동 규칙

수비 agent는 다음 우선순위를 따르면 된다.

```text
1. 보이는 invader가 있으면 추적한다.
2. invader가 안 보이면 최근 먹힌 food 위치로 이동한다.
3. 정보가 없으면 중앙 choke point 근처를 순찰한다.
```

### Invader 추적

상대 Pacman이 보이면 가장 가까운 invader를 향해 이동한다.

```text
좋은 action = invaderDistance를 줄이는 action
```

상대 Pacman을 잡으면 3점이므로, 수비 성공은 매우 크다.

---

### 먹힌 food 기반 추적

상대가 시야 밖에 있더라도, 이전 턴과 현재 턴의 우리 food 목록을 비교하면 어떤 food가 사라졌는지 알 수 있다.

```text
previousFood - currentFood = eatenFood
```

사라진 food 위치가 있으면 그 위치 근처로 이동한다.

이 기능은 baselineTeam 상대에서 매우 효과적이다.

---

### 수비 중심 위치 설정

invader가 보이지 않을 때는 맵 중앙 근처, 특히 우리 진영과 상대 진영의 경계 근처를 지키는 것이 좋다.

```text
defensiveCenter = 우리 진영 중앙선 근처의 통로 위치
```

단순히 시작 위치에 서 있는 것보다 중앙선 근처를 지키는 것이 상대 Pacman을 빨리 발견할 수 있다.

---

## 1.7 baselineTeam 상대 추천 구현 순서

다음 순서대로 구현하는 것을 추천한다.

### Step 1. 기본 Reflex 구조 만들기

```text
getFeatures(gameState, action)
getWeights(gameState, action)
evaluate(gameState, action)
chooseAction(gameState)
```

각 action에 대해 feature와 weight를 곱해서 점수를 계산한다.

---

### Step 2. OffensiveAgent 구현

먼저 공격 agent가 food를 먹고 돌아오게 만든다.

최소 구현 목표는 다음과 같다.

```text
- 가장 가까운 food로 이동
- ghost가 가까우면 도망
- carrying이 많으면 home으로 복귀
- STOP 금지
```

---

### Step 3. DefensiveAgent 구현

다음 기능을 넣는다.

```text
- invader가 보이면 추적
- invader가 안 보이면 중앙선 근처 순찰
- 이전 food와 현재 food를 비교해서 먹힌 food 위치 추적
```

---

### Step 4. autograder로 반복 테스트

다음 명령어로 테스트한다.

```bash
python autograder.py -q
```

확인할 것은 단순 승률만이 아니다.

```text
- Red일 때 승률
- Blue일 때 승률
- 평균 action time
- max action time
- 자주 지는 replay 상황
```

---

## 1.8 baselineTeam을 이기기 위한 weight 예시 방향

정확한 숫자는 직접 실험해야 하지만, 방향은 다음과 같다.

### OffensiveAgent weight 방향

| Feature | Weight 방향 |
|---|---:|
| `successorScore` | 매우 크게 + |
| `distanceToFood` | - |
| `distanceToGhost` | ghost가 가까우면 크게 + |
| `distanceToHome` | 귀환 모드에서 - |
| `distanceToCapsule` | 위험할 때 - |
| `stop` | 크게 - |
| `reverse` | 약하게 - |

### DefensiveAgent weight 방향

| Feature | Weight 방향 |
|---|---:|
| `numInvaders` | 매우 크게 - |
| `invaderDistance` | - |
| `onDefense` | + |
| `distanceToDefensiveCenter` | - |
| `stop` | 크게 - |
| `reverse` | 약하게 - |

---

# 2. 다른 상대와의 경쟁을 이기는 방법

baselineTeam을 넘은 이후에는 단순 Reflex만으로는 한계가 있다. 다른 학생 팀은 baseline보다 공격적이고, capsule을 잘 쓰거나, 수비를 더 강하게 만들 가능성이 높다.

따라서 본선 경쟁용 agent는 다음 요소를 추가해야 한다.

---

## 2.1 고정 역할보다 동적 역할 전환

baselineTeam 상대에서는 공격 1명, 수비 1명이 안정적이다. 하지만 다른 팀과 경쟁할 때는 상황에 따라 역할을 바꾸는 것이 좋다.

예를 들어 다음과 같이 전환한다.

```text
상황 A: 우리가 지고 있음
→ 두 agent 모두 공격 비중 증가

상황 B: 우리가 이기고 있음
→ 한 명은 수비 고정, 한 명은 안전 공격

상황 C: 상대 Pacman 2명이 우리 진영에 들어옴
→ 두 agent 모두 수비 전환

상황 D: 상대 ghost가 scared 상태
→ 공격 강화
```

즉, agent를 다음처럼 설계한다.

```text
HybridAgent:
    if urgentDefense:
        act as defender
    elif safeAttackChance:
        act as attacker
    elif carryingManyFood:
        return home
    else:
        balanced action
```

---

## 2.2 Choke point 장악

경쟁전에서는 단순히 가장 가까운 food만 먹는 agent는 쉽게 잡힌다. 맵에는 중앙선을 통과하는 좁은 길, 즉 **choke point**가 있다.

수비 agent는 이 choke point를 장악해야 한다.

```text
좋은 수비 위치 = 상대가 우리 진영으로 들어올 때 자주 지나가는 중앙 통로
```

구현 방법은 다음과 같다.

1. 맵의 중앙 x 좌표를 찾는다.
2. 중앙선 근처에서 wall이 아닌 위치들을 찾는다.
3. 그중 우리 food와 가까운 위치 또는 통로가 좁은 위치를 patrol point로 정한다.
4. invader가 보이지 않으면 patrol point를 순찰한다.

이렇게 하면 상대가 보이기 전부터 좋은 위치를 잡을 수 있다.

---

## 2.3 Dead-end 회피

다른 팀과 경쟁할 때 가장 큰 문제는 공격 agent가 막다른 길, 즉 **dead-end**에 들어갔다가 ghost에게 잡히는 것이다.

따라서 offensive agent는 food를 먹기 전에 다음을 판단해야 한다.

```text
이 food가 dead-end 안에 있는가?
주변에 ghost가 있는가?
탈출 경로가 있는가?
```

구현 난이도를 낮추려면 다음 정도만 해도 된다.

```text
- 현재 위치에서 legal neighbor 수가 1이면 dead-end 가능성 높음
- food까지 가는 경로 주변에 ghost가 가까우면 해당 food 회피
- ghost와 home boundary 사이에 내가 끼이면 위험으로 판단
```

더 강하게 하려면 미리 모든 위치에 대해 dead-end depth를 계산할 수 있다.

```text
deadEndDepth[position] = 막다른 길 안쪽으로 얼마나 깊은 위치인지
```

`deadEndDepth`가 높은 위치는 ghost가 근처에 있을 때 피한다.

---

## 2.4 Capsule을 전략적으로 사용

baselineTeam 상대에서는 capsule을 단순히 위험할 때 먹어도 충분할 수 있다. 하지만 경쟁전에서는 capsule이 매우 중요하다.

추천 전략은 다음과 같다.

```text
1. ghost가 가까울 때 capsule로 이동한다.
2. capsule을 먹은 직후에는 food를 많이 먹는다.
3. 상대 ghost scared timer가 줄어들면 귀환한다.
```

특히 capsule을 먹고 나서도 무작정 공격하면 안 된다.

```text
scaredTimer가 충분히 남음 → 공격
scaredTimer가 얼마 안 남음 → 귀환
```

---

## 2.5 Carrying food 관리

경쟁전에서는 food를 많이 들고 있다가 죽는 것이 치명적이다.

추천 규칙은 다음과 같다.

```text
carrying 1~2개: 계속 공격 가능
carrying 3~4개: ghost 위치를 보고 판단
carrying 5개 이상: 귀환 우선
후반부: carrying 1개라도 귀환 고려
```

또한 score 상황도 반영해야 한다.

```text
우리가 이기는 중 → 무리하지 말고 귀환
우리가 지는 중 → 위험을 감수하고 추가 공격
```

---

## 2.6 상대 위치 추론

규칙상 상대 위치는 Manhattan distance 5 이내에서만 정확히 보인다. 멀리 있으면 noisy distance만 주어진다.

고급 전략에서는 상대 위치를 완전히 모르더라도 다음 정보로 추론한다.

```text
- noisy distance
- 최근 먹힌 우리 food 위치
- 상대가 마지막으로 관측된 위치
- 상대가 이동 가능한 최대 범위
```

처음부터 복잡한 belief tracking을 구현할 필요는 없지만, 최소한 다음은 넣는 것이 좋다.

```text
lastSeenOpponentPosition
lastEatenFoodPosition
```

수비 agent는 invader가 사라졌을 때 마지막으로 본 위치 또는 먹힌 food 위치로 이동한다.

---

## 2.7 짧은 depth search 추가

경쟁전에서는 단일 action만 평가하는 Reflex agent보다, 2~3 step 앞을 보는 agent가 더 강하다.

하지만 시간 제한이 있으므로 깊은 search는 위험하다.

추천 방식은 다음과 같다.

```text
공격 agent:
- depth 2 정도의 action sequence 평가
- 첫 action만 실제로 선택

수비 agent:
- invader가 보일 때만 maze distance 기반 추적
- 그 외에는 reflex 평가
```

주의할 점은 매 턴 모든 action sequence를 깊게 탐색하면 시간이 초과될 수 있다는 것이다.

예를 들어 legal action이 평균 5개이고 depth 3이면 최대 125개 후보를 평가한다. 여기에 maze distance 계산이 반복되면 1초 제한을 넘을 수 있다.

따라서 다음을 적용한다.

```text
- STOP 제거
- 명백히 위험한 action 먼저 제거
- maze distance 결과 캐싱
- depth는 2 정도로 제한
```

---

## 2.8 Maze distance 캐싱

Pacman 맵에서 단순 Manhattan distance는 벽을 고려하지 못한다. 따라서 가능하면 maze distance를 써야 한다.

하지만 maze distance를 매번 BFS로 계산하면 느릴 수 있다.

추천 방식은 다음과 같다.

```python
self.distancer.getDistance(pos1, pos2)
```

가능하다면 초기화 단계에서 distance calculator를 준비하고, 반복 계산되는 위치 쌍은 캐싱한다.

```text
distanceCache[(pos1, pos2)] = distance
```

시간 제한이 중요한 과제이므로, 계산 최적화는 실제 승률에도 영향을 준다.

---

## 2.9 경쟁전용 행동 우선순위

경쟁전에서 agent는 다음 우선순위를 가지는 것이 좋다.

### 공격 우선순위

```text
1. 죽지 않기
2. 들고 있는 food를 안전하게 귀환시키기
3. capsule 기회 활용하기
4. 가까운 food 먹기
5. dead-end food는 상황 보고 먹기
```

### 수비 우선순위

```text
1. 보이는 invader 잡기
2. 먹힌 food 위치로 이동하기
3. choke point 장악하기
4. score가 많이 지고 있으면 공격 지원하기
```

---

## 2.10 본선용 개선 순서

baselineTeam을 이기는 agent를 만든 뒤, 다음 순서로 개선하는 것이 좋다.

### Step 1. Replay 분석

autograder가 만든 replay를 보면서 다음을 확인한다.

```text
- 언제 죽는가?
- 어느 위치에서 자주 잡히는가?
- food를 먹고도 왜 귀환하지 못하는가?
- 수비 agent가 침입자를 놓치는 이유는 무엇인가?
```

---

### Step 2. 귀환 조건 조정

가장 먼저 조정할 것은 carrying food 기준이다.

```text
너무 자주 죽음 → 더 빨리 귀환
점수를 못 냄 → 귀환 기준 완화
```

---

### Step 3. 수비 patrol point 개선

수비 agent가 쓸데없이 움직이면 침입자를 놓친다. 중앙선 근처 choke point를 더 잘 잡게 수정한다.

---

### Step 4. Capsule과 scaredTimer 활용

상대 ghost가 scared 상태일 때는 공격하고, 시간이 끝나기 전에 돌아오게 만든다.

---

### Step 5. 위험 지역 회피

자주 죽는 위치를 분석해서 dead-end penalty 또는 ghost danger penalty를 강화한다.

---

# 3. 최종 추천 구조

최종적으로는 다음 구조를 추천한다.

```text
myTeam.py
│
├── createTeam(...)
│
├── BaseCaptureAgent
│   ├── registerInitialState(...)
│   ├── chooseAction(...)
│   ├── getFeatures(...)
│   ├── getWeights(...)
│   ├── evaluate(...)
│   ├── getMazeDistanceCached(...)
│   ├── getHomeBoundary(...)
│   ├── getDefensiveCenter(...)
│   └── detectEatenFood(...)
│
├── OffensiveAgent
│   ├── getMode(...)
│   ├── getFeatures(...)
│   └── getWeights(...)
│
└── DefensiveAgent
    ├── getFeatures(...)
    └── getWeights(...)
```

또는 본선용으로는 다음 구조도 좋다.

```text
HybridAgent
├── decideRole(...)
├── evaluateOffense(...)
├── evaluateDefense(...)
├── evaluateReturn(...)
└── chooseAction(...)
```

---

# 4. 구현 난이도별 추천

## 최소 통과용

```text
Feature-based OffensiveAgent + DefensiveAgent
```

필수 기능:

```text
- 가까운 food 먹기
- ghost 가까우면 도망
- carrying 많으면 귀환
- invader 보이면 추적
- invader 없으면 중앙선 순찰
```

---

## 안정적인 baseline 승리용

```text
Feature-based agent + 먹힌 food 추적 + capsule 활용
```

추가 기능:

```text
- previousFood 비교
- lastEatenFood 추적
- capsule 근처에서 공격/도망 판단
- score와 timeleft 기반 귀환
```

---

## 본선 경쟁용

```text
HybridAgent + choke point + dead-end 회피 + 짧은 depth search
```

추가 기능:

```text
- 동적 역할 전환
- choke point patrol
- dead-end penalty
- scaredTimer 활용
- maze distance caching
- depth 2 action sequence 평가
```

---

# 5. 최종 결론

가장 추천하는 구현 전략은 다음과 같다.

```text
1단계: 공격 1명 + 수비 1명의 feature-based reflex agent 구현
2단계: baselineTeam 상대로 65% 이상 승률 확보
3단계: 먹힌 food 추적, capsule 활용, 귀환 조건 개선
4단계: choke point 장악, dead-end 회피, 동적 역할 전환 추가
5단계: replay를 보면서 weight와 조건을 반복 조정
```

처음부터 복잡한 RL이나 Q-learning을 구현하는 것보다, 이 과제에서는 **잘 설계된 feature와 안정적인 rule**이 더 현실적이다.

특히 제출 조건상 학습 결과를 외부 파일로 불러올 수 없고, action마다 시간 제한이 있으므로, 본선까지 고려해도 가장 실용적인 방향은 다음이다.

```text
Feature-based Reflex Agent
+ 상황별 mode 전환
+ maze distance caching
+ 수비 patrol
+ 위험 회피 규칙
```

이 방식이 구현 난이도, 안정성, 시간 제한, baseline 승률, 본선 경쟁력 사이의 균형이 가장 좋다.
