import random
import numpy as np
import matplotlib.pyplot as plt


# Environment size
width = 5
height = 16
epochs = 1000

# Actions
num_actions = 4

actions_list = {"UP": 0,
                "RIGHT": 1,
                "DOWN": 2,
                "LEFT": 3
                }
list_actions = {v: k for k, v in actions_list.iteritems()}

actions_vectors = {"UP": (-1, 0),
                   "RIGHT": (0, 1),
                   "DOWN": (1, 0),
                   "LEFT": (0, -1)
                   }

# Discount factor
discount = 0.8

Q = np.zeros((height * width, num_actions))  # Q matrix
Rewards = np.zeros(height * width)  # Reward matrix, it is stored in one dimension


def getState(y, x):
    return y * width + x


def getStateCoord(state):
    return int(state / width), int(state % width)


def getActions(state):
    y, x = getStateCoord(state)
    actions = []
    if x < width - 1:
        actions.append("RIGHT")
    if x > 0:
        actions.append("LEFT")
    if y < height - 1:
        actions.append("DOWN")
    if y > 0:
        actions.append("UP")
    return actions


def getRndAction(state):
    return random.choice(getActions(state))


def getRndState():
    return random.randint(0, height * width - 1)


Rewards[4 * width + 3] = -10000
Rewards[4 * width + 2] = -10000
Rewards[4 * width + 1] = -10000
Rewards[4 * width + 0] = -10000

Rewards[9 * width + 4] = -10000
Rewards[9 * width + 3] = -10000
Rewards[9 * width + 2] = -10000
Rewards[9 * width + 1] = -10000

Rewards[3 * width + 3] = 100
final_state = getState(3, 3)

print np.reshape(Rewards, (height, width))


def qlearning(s1, a, s2):
    Q[s1][a] = Rewards[s2] + discount * max(Q[s2])
    return

def greedy(state):
    max_action = list_actions[np.argmax(Q[state])]
    if max(Q[state]) != 0:
        return max_action
    else:
        return getRndAction(state)

actions = 0
# Episodes
def getNextAction(state, policy):
    policy = policy.split(" ")
    if policy[0] == "greedy":
        return greedy(state)
    elif policy[0] == "e-greedy":
        prob = random.random()
        if prob > float(policy[1]):
            return getRndAction(state)
        else:
            return greedy(state)
    return getRndAction(state)

results = []
policies = ("random", "greedy", "e-greedy 0.8", "e-greedy 0.5", "e-greedy 0.7") # prob=1 --> greedy

for n in policies:
    Q = np.zeros((height * width, num_actions))
    actions = 0
    for i in xrange(epochs):
        state = getRndState()
        while state != final_state:
            action = getNextAction(state, n)
            y = getStateCoord(state)[0] + actions_vectors[action][0]
            x = getStateCoord(state)[1] + actions_vectors[action][1]
            new_state = getState(y, x)
            qlearning(state, actions_list[action], new_state)
            actions += 1
            state = new_state
    # print Q
    print "Numero promedio de acciones en ", n, " -> ", actions/epochs
    results.append(actions / epochs)

number_of_policies = []
for v in range(len(policies)):
    number_of_policies.append(v)

plt.bar(number_of_policies, results, align="center")
plt.xticks(number_of_policies, policies)
plt.show()

# Q matrix plot

# s = 0
# ax = plt.axes()
# ax.axis([-1, width + 1, -1, height + 1])
#
# for j in xrange(height):
#
#     plt.plot([0, width], [j, j], 'b')
#     for i in xrange(width):
#         plt.plot([i, i], [0, height], 'b')
#
#         direction = np.argmax(Q[s])
#         if s != final_state:
#             if direction == 0:
#                 ax.arrow(i + 0.5, 0.75 + j, 0, -0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
#             if direction == 1:
#                 ax.arrow(0.25 + i, j + 0.5, 0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
#             if direction == 2:
#                 ax.arrow(i + 0.5, 0.25 + j, 0, 0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
#             if direction == 3:
#                 ax.arrow(0.75 + i, j + 0.5, -0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
#         s += 1
#
#     plt.plot([i+1, i+1], [0, height], 'b')
#     plt.plot([0, width], [j+1, j+1], 'b')
#
# plt.show()