# qlearningAgents.py
# ------------------

from turtle import dot
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math,json

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
    """
    def __init__(self, isDisplayed, jsonFile, createTable, qTableFile, ghostAgents=None, **args):
        "Initialize Q-values"
        ReinforcementAgent.__init__(self, **args)

        self.actions = {"North":0, "East":1, "South":2, "West":3, "Stop":4}
        self.isDisplayed = isDisplayed
        self.lista_bucle = []

        self.json_data = []
        with open("./" + jsonFile, "r") as file:
            self.json_data = json.load(file)
        self.numStates = 1
        for i in self.json_data:
            self.numStates *= i["max-value"]
            
        if createTable:
            self.table_file = open(qTableFile + ".txt", "x")
            self.table_file.seek(0)
            self.table_file.truncate()
            for i in range(self.numStates):
                self.table_file.write("0.0 0.0 0.0 0.0 0.0")
                self.table_file.write("\n")
            self.table_file.close()
        self.table_file = open(qTableFile + ".txt", "r+")
#        self.table_file_csv = open("qtable.csv", "r+")        
        self.q_table = self.readQtable()

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()
        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item)+" ")
            self.table_file.write("\n")

#         self.table_file_csv.seek(0)
#         self.table_file_csv.truncate()
#         for line in self.q_table:
#             for item in line[:-1]:
#                 self.table_file_csv.write(str(item)+", ")
#             self.table_file_csv.write(str(line[-1]))                
#             self.table_file_csv.write("\n")

            
    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")    
            
    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()

    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        For instance, the state (3,1) is the row 7
        """

        line = 0
        numAttributes = len(state) - 1
        slicer = self.numStates
        for i in reversed(self.json_data):
            slicer //= i["max-value"]
            if i["type"] == "integer":
                line += slicer * state[numAttributes]
            else:
                line += slicer * i[state[numAttributes]]
            numAttributes -= 1
        # print(line)
        return line

    def getQValue(self, state, action):

        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        position = self.computePosition(state)
        action_column = self.actions[action]

        return self.q_table[position][action_column]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions)==0:
          return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = state[0]
        attributes = state[1]
        if len(legalActions)==0:
          return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(attributes, legalActions[0])
        for action in legalActions:
            value = self.getQValue(attributes, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """

        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        if len(legalActions) == 0:
             return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            return random.choice(legalActions)
        atrib = list(self.getAttributes(state))
        # print(atrib[0], atrib[1], atrib[2], atrib[3], atrib[4], self.lista_bucle)
        return self.getPolicy([legalActions, atrib])

    def getAttributes(self, state):

        minDistance = 99
        minIndex = 0
        pacmanPos = state.getPacmanPosition()
        livingGhosts = state.getLivingGhosts()[1:]
        ghostsPos = state.getGhostPositions()
        ghostsDist = state.data.ghostDistances
        dotDist, dotPos = state.getDistanceNearestFood()
        numDots = state.getNumFood()
        numAGhosts = sum(livingGhosts)
        
        for i in range(len(ghostsDist)):
            if livingGhosts[i]:
                if ghostsDist[i] < minDistance:
                    minDistance = ghostsDist[i]
                    minIndex = i
        
        if dotDist is not None and dotDist < minDistance:
            
            dist_x = dotPos[0] - pacmanPos[0]
            dist_y = dotPos[1] - pacmanPos[1]
            
        else:
            dist_x = ghostsPos[minIndex][0] - pacmanPos[0]
            dist_y = ghostsPos[minIndex][1] - pacmanPos[1]
        
        direction = ""
        if dist_x > 0:
            direction += "East"
        elif dist_x < 0:
            direction += "West"
        
        if dist_y > 0:
            direction += "North"
        elif dist_y < 0:
            direction += "South"
        
        return numDots, numAGhosts, minDistance, direction

    def update(self, state, action, nextState, reward):
        """ Update of Q-table """
        # TRACE for transition and position to update. Comment the following lines if you do not want to see that trace
#         print("Update Q-table with transition: ", state, action, nextState, reward)
#         position = self.computePosition(state)
#         action_column = self.actions[action]
#         print("Corresponding Q-table cell to update:", position, action_column)

        "*** YOUR CODE HERE ***"
        prevLiving = state.getLivingGhosts()[1:]
        prevFoodNum = state.getNumFood()
        prevSum = sum(prevLiving) + prevFoodNum
        livingGhosts = nextState.getLivingGhosts()[1:]
        numFood = state.getNumFood()
        sumGhosts = sum(livingGhosts)
        newSum = sumGhosts + numFood
        cAttributes = self.getAttributes(state)
        if prevSum != newSum:
            self.lista_bucle = []
            
        if len(self.lista_bucle) < 4:
            self.lista_bucle.append(state.getPacmanPosition())
        else:
            self.lista_bucle.pop(0)
            self.lista_bucle.append(state.getPacmanPosition())
        
        nAttributes = self.getAttributes(nextState)
        reward += self.getReward(cAttributes, action, nAttributes)
        pos = self.computePosition(cAttributes)
        column = self.actions[action]
        
        if sumGhosts == 0:
            self.lista_bucle = []
            self.q_table[pos][column] = (1 - self.alpha) * self.getQValue(cAttributes, action) + self.alpha * (reward + 0)
        else:
            best_action = self.getPolicy([self.getLegalActions(state), list(nAttributes)])
            self.q_table[pos][column] = (1 - self.alpha) * self.getQValue(cAttributes, action) + self.alpha * (reward + self.discount * self.getQValue(nAttributes, best_action))

        # TRACE for updated q-table. Comment the following lines if you do not want to see that trace
#         print("Q-table:")
#         self.printQtable()

    def getReward(self, cAttributes, action, nAttributes):

        if action == "Stop":
            return -159
        reward = 0
        if nAttributes[-1] == 1 and cAttributes[-1] == 0:
            reward -= 79
        if nAttributes[-1] == 1 and cAttributes[-1] == 1:
            reward -= 39
        if nAttributes[-1] == 0 and cAttributes[-1] == 1:
            reward += 81

        if cAttributes[-2] > 0:
            if cAttributes[-2] == 1:
                if action == "North":
                    if cAttributes[2] == 1 or nAttributes[2] == 1:
                        if "N" in cAttributes[0]:
                            reward -= 14
                        elif "S" in cAttributes[0]:
                            reward += 16
                elif action == "South":
                    if cAttributes[2] == 1 or nAttributes[2] == 1:
                        if "S" in cAttributes[0]:
                            reward -= 14
                        elif "N" in cAttributes[0]:
                            reward += 16
                elif cAttributes[2] == 0:
                    if action == "East":
                        reward -= 14
                    elif action == "West":
                        reward -= 14
            elif cAttributes[-2] == 2:
                if action == "East":
                    if cAttributes[1] == 1 or nAttributes[1] == 1:
                        if "E" in cAttributes[0]:
                            reward -= 14
                        elif "W" in cAttributes[0]:
                            reward += 16
                elif action == "West":
                    if cAttributes[1] == 1 or nAttributes[1] == 1:
                        if "W" in cAttributes[0]:
                            reward -= 14
                        elif "E" in cAttributes[0]:
                            reward += 16
                elif cAttributes[1] == 0:
                    if action == "North":
                        reward -= 14
                    elif action == "South":
                        reward -= 14
            elif cAttributes[-2] == 3:
                reward += 16
        
        if cAttributes[-3] == 0:
            if action == "North":
                reward -= 4
            elif action == "South":
                reward -= 4
        else:
            if action == "East":
                reward -= 4
            elif action == "West":
                reward -= 4

        # if action == "East":
        #     if cAttributes[1] == 1:
        #         if "E" in nAttributes[0]:
        #             reward -= 15

        # if action == "West":
        #     if nAttributes[1] == 1:
        #         if "W" in nAttributes[0]:
        #             reward -= 15

        # if action == "North":
        #     if nAttributes[2] == 1:
        #         if "N" in nAttributes[0]:
        #             reward -= 15

        # if action == "South":
        #     if nAttributes[2] == 1:
        #         if "S" in nAttributes[0]:
        #             reward -= 15
        

        
        return reward

    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)


class Aprox2QAgent(QLearningAgent):
    
    def __init__(self, **args):
        QLearningAgent.__init__(self, **args)
    
    def getAttributes(self, state):
        
        minDistance = 99
        minIndex = 0
        pacmanPos = state.getPacmanPosition()
        livingGhosts = state.getLivingGhosts()[1:]
        ghostsPos = state.getGhostPositions()
        ghostsDist = state.data.ghostDistances
        dotDist, dotPos = state.getDistanceNearestFood()
        map = state.getWalls()
        numDots = state.getNumFood()
        numAGhosts = sum(livingGhosts)
        
        for i in range(len(ghostsDist)):
            if livingGhosts[i]:
                if ghostsDist[i] < minDistance:
                    minDistance = ghostsDist[i]
                    minIndex = i
        
        if dotDist is not None and dotDist < minDistance:
            
            dist_x = dotPos[0] - pacmanPos[0]
            dist_y = dotPos[1] - pacmanPos[1]
            
            
        else:
            dist_x = ghostsPos[minIndex][0] - pacmanPos[0]
            dist_y = ghostsPos[minIndex][1] - pacmanPos[1]
        
        direction = ""
        if dist_x > 0:
            direction += "East"
        elif dist_x < 0:
            direction += "West"
        
        if dist_y > 0:
            direction += "North"
        elif dist_y < 0:
            direction += "South"

        if dotDist is not None and dotDist < minDistance:
            wall_dir = self.wallInDirection(map, pacmanPos, dotPos, direction)
        else:
            wall_dir = self.wallInDirection(map, pacmanPos, ghostsPos[minIndex], direction)
        
        return numDots, numAGhosts, direction, wall_dir
    
    def wallInDirection(self, map, pacPos, objectPos, direction):
        found_wall = 0

        if "East" in direction or "West" in direction:
            if "East" in direction:
                for i in range(pacPos[0], objectPos[0]):
                    if map[i][pacPos[1]]:
                        found_wall = 1
                        break
            else:
                for i in range(objectPos[0], pacPos[0]):
                    if map[i][pacPos[1]]:
                        found_wall = 1
                        break
        if "North" in direction or "South" in direction:
            if "North" in direction:
                for j in range(pacPos[1], objectPos[1]):
                    if map[pacPos[0]][j]:
                        found_wall = 1
                        break
            else:
                for j in range(objectPos[1], pacPos[1]):
                    if map[pacPos[0]][j]:
                        found_wall = 1
                        break
        
        return found_wall

class Aprox3QAgent(QLearningAgent):
    
    def __init__(self, **args):
        QLearningAgent.__init__(self, **args)
    
    def getAttributes(self, state):
        
        minDistance = 99
        minIndex = 0
        pacmanPos = state.getPacmanPosition()
        livingGhosts = state.getLivingGhosts()[1:]
        ghostsPos = state.getGhostPositions()
        ghostsDist = state.data.ghostDistances
        dotDist, dotPos = state.getDistanceNearestFood()
        map = state.getWalls()
        
        for i in range(len(ghostsDist)):
            if livingGhosts[i]:
                if ghostsDist[i] < minDistance:
                    minDistance = ghostsDist[i]
                    minIndex = i
        
        if dotDist is not None and dotDist < minDistance:
            
            dist_x = dotPos[0] - pacmanPos[0]
            dist_y = dotPos[1] - pacmanPos[1]
            
            
        else:
            dist_x = ghostsPos[minIndex][0] - pacmanPos[0]
            dist_y = ghostsPos[minIndex][1] - pacmanPos[1]
        
        direction = ""
        if dist_x > 0:
            direction += "East"
        elif dist_x < 0:
            direction += "West"
        
        if dist_y > 0:
            direction += "North"
        elif dist_y < 0:
            direction += "South"

        if dotDist is not None and dotDist < minDistance:
            wall_dir = self.wallInDirection(map, pacmanPos, dotPos, direction)
        else:
            wall_dir = self.wallInDirection(map, pacmanPos, ghostsPos[minIndex], direction)
        
        return direction, wall_dir
    
    def wallInDirection(self, map, pacPos, objectPos, direction):
        found_wall = 0

        if "East" in direction or "West" in direction:
            if "East" in direction:
                for i in range(pacPos[0], objectPos[0]):
                    if map[i][pacPos[1]]:
                        found_wall = 1
                        break
            else:
                for i in range(objectPos[0], pacPos[0]):
                    if map[i][pacPos[1]]:
                        found_wall = 1
                        break
        if "North" in direction or "South" in direction:
            if "North" in direction:
                for j in range(pacPos[1], objectPos[1]):
                    if map[pacPos[0]][j]:
                        found_wall = 1
                        break
            else:
                for j in range(objectPos[1], pacPos[1]):
                    if map[pacPos[0]][j]:
                        found_wall = 1
                        break
        
        return found_wall


class Aprox4QAgent(QLearningAgent):
    
    def __init__(self, **args):
        QLearningAgent.__init__(self, **args)
    
    def getAttributes(self, state):

        minDistance = 99
        minIndex = 0
        pacmanPos = state.getPacmanPosition()
        livingGhosts = state.getLivingGhosts()[1:]
        ghostsPos = state.getGhostPositions()
        ghostsDist = state.data.ghostDistances
        dotDist, dotPos = state.getDistanceNearestFood()
        map = state.getWalls()

        for i in range(len(ghostsDist)):
            if livingGhosts[i]:
                if ghostsDist[i] < minDistance:
                    minDistance = ghostsDist[i]
                    minIndex = i
        
        if dotDist is not None and dotDist < minDistance:
            
            dist_x = dotPos[0] - pacmanPos[0]
            dist_y = dotPos[1] - pacmanPos[1]
             
        else:
            dist_x = ghostsPos[minIndex][0] - pacmanPos[0]
            dist_y = ghostsPos[minIndex][1] - pacmanPos[1]
        
        direction = ""
        if dist_x > 0:
            direction += "E"

        elif dist_x < 0:
            direction += "W"
        
        if dist_y > 0:
            direction += "N"
        elif dist_y < 0:
            direction += "S"

        eastWest, northSouth = self.wallOneDirection(map, pacmanPos, direction)
        if len(direction) == 2:
            if eastWest != northSouth:
                if eastWest == 1:
                    northSouth = self.wallInDirection(map, pacmanPos, direction[0], direction[1])
                else:
                    eastWest = self.wallInDirection(map, pacmanPos, direction[1], direction[0])
        
        bucle = 0
        if pacmanPos in self.lista_bucle:
            bucle = 1 
        
        return direction, eastWest, northSouth, bucle

    def wallOneDirection(self, map, pacPos, direction):
        auxEW = 0
        auxNS = 0
        if "E" in direction:
            auxEW = 1 if map[pacPos[0] + 1][pacPos[1]] else 0
        if "W" in direction:
            auxEW = 1 if map[pacPos[0] - 1][pacPos[1]] else 0
        if "N" in direction:
            auxNS = 1 if map[pacPos[0]][pacPos[1] + 1] else 0
        if "S" in direction:
            auxNS = 1 if map[pacPos[0]][pacPos[1] - 1] else 0
        return auxEW, auxNS

    def wallInDirection(self, map, pacPos, check_pared, move_direction):

        pared = 1

        if check_pared == "E" and move_direction == "N":
            i = pacPos[1] + 1
            j = pacPos[0] + 1
            while not map[pacPos[0]][i]:
                if not map[j][i]:
                    pared = 0
                    break
                i += 1
        elif check_pared == "E" and move_direction == "S":
            j = pacPos[0] + 1
            i = pacPos[1] - 1
            while not map[pacPos[0]][i]:
                if not map[j][i]:
                    pared = 0
                    break
                i -= 1
        elif check_pared == "W" and move_direction == "N":
            i = pacPos[1] + 1
            j = pacPos[0] - 1
            while not map[pacPos[0]][i]:
                if not map[j][i]:
                    pared = 0
                    break
                i += 1
        elif check_pared == "W" and move_direction == "S":
            i = pacPos[1] - 1
            j = pacPos[0] - 1
            while not map[pacPos[0]][i]:
                if not map[j][i]:
                    pared = 0
                    break
                i -= 1
        elif check_pared == "S" and move_direction == "E":
            i = pacPos[0] + 1
            j = pacPos[1] - 1
            while not map[i][pacPos[1]]:
                if not map[i][j]:
                    pared = 0
                    break
                i += 1
        
        elif check_pared == "S" and move_direction == "W":
            i = pacPos[0] - 1
            j = pacPos[1] - 1
            while not map[i][pacPos[1]]:
                if not map[i][j]:
                    pared = 0
                    break
                i -= 1
        
        elif check_pared == "N" and move_direction == "E":
            i = pacPos[0] + 1
            j = pacPos[1] + 1
            while not map[i][pacPos[1]]:
                if not map[i][j]:
                    pared = 0
                    break
                i += 1     
        else:
            i = pacPos[0] - 1
            j = pacPos[1] + 1
            while not map[i][pacPos[1]]:
                if not map[i][j]:
                    pared = 0
                    break
                i -= 1
        return pared

class Aprox5QAgent(QLearningAgent):
    def __init__(self, **args):
        QLearningAgent.__init__(self, **args)
    
    def getAttributes(self, state):
        minDistance = 99
        minIndex = 0
        pacmanPos = state.getPacmanPosition()
        livingGhosts = state.getLivingGhosts()[1:]
        ghostsPos = state.getGhostPositions()
        ghostsDist = state.data.ghostDistances
        dotDist, dotPos = state.getDistanceNearestFood()
        map = state.getWalls()

        for i in range(len(ghostsDist)):
            if livingGhosts[i]:
                if ghostsDist[i] < minDistance:
                    minDistance = ghostsDist[i]
                    minIndex = i
        
        if dotDist is not None and dotDist < minDistance:
            
            dist_x = dotPos[0] - pacmanPos[0]
            dist_y = dotPos[1] - pacmanPos[1]
            objectPos = dotPos    
        else:
            dist_x = ghostsPos[minIndex][0] - pacmanPos[0]
            dist_y = ghostsPos[minIndex][1] - pacmanPos[1]
            objectPos = ghostsPos[minIndex]
             
        direction = ""
        if dist_x > 0:
            direction += "E"
        elif dist_x < 0:
            direction += "W"
        
        if dist_y > 0:
            direction += "N"
        elif dist_y < 0:
            direction += "S"


        found_wall_x = False
        found_wall_y = False
        
        if "E" in direction:
            for i in range(pacmanPos[0], objectPos[0] + 1):
                if map[i][pacmanPos[1]]:
                    found_wall_x = True
                    break
        if "W" in direction:
            for i in range(pacmanPos[0], objectPos[0] - 1, -1):
                if map[i][pacmanPos[1]]:
                    found_wall_x = True
                    break
        if "N" in direction:
            for j in range(pacmanPos[1], objectPos[1]  + 1):
                if map[pacmanPos[0]][j]:
                    found_wall_y = True
                    break
        if "S" in direction:
            for j in range(pacmanPos[1], objectPos[1] - 1, -1):
                if map[pacmanPos[0]][j]:
                    found_wall_y = True
                    break
        
        found_wall = 0
        if len(direction) == 1 and (found_wall_y or found_wall_x):
                found_wall = 1
                
        elif len(direction) == 2:
            if not found_wall_x:
                if "N" in direction:
                    for j in range(objectPos[1], pacmanPos[1] - 1, -1):
                        if map[objectPos[0]][j]:
                            found_wall_x = True
                            break
                elif "S" in direction:
                    for j in range(objectPos[1], pacmanPos[1] + 1):
                        if map[objectPos[0]][j]:
                            found_wall_x = True
                            break
            elif not found_wall_y:
                if "E" in direction:
                    for i in range(objectPos[0], pacmanPos[0] - 1, -1):
                        if map[i][objectPos[1]]:
                            found_wall_y = True
                            break
                elif "W" in direction:
                    for i in range(objectPos[0], pacmanPos[0] + 1):
                        if map[i][objectPos[1]]:
                            found_wall_y = True
                            break
            if found_wall_x and found_wall_y:
                found_wall = 1
        
        if found_wall:
            to_check = []
            if map[pacmanPos[0] + 1][pacmanPos[1]]:
                to_check.append("E")
            if map[pacmanPos[0] - 1][pacmanPos[1]]:
                to_check.append("W")
            if map[pacmanPos[0]][pacmanPos[1] + 1]:
                to_check.append("N")
            if map[pacmanPos[0]][pacmanPos[1] - 1]:
                to_check.append("S")
        
        wall = ""
        rand = random.randint(0, len(to_check)-1)

        for i in to_check:
            if i in direction:
                wall = i
                break
        
        if wall == "":
            wall = to_check[rand]

        to_N = 0
        to_S = 0
        to_E = 0
        to_W = 0
        if wall == "N" or wall == "S":
            wall = 0
            if wall == "N":
                to_N = 1
            to_E = 1 if map[pacmanPos[0] + 1][pacmanPos[1]] else 0
            to_W = 1 if map[pacmanPos[0] - 1][pacmanPos[1]] else 0
        elif wall == "E" or wall == "W":
            wall = 1
            to_N = self.wallInDirection(map, pacmanPos, wall, "N")
            to_S = self.wallInDirection(map, pacmanPos, wall, "S")

        bucle = 0
        if pacmanPos in self.lista_bucle:
            bucle = 1  
        
        return direction, to_N, to_S, to_E, to_W, wall, bucle


class Aprox6QAgent(QLearningAgent):
    
    def __init__(self, **args):
        QLearningAgent.__init__(self, **args)
    
    def getAttributes(self, state):

        minDistance = 99
        minIndex = 0
        pacmanPos = state.getPacmanPosition()
        livingGhosts = state.getLivingGhosts()[1:]
        ghostsPos = state.getGhostPositions()
        ghostsDist = state.data.ghostDistances
        dotDist, dotPos = state.getDistanceNearestFood()
        map = state.getWalls()

        for i in range(len(ghostsDist)):
            if livingGhosts[i]:
                if ghostsDist[i] < minDistance:
                    minDistance = ghostsDist[i]
                    minIndex = i
        
        if dotDist is not None and dotDist < minDistance:
            
            dist_x = dotPos[0] - pacmanPos[0]
            dist_y = dotPos[1] - pacmanPos[1]
             
        else:
            dist_x = ghostsPos[minIndex][0] - pacmanPos[0]
            dist_y = ghostsPos[minIndex][1] - pacmanPos[1]
        
        direction = ""
        wall_dir = 0
        if dist_x > 0:
            direction += "E"
            if map[pacmanPos[0] + 1][pacmanPos[1]]:
                wall_dir += 1
        elif dist_x < 0:
            direction += "W"
            if map[pacmanPos[0] - 1][pacmanPos[1]]:
                wall_dir += 1
        if dist_y > 0:
            direction += "N"
            if map[pacmanPos[0]][pacmanPos[1] + 1]:
                wall_dir += 2
        elif dist_y < 0:
            direction += "S"
            if map[pacmanPos[0]][pacmanPos[1] - 1]:
                wall_dir += 2

        eastWest, northSouth = self.wallOneDirection(map, pacmanPos, direction)
        if len(direction) == 2:
            if eastWest != northSouth:
                if eastWest == 1:
                    northSouth = self.wallInDirection(map, pacmanPos, direction[0], direction[1])
                else:
                    eastWest = self.wallInDirection(map, pacmanPos, direction[1], direction[0])

        
        bucle = 0
        if pacmanPos in self.lista_bucle:
            bucle = 1
        
        menor_dist = 0 if abs(dist_y) > abs(dist_x) else 1
        
        return direction, eastWest, northSouth, menor_dist, wall_dir, bucle

    def wallOneDirection(self, map, pacPos, direction):
        auxEW = 0
        auxNS = 0
        if "E" in direction:
            auxEW = 1 if map[pacPos[0] + 1][pacPos[1]] else 0
        if "W" in direction:
            auxEW = 1 if map[pacPos[0] - 1][pacPos[1]] else 0
        if "N" in direction:
            auxNS = 1 if map[pacPos[0]][pacPos[1] + 1] else 0
        if "S" in direction:
            auxNS = 1 if map[pacPos[0]][pacPos[1] - 1] else 0
        return auxEW, auxNS

    def wallInDirection(self, map, pacPos, check_pared, move_direction):

        pared = 1

        if check_pared == "E" and move_direction == "N":
            i = pacPos[1] + 1
            j = pacPos[0] + 1
            while not map[pacPos[0]][i]:
                if not map[j][i]:
                    pared = 0
                    break
                i += 1
        elif check_pared == "E" and move_direction == "S":
            j = pacPos[0] + 1
            i = pacPos[1] - 1
            while not map[pacPos[0]][i]:
                if not map[j][i]:
                    pared = 0
                    break
                i -= 1
        elif check_pared == "W" and move_direction == "N":
            i = pacPos[1] + 1
            j = pacPos[0] - 1
            while not map[pacPos[0]][i]:
                if not map[j][i]:
                    pared = 0
                    break
                i += 1
        elif check_pared == "W" and move_direction == "S":
            i = pacPos[1] - 1
            j = pacPos[0] - 1
            while not map[pacPos[0]][i]:
                if not map[j][i]:
                    pared = 0
                    break
                i -= 1
        elif check_pared == "S" and move_direction == "E":
            i = pacPos[0] + 1
            j = pacPos[1] - 1
            while not map[i][pacPos[1]]:
                if not map[i][j]:
                    pared = 0
                    break
                i += 1
        
        elif check_pared == "S" and move_direction == "W":
            i = pacPos[0] - 1
            j = pacPos[1] - 1
            while not map[i][pacPos[1]]:
                if not map[i][j]:
                    pared = 0
                    break
                i -= 1
        
        elif check_pared == "N" and move_direction == "E":
            i = pacPos[0] + 1
            j = pacPos[1] + 1
            while not map[i][pacPos[1]]:
                if not map[i][j]:
                    pared = 0
                    break
                i += 1     
        else:
            i = pacPos[0] - 1
            j = pacPos[1] + 1
            while not map[i][pacPos[1]]:
                if not map[i][j]:
                    pared = 0
                    break
                i -= 1
        return pared


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, ghostAgents=None, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        feats = self.featExtractor.getFeatures(state, action)
        for f in feats:
          self.weights[f] = self.weights[f] + self.alpha * feats[f]*((reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action))

        # util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
