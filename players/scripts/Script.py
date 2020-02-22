import random

class Script:
    def __init__(self, rules, id=0):
        self._rules = rules
        self._id = id
        self._fitness = 0
        self._matches_played = 0
        
        self._py = r'''
from players.player import Player
import random
from players.scripts.DSL import DSL

class Script{0}(Player):

    def __init__(self):
        self._counter_calls = []
        for i in range({1}):
            self._counter_calls.append(0)
            
    def get_counter_calls(self):
        return self._counter_calls 

    def get_action(self, state):
        actions = state.available_moves()
        
        for a in actions:
        '''
        
        self._if_string = r'''
            if {0}:
                self._counter_calls[{1}] += 1
                return a
                    '''
        self._end_script = r'''
        return actions[0]
                    '''
        
    def __eq__(self, other):
        return self._id == other._id
    
    def getId(self):
        return self._id
    
    def setId(self, id):
        self._id = id
        
    def getRules(self):
        return self._rules
        
    def clearAttributes(self):
        self._fitness = 0
        self._matches_played = 0
        
    def remove_unused_rules(self, counter_calls):
        new_rules = []
        for i in range(len(self._rules)):
            if counter_calls[i] > 0:
                new_rules.append(self._rules[i])
        if len(new_rules) > 0:
            self._rules = new_rules
        
    def mutate(self, rate, dsl):
        mutated_rules = []
        has_mutated = False
        for i in range(len(self._rules)):
            #checking if mutation will happen
            if random.randint(0, 100) < rate * 100:
                has_mutated = True
                rule = dsl.generateRandomScript('SR').getRules()
                #verify if mutation replaces old rule
                if random.randint(0, 100) < rate * 100:
                    mutated_rules.append(rule[0])
                else:
                    mutated_rules.append(self._rules[i])
                    mutated_rules.append(rule[0])
            else:
                mutated_rules.append(self._rules[i])
                
        if has_mutated:
            self._rules = mutated_rules
            
            
    def forcedMutation(self, rate, dsl):
        mutated_rules = []
        for i in range(len(self._rules)):
            rule = dsl.generateRandomScript('SR').getRules()
            #verify if mutation replaces old rule
            if random.randint(0, 100) < rate * 100:
                mutated_rules.append(rule[0])
            else:
                mutated_rules.append(self._rules[i])
                mutated_rules.append(rule[0])
                
        self._rules = mutated_rules
    
    def generateSplit(self):
        split_index = random.randint(0, len(self._rules))
        split1 = self._rules[0:split_index + 1]
        split2 = self._rules[split_index + 1: len(self._rules) + 1]
                
        return split1, split2
    
    def addFitness(self, v):
        self._fitness += v
        
    def getFitness(self):
        return self._fitness
        
    def incrementMatchesPlayed(self):
        self._matches_played += 1
        
    def getMatchesPlayed(self):
        return self._matches_played
    
    def _generateTextScript(self):
        
        number_rules = len(self._rules)
        py = self._py.format(str(self._id), str(number_rules))
        
        for i in range(number_rules):
            py += self._if_string.format(self._rules[i][0], i)
        py += self._end_script
        
        return py
                        
    def saveFile(self, path):
        py = self._generateTextScript()
        
        file = open(path + 'Script'+ str(self._id) + '.py', 'w')
        file.write(py)
        file.close()
        
    def print(self):
        py = self._generateTextScript()
        print(py)