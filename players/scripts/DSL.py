import copy
import random
from players.scripts.Script import Script

class DSL:
    
    def __init__(self):
        
        self.start = 'S'
        
        self.grammar = {}
        self.grammar[self.start] = ['if B : \n \t return a \n S', '']
        self.grammar['B'] = ['B and B', 'B or B', 'F > NUMBER', 'F < NUMBER', 'F == NUMBER', 'B1']
        self.grammar['B1'] = ['isDoubles(a)', 'containsNumber(a, NUMBER )', 'actionWinsColumn(state,a)', 'DSL.isStopAction(a)']
        self.grammar['F'] = ['numberColumnsWon(state,a)']
        self.grammar['NUMBER'] = ['0', '1', '2', '3', '4', '5', '6']
        
        self.setRules = [] 

    def generateRandomScript(self, id):
        script_text = self._dfsGenerateRandomScript(self.start, 0)
        script = Script(script_text, self.setRules, id)
        self.setRules = []
        script.print()
        print('Saving file...')
        script.saveFile()
        
        
        return script
            
    def _dfsGenerateRandomScript(self, symbol, depth):
        if symbol not in self.grammar:
            return symbol + ' '
        
        rules = []
        #forcing script to have at least one rule
        if depth == 0 and symbol == self.start:
            index = 0
        else:
            index = random.randint(0, len(self.grammar[symbol]) - 1)
        random_rule = self.grammar[symbol][index]
        symbols = random_rule.split()
        script = ''
        for s in symbols:
            terminal = self._dfsGenerateRandomScript(s, depth + 1)            
            script += terminal
            
            if symbol == self.start and s != self.start:
                rules.append(terminal)
        
        if len(rules) > 0:
            self.setRules.append(rules)
        return script
    
    @staticmethod
    def isDoubles(action):
        if len(action) > 1 and action[0] == action[1]:
            return True
        else:
            return False
        
    @staticmethod    
    def containsNumber(action, number):
        if not isinstance(action, str):
            if number in action:
                return True
        return False
    
    @staticmethod
    def actionWinsColumn(state, action):
        copy_state = copy.deepcopy(state)
        copy_state.play(action)
        columns_won = copy_state.columns_won_current_round()
        columns_won_previously = state.columns_won_current_round()
        if len(columns_won) > 0 and columns_won != columns_won_previously:
            return True
        return False
    
    @staticmethod
    def numberColumnsWon(state, action):
        return len(state.columns_won_current_round())
    
    @staticmethod
    def isStopAction(action):
        if isinstance(action, str) and action == 'n':
            return True
        return False