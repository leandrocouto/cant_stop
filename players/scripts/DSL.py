import copy
import random
from players.scripts.Script import Script

class DSL:
    
    def __init__(self):
        
        self.start = 'S'
        self.start_single = 'SR'
        self.boolean = 'B'
        
        self._grammar = {}
        self._grammar[self.start] = ['if B S', '']
        self._grammar[self.start_single] = ['if B', '']
        #self._grammar['B'] = ['B and B', 'B or B', 'F > NUMBER', 'F < NUMBER', 'F == NUMBER', 'B1']
#         self._grammar['B'] = ['B1', 'B1 and B1', 'B1 or B1']
        self._grammar['B'] = ['B1', 'B1 and B1']
        self._grammar['B1'] = ['DSL.isDoubles(a)', 'DSL.containsNumber(a, NUMBER )', 'DSL.actionWinsColumn(state,a)', 'DSL.hasWonColumn(state,a)', 
                               'DSL.numberPositionsProgressedThisRoundColumn(state, NUMBER ) > SMALL_NUMBER and DSL.isStopAction(a)', 'DSL.isStopAction(a)',
                               'DSL.numberPositionsConquered(state, NUMBER ) > SMALL_NUMBER and DSL.containsNumber(a, NUMBER )']
        #self._grammar['F'] = ['DSL.numberColumnsWon(state,a)']
        #self._grammar['NUMBER'] = ['0', '1', '2', '3', '4', '5', '6']
        self._grammar['NUMBER'] = ['2', '3', '4', '5', '6']
#         self._grammar['NUMBER'] = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        self._grammar['SMALL_NUMBER'] = ['0', '1', '2']
        
        self._reservedWords = ['if']
        
        self.setRules = [] 

    def generateRandomScript(self, id = 0, start='S'):
        _ = self._dfsGenerateRandomScript(start, 0)
        script = Script(self.setRules, id)
        self.setRules = []
        return script
            
    def _dfsGenerateRandomScript(self, symbol, depth):
        if symbol not in self._grammar:
            return symbol + ' '
        
        rules = []
        #forcing script to have at least one rule
        if depth == 0 and symbol == self.start:
            index = 0
        else:
            index = random.randint(0, len(self._grammar[symbol]) - 1)
        random_rule = self._grammar[symbol][index]
        
        symbols = random_rule.split()
        script = ''
        for s in symbols:
            terminal = self._dfsGenerateRandomScript(s, depth + 1)            
            script += terminal
            
            if symbol == self.start and s != self.start and s not in self._reservedWords:
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
    def numberPositionsProgressedThisRoundColumn(state, column):
        return state.number_positions_conquered_this_round(column)
    
    @staticmethod
    def numberPositionsConquered(state, column):
        return state.number_positions_conquered(column)

    @staticmethod
    def hasWonColumn(state, action):
        return len(state.columns_won_current_round()) > 0
    
    @staticmethod
    def isStopAction(action):
        if isinstance(action, str) and action == 'n':
            return True
        return False