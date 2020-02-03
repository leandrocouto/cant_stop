import random
from players.scripts.Script import Script


class CFG:
    def __init__(self):
        
        self.start = 'S'
        
        self.grammar = {}
        self.grammar[self.start] = ['if B : \n \t return a \n S', '']
        self.grammar['B'] = ['B and B', 'B or B', 'F > NUMBER', 'F < NUMBER', 'F == NUMBER', 'B1']
        self.grammar['B1'] = ['isDoubles(a)', 'containsNumber(a, NUMBER )', 'actionWinsColumn(state,a)']
        self.grammar['F'] = ['numberColumnsWon(state,a)']
        self.grammar['NUMBER'] = ['0', '1', '2', '3', '4', '5', '6']
        
        self.setRules = []

    def generateRandom(self):
        script_text = self._dfsGenerateRandom(self.start, 0)
        script = Script(script_text, self.setRules, 1)
        self.setRules = []
        script.print()
        print('Saving file...')
        script.saveFile()
        
        
        return script
            
    def _dfsGenerateRandom(self, symbol, depth):
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
            terminal = self._dfsGenerateRandom(s, depth + 1)            
            script += terminal
            
            if symbol == self.start and s != self.start:
                rules.append(terminal)
        
        if len(rules) > 0:
            self.setRules.append(rules)
        return script
        
        
        