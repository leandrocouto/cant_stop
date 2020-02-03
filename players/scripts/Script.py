class Script:
    def __init__(self, script, rules, id):
        self.script = script
        self.rules = rules
        self.id = id
        
    def saveFile(self):
        py = r'''
from players.player import Player
import random
from players.scripts.domain_functions import DSL

class Script{0}(Player):

    def get_action(self, state):
        actions = state.available_moves()
        
        for a in actions:
            if DSL.isDoubles(a):
                return a            
        return random.choice(actions)
        '''
        
        file = open('players/scripts/generated/Script'+ str(self.id) + '.py', 'w')
        file.write(py.format(str(self.id)))
        file.close()
        
        
    def print(self):
        print(self.script)
        print(self.rules)