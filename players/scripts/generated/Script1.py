
from players.player import Player
import random
from players.scripts.DSL import DSL

class Script1(Player):

    def get_action(self, state):
        actions = state.available_moves()
        
        for a in actions:
            if DSL.isDoubles(a):
                return a            
        return random.choice(actions)
        