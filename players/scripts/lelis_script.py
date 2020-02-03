from players.player import Player
import random
from players.scripts.domain_functions import DSL

class LelisPlayer(Player):

    def get_action(self, state):
        actions = state.available_moves()
        
        for a in actions:
            if isinstance(a, str) and a == 'n' and DSL.numberColumnsWon(state, a) > 0:
                return a
            
            if DSL.actionWinsColumn(state, a):
                return a
            
            if DSL.containsNumber(a, 2) or DSL.containsNumber(a, 6):
                return a
            
            if DSL.isDoubles(a):
                return a            
        return random.choice(actions)