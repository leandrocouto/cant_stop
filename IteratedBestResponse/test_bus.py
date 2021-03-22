import random
import numpy as np
import itertools
import sys
from BottomUpSearch import BottomUpSearch
from DSL import Sum, VarList, Function, VarScalar, VarScalarFromArray, Plus, Map, NumberAdvancedThisRound, Times, Constant
sys.path.insert(0,'..')
from game import Board, Game
from players.rule_of_28_player_functional import Rule_of_28_Player_2

game = Game(n_players = 2, dice_number = 4, dice_value = 6, column_range = [2,12],
            offset = 2, initial_height = 3)

for _ in range(80):
    actions = game.available_moves()
    
    if len(actions) == 0:
        break
    
    game.play(random.choice(actions))

progress_value = [0, 0, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6]
neutrals = [col[0] for col in game.neutral_positions]

env = {}
env['state'] = game
env['progress_value'] = progress_value
env['neutrals'] = neutrals

player = Rule_of_28_Player_2()

score = np.sum(
                list(
                    map(
                        lambda col : (player.number_advanced_this_round_2(game, col) + 1) * progress_value[col], neutrals
                        )
                    )
                )

synthesizer = BottomUpSearch()
p, num = synthesizer.synthesize(9, 
                                [Sum, Map, Function, Plus, Times], 
                                [1], 
                                ['neutrals'], 
                                ['progress_value'], 
                                [NumberAdvancedThisRound],
                                [{'state':game, 'progress_value':progress_value, 'neutrals':neutrals, 'out':score}])

print('Score test: ', score)

score_advanced = np.sum(list(map(
                        lambda col : player.number_advanced_this_round_2(game, col), neutrals
                        )
                    )
                )
score_2 = Sum(
            Map(
                Function(
                        Times(
                                Plus(
                                    NumberAdvancedThisRound(), Constant(1)
                                ), 
                                VarScalarFromArray('progress_value')
                            ) 
                        ),
                VarList('neutrals')
                )  
            )
print('Number advanced target: ', score, score_2.interpret(env))
if p is not None:
    print('Program Found: ', p.toString(), score, p.interpret(env))
