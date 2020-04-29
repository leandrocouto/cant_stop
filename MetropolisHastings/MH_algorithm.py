import sys
sys.path.insert(0,'..')
import math
import copy
from game import Game
from play_game_template import play_single_game
from players.vanilla_uct_player import Vanilla_UCT
from players.uct_player import UCTPlayer
from players.random_player import RandomPlayer
from MetropolisHastings.MH_tree import DSL, DSLTree, Node
import time

class MetropolisHastings:
    def __init__(self, beta):
        self.beta = beta
        self.data = []
        '''
        # Toy version
        self.column_range = [2,6]
        self.offset = 2
        self.initial_height = 1
        self.n_players = 2
        self.dice_number = 4
        self.dice_value = 3
        self.max_game_length = 50
        '''
        # Original version
        self.column_range = [2,12]
        self.offset = 2
        self.initial_height = 2 
        self.n_players = 2
        self.dice_number = 4
        self.dice_value = 6 
        self.max_game_length = 500
        
    def run(self):
        script_object = None
        dsl = DSL()
        tree = DSLTree(Node('S', ''), dsl)
        tree.build_tree()
        current_best_program = tree.generate_random_program()
        #print('program = ', current_best_program)
        #print('aqui: ', script_player)

        self.generate_oracle_data(n_games = 3)

        for i in range(1000):
            
            new_tree = copy.deepcopy(tree)
            mutated_program = new_tree.generate_mutated_program(current_best_program)

            script_best_player = tree.generate_player(current_best_program)
            script_mutated_player = new_tree.generate_player(mutated_program)

            score_best, n_errors_best, errors_rate_best, v = \
                        self.calculate_score_function(script_best_player)
            score_mutated, n_errors_mutated, errors_rate_mutated, v2 = \
                        self.calculate_score_function(script_mutated_player)

            accept = min(1, score_mutated/score_best)
            #print('Vezes que o script mutado nao entrou nas regras: ', v2, 'de um total de', len(self.data), 'instancias')
            if accept == 1:
                current_best_program = mutated_program
                tree = new_tree
                #print('error rate at iteration', i, ' - ', errors_rate_mutated)
            
        return current_best_program

    def generate_oracle_data(self, n_games):
        
        for i in range(n_games):
            print('jogo ',  i)
            game = Game(self.n_players, self.dice_number, self.dice_value, 
                        self.column_range, self.offset, self.initial_height
                        )
            player1 = Vanilla_UCT(c = 1, n_simulations = 50)
            player2 = Vanilla_UCT(c = 1, n_simulations = 50)
            self.simplified_play_single_game(player1, player2, game, 
                                                self.max_game_length
                                            )

    def calculate_score_function(self, program):
        n_errors, errors_rate, v = self.calculate_errors(program)
        return math.exp(-self.beta * n_errors), n_errors, errors_rate, v

    def calculate_errors(self, program):
        n_errors = 0
        v = 0
        for i in range(len(self.data)):
            chosen_play, value = program.get_action(self.data[i][0])
            if value == 1:
                v += 1
            if chosen_play != self.data[i][1]:
                n_errors += 1
        return n_errors,  n_errors / len(self.data), v


    def simplified_play_single_game(self, player1, player2, game, max_game_length):
        """ Play a single game between player1 and player2. """

        is_over = False
        rounds = 0
        # actions_taken actions in a row from a player. Used in UCTPlayer players. 
        # List of tuples (action taken, player turn, Game instance).
        # If players change turn, empty the list.
        actions_taken = []
        actions_from_player = 1

        # Game loop
        while not is_over:
            rounds += 1
            moves = game.available_moves()
            if game.is_player_busted(moves):
                actions_taken = []
                actions_from_player = game.player_turn
                continue
            else:
                # UCTPlayer players receives an extra parameter in order to
                # maintain the tree between plays whenever possible
                if game.player_turn == 1 and isinstance(player1, UCTPlayer):
                    if actions_from_player == game.player_turn:
                        chosen_play = player1.get_action(game, [])
                    else:
                        chosen_play = player1.get_action(game, actions_taken)
                elif game.player_turn == 1 and not isinstance(player1, UCTPlayer):
                        chosen_play = player1.get_action(game)
                elif game.player_turn == 2 and isinstance(player2, UCTPlayer):
                    if actions_from_player == game.player_turn:
                        chosen_play = player2.get_action(game, [])
                    else:
                        chosen_play = player2.get_action(game, actions_taken)
                elif game.player_turn == 2 and not isinstance(player2, UCTPlayer):
                        chosen_play = player2.get_action(game)

                self.data.append((game.clone(), chosen_play))
                # Needed because game.play() can automatically change 
                # the player_turn attribute.
                actual_player = game.player_turn
                
                # Clear the plays info so far if player_turn 
                # changed last iteration.
                if actions_from_player != actual_player:
                    actions_taken = []
                    actions_from_player = game.player_turn

                # Apply the chosen_play in the game
                game.play(chosen_play)

                # Save game history
                actions_taken.append((chosen_play, actual_player, game.clone()))

            # if the game has reached its max number of plays, end the game
            # and who_won receives 0, which means no players won.

            if rounds > max_game_length:
                is_over = True
            else:
                _, is_over = game.is_finished()
MH = MetropolisHastings(beta=0.5)
#MH.generate_oracle_data(n_games=5)
#p = RandomPlayer()
#p = Vanilla_UCT(c = 1, n_simulations = 100)
for i in range(10):
    MH.data = []
    program = MH.run()
    print(program)
#score = MH.calculate_score_function(p)
#print('score = ', score)
