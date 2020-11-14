from abc import ABC, abstractmethod
import math
import sys
import re
import os
sys.path.insert(0,'..')
from game import Game
from solitaire_game import SolitaireGame
from sketch import Sketch
#from experimental_sketch import Sketch
from MetropolisHastings.DSL import DSL
#from MetropolisHastings.experimental_DSL import ExperimentalDSL
from MetropolisHastings.two_weights_DSL import TwoWeightsDSL
from MetropolisHastings.shared_weights_DSL import SharedWeightsDSL
from players.glenn_player import Glenn_Player
from players.vanilla_uct_player import Vanilla_UCT
from play_game_template import simplified_play_single_game 
from play_game_template import play_single_game
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

class Algorithm(ABC):

    def __init__(self, tree_max_nodes, n_iterations, n_games_glenn, n_games_uct,
        n_games_solitaire, uct_playouts, max_game_rounds, yes_no_dsl, column_dsl):

        self.tree_max_nodes = tree_max_nodes
        self.n_iterations = n_iterations
        self.n_games_glenn = n_games_glenn
        self.n_games_uct = n_games_uct
        self.n_games_solitaire = n_games_solitaire
        self.uct_playouts = uct_playouts
        self.max_game_rounds = max_game_rounds
        self.yes_no_dsl = yes_no_dsl
        self.column_dsl = column_dsl

        # For analysis
        self.victories = []
        self.losses = []
        self.draws = []
        self.games_played_successful = []
        self.games_played_all = []
        self.games_played_uct = []
        self.games_played = 0

        # For analysis - Games against Glenn
        self.victories_against_glenn = []
        self.losses_against_glenn = []
        self.draws_against_glenn = []

        # For analysis - Games against UCT
        self.victories_against_UCT = []
        self.losses_against_UCT = []
        self.draws_against_UCT = []

        # For analysis - Solitaire games
        self.avg_rounds_solitaire = []
        self.std_rounds_solitaire = []

    @abstractmethod
    def accept_new_program(self):
        """ 
        Check if the new synthesized program is better than the current best.
        Possible regularization techniques must be implemented here.
        Return True if the new program is accepted, False otherwise.
        Concrete classes must implement this method.
        """
        pass

    def _string_to_object(self, str_class, *args, **kwargs):
        """ Transform a program written inside str_class to an object. """

        exec(str_class)
        class_name = re.search("class (.*):", str_class).group(1).partition("(")[0]
        return locals()[class_name](*args, **kwargs)

    def generate_player(self, program_string, program_column, iteration):
        """ Generate an object of the class inside the string self._py """

        script = Sketch(
                        program_string, 
                        program_column, 
                        self.n_iterations, 
                        self.tree_max_nodes
                    )
        return self._string_to_object(script.generate_text(iteration))

    def validate_against_glenn(self, script):
        """ Validate current script against Glenn's heuristic player. """

        glenn = Glenn_Player()

        victories = 0
        losses = 0
        draws = 0

        for i in range(self.n_games_glenn):
            game = game = Game(2, 4, 6, [2,12], 2, 2)
            if i%2 == 0:
                    who_won = simplified_play_single_game(
                                                        script, 
                                                        glenn, 
                                                        game, 
                                                        self.max_game_rounds
                                                    )
                    if who_won == 1:
                        victories += 1
                    elif who_won == 2:
                        losses += 1
                    else:
                        draws += 1
            else:
                who_won = simplified_play_single_game(
                                                    glenn, 
                                                    script, 
                                                    game, 
                                                    self.max_game_rounds
                                                )
                if who_won == 2:
                    victories += 1
                elif who_won == 1:
                    losses += 1
                else:
                    draws += 1

        return victories, losses, draws

    def validate_against_UCT(self, script):
        """ Validate current script against UCT. """

        victories = []
        losses = []
        draws = []

        for i in range(len(self.uct_playouts)):
            v = 0
            l = 0
            d = 0
            for j in range(self.n_games_uct):
                game = game = Game(2, 4, 6, [2,12], 2, 2)
                uct = Vanilla_UCT(c = 1, n_simulations = self.uct_playouts[i])
                if j%2 == 0:
                        who_won = play_single_game(
                                                    script, 
                                                    uct, 
                                                    game, 
                                                    self.max_game_rounds
                                                    )
                        if who_won == 1:
                            v += 1
                        elif who_won == 2:
                            l += 1
                        else:
                            d += 1
                else:
                    who_won = play_single_game(
                                                uct, 
                                                script, 
                                                game, 
                                                self.max_game_rounds
                                                )
                    if who_won == 2:
                        v += 1
                    elif who_won == 1:
                        l += 1
                    else:
                        d += 1
            
            victories.append(v)
            losses.append(l)
            draws.append(d)

        return victories, losses, draws

    def validate_solitaire(self, script):
        """ Validate current script in a Solitaire version of Can't Stop. """

        rounds = []

        for i in range(self.n_games_solitaire):
            game = game = SolitaireGame(4, 6, [2,12], 2, 2)
            rounds_played = play_solitaire_single_game(
                                                        script, 
                                                        game, 
                                                        self.max_game_rounds
                                                        )
            rounds.append(rounds_played)

        avg_round = sum(rounds) / len(rounds)
        
        std_sum = 0
        for i in range(len(rounds)):
            std_sum += (rounds[i] - avg_round) * (rounds[i] - avg_round)

        std_round = math.sqrt(std_sum / len(rounds)) 

        return avg_round, std_round