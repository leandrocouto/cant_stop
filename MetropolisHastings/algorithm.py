from abc import ABC, abstractmethod
import math
import sys
import re
import os
sys.path.insert(0,'..')
from game import Game
from solitaire_game import SolitaireGame
#from sketch import Sketch
from experimental_sketch import Sketch
from MetropolisHastings.DSL import DSL
#from MetropolisHastings.experimental_DSL import ExperimentalDSL
from MetropolisHastings.two_weights_DSL import TwoWeightsDSL
from MetropolisHastings.shared_weights_DSL import SharedWeightsDSL
from players.glenn_player import Glenn_Player
from players.vanilla_uct_player import Vanilla_UCT
from play_game_template import simplified_play_single_game 
from play_game_template import play_single_game
from play_game_template import play_solitaire_single_game
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

class Algorithm(ABC):

    def __init__(self, tree_max_nodes, n_iterations, n_games_glenn, n_games_uct,
        n_games_solitaire, uct_playouts, max_game_rounds, yes_no_dsl, column_dsl,
        n_cores):

        self.tree_max_nodes = tree_max_nodes
        self.n_iterations = n_iterations
        self.n_games_glenn = n_games_glenn
        self.n_games_uct = n_games_uct
        self.n_games_solitaire = n_games_solitaire
        self.uct_playouts = uct_playouts
        self.max_game_rounds = max_game_rounds
        self.yes_no_dsl = yes_no_dsl
        self.column_dsl = column_dsl
        self.n_cores = n_cores

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

    def helper_glenn(self, args):
        return simplified_play_single_game(args[0], args[1], args[2], args[3])

    def validate_against_glenn(self, script):

        glenn = Glenn_Player()

        victories = 0
        losses = 0
        draws = 0

        # First with the current script as first player, then the opposite

        # ProcessPoolExecutor() will take care of joining() and closing()
        # the processes after they are finished.
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            # Specify which arguments will be used for each parallel call
            args_1 = (
                        (
                        script, glenn, Game(2, 4, 6, [2,12], 2, 2), 
                        self.max_game_rounds
                        ) 
                    for _ in range(self.n_games_glenn // 2)
                    )
            results_1 = executor.map(self.helper_glenn, args_1)

        # Current script is now the second player
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            # Specify which arguments will be used for each parallel call
            args_2 = (
                        (
                        glenn, script, Game(2, 4, 6, [2,12], 2, 2), 
                        self.max_game_rounds
                        ) 
                    for _ in range(self.n_games_glenn // 2)
                    )
            results_2 = executor.map(self.helper_glenn, args_2)

        for result in results_1:
            if result == 1:
                victories += 1
            elif result == 2:
                losses += 1
            else:
                draws += 1 

        for result in results_2:
            if result == 1:
                losses += 1
            elif result == 2:
                victories += 1
            else:
                draws += 1 
        return victories, losses, draws

    def helper_uct(self, args):
        return play_single_game(args[0], args[1], args[2], args[3])
    def validate_against_UCT(self, script):

        victories = []
        losses = []
        draws = []

        for i in range(len(self.uct_playouts)):
            v = 0
            l = 0
            d = 0

            uct = Vanilla_UCT(c = 1, n_simulations = self.uct_playouts[i])
            # ProcessPoolExecutor() will take care of joining() and closing()
            # the processes after they are finished.
            with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
                # Specify which arguments will be used for each parallel call
                args_1 = (
                            (
                            script, uct, Game(2, 4, 6, [2,12], 2, 2), 
                            self.max_game_rounds
                            ) 
                        for _ in range(self.n_games_uct // 2)
                        )
                results_1 = executor.map(self.helper_uct, args_1)

            uct = Vanilla_UCT(c = 1, n_simulations = self.uct_playouts[i])
            # Current script is now the second player
            with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
                # Specify which arguments will be used for each parallel call
                args_2 = (
                            (
                            uct, script, Game(2, 4, 6, [2,12], 2, 2), 
                            self.max_game_rounds
                            )  
                        for _ in range(self.n_games_uct // 2)
                        )
                results_2 = executor.map(self.helper_uct, args_2)

            for result in results_1:
                if result == 1:
                    v += 1
                elif result == 2:
                    l += 1
                else:
                    d += 1 

            for result in results_2:
                if result == 1:
                    l += 1
                elif result == 2:
                    v += 1
                else:
                    d += 1

            victories.append(v)
            losses.append(l)
            draws.append(d)

        return victories, losses, draws

    def validate_solitaire(self, current_script):
        """ Validate current script in a Solitaire version of Can't Stop. """

        rounds = []

        for i in range(self.n_games_solitaire):
            game = game = SolitaireGame(4, 6, [2,12], 2, 2)
            rounds_played = play_solitaire_single_game(
                                                        current_script, 
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