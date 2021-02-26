import math
import sys
import pickle
import time
import os
sys.path.insert(0,'..')
#from IteratedWidth.sketch import Sketch
from IteratedWidth.new_sketch import NewSketch
from game import Game
from play_game_template import simplified_play_single_game

class SimulatedAnnealing:
    def __init__(self, n_SA_iterations, max_game_rounds, n_games, init_temp, d,
                to_log):
        self.n_SA_iterations = n_SA_iterations
        self.max_game_rounds = max_game_rounds
        self.n_games = n_games
        self.init_temp = init_temp
        self.d = d
        # Bool representing if successful iterations are to be saved to file
        self.to_log = to_log
        # Logging info
        self.folder = None
        self.filename = None
        self.suffix = None

    def run(self, initial_state, player_to_be_beaten):
        """ 
        Main routine of Simulated Annealing. SA will start mutating from
        initial_state and the first player to be beaten is player_to_be_beaten.
        If this player is beaten, the player that beat it will be the one to be
        beaten in the next iteration. 
        """

        # Original program
        curr_tree = pickle.loads(pickle.dumps(initial_state, -1))
        curr_player = NewSketch(curr_tree.generate_program()).get_object()
        # Program to be beaten is given from selfplay
        best_tree = None
        best_player = player_to_be_beaten

        # Evaluate the starting program against the program from selfplay
        try:
            victories, _, _ = self.evaluate(curr_player, best_player)
        # If the program gives an error during evaluation, this program should
        # be 'discarded/bad' -> set victory to 0.
        except Exception as e:
            victories = 0
        

        best_score = victories

        # Initial program
        with open(self.filename, 'a') as f:
            print('Initial SA program below', file=f)
            print(initial_state.generate_program(), file=f)
            print(file=f)
            
        # Number of "successful" iterations
        successful = 0 
        curr_temp = self.init_temp
        for i in range(2, self.n_SA_iterations + 2):
            with open(self.filename, 'a') as f:
                print('Beginning iteration ',i, file=f)
            start = time.time()
            #print('sa ite', i)
            # Make a copy of curr_tree
            mutated_tree = pickle.loads(pickle.dumps(curr_tree, -1))
            # Mutate it
            mutated_tree.mutate_tree()
            # Get the object of the mutated program
            mutated_player = NewSketch(mutated_tree.generate_program()).get_object()
            # Evaluates the mutated program against best_program
            try:
                victories_mut, _, _ = self.evaluate(mutated_player, curr_player)
            except Exception as e:
                victories_mut = 0

            new_score = victories_mut

            # Update score given the temperature parameters
            updated_score_best, updated_score_mutated = self.update_score(
                                                                best_score, 
                                                                new_score, 
                                                                curr_temp
                                                            )
            
            if updated_score_mutated > updated_score_best:
                successful += 1
                if self.to_log:
                    # Save to file
                    inner_folder = self.folder + '/Successful_SA/'
                    if not os.path.exists(inner_folder):
                        os.makedirs(inner_folder)
                    NewSketch(mutated_tree.generate_program()).save_file_custom(inner_folder, self.suffix + '_SA_ite' + str(i))

                with open(self.filename, 'a') as f:
                    elapsed = time.time() - start
                    print('SA - Iteration = ',i, '- Mutated player was better - Victories =', victories_mut, file=f)
                    print('Mutated program below - Time of this iteration =', elapsed, 'Successful ite. so far = ', successful, file=f)
                    print(mutated_tree.generate_program(), file=f)
                    print(file=f)
                best_score = new_score
                # Copy the trees
                best_tree = mutated_tree
                # Copy the script
                best_player = mutated_player
            # update temperature according to schedule
            curr_temp = self.temperature_schedule(i)

        # It can happen that the mutated state is never better than 
        # player_to_be_beaten, therefore we set best_tree and best_player to the 
        # original initial_state.
        if best_tree is None:
            best_tree = curr_tree
            best_player = curr_player
        with open(self.filename, 'a') as f:
            print('Successful iterations of this SA = ', successful, 'out of', self.n_SA_iterations, 'iterations.',  file=f)
        return best_tree, best_player

    def temperature_schedule(self, iteration):
        """ Calculate the next temperature used for the score calculation. """

        return self.d/math.log(iteration)

    def update_score(self, score_best, score_mutated, curr_temp):
        """ Update the score according to the current temperature. """
        
        new_score_best = score_best**(1 / curr_temp)
        new_score_mutated = score_mutated**(1 / curr_temp)
        return new_score_best, new_score_mutated

    def evaluate(self, player_1, player_2):
        """ 
        Play self.n_games Can't Stop games between player_1 and player_2. The
        players swap who is the first player per iteration because Can't Stop
        is biased towards the player who plays the first move.
        """

        victories = 0
        losses = 0
        draws = 0

        for i in range(self.n_games):
            game = Game(2, 4, 6, [2,12], 2, 2)
            if i%2 == 0:
                    who_won = simplified_play_single_game(
                                                        player_1, 
                                                        player_2, 
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
                                                    player_2, 
                                                    player_1, 
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