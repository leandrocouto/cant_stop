import math
import sys
import pickle
sys.path.insert(0,'..')
from IteratedWidth.sketch import Sketch

class SimulatedAnnealing:
    def __init__(self, n_SA_iterations, n_games, init_temp, d):
        self.n_SA_iterations = n_SA_iterations
        self.n_games = n_games
        self.init_temp = init_temp
        self.d = d

    def run(self, tree, tree_to_be_beaten):
        # Original program
        curr_tree = pickle.loads(pickle.dumps(tree, -1))
        curr_program = Sketch(curr_tree.generate_program()).get_object()
        # Program to be beaten is given from selfplay
        best_tree = pickle.loads(pickle.dumps(tree_to_be_beaten, -1))
        best_program = Sketch(best_tree.generate_program()).get_object()

        # Evaluate the starting program against the program from selfplay
        try:
            victories, _, _ = self.evaluate(best_program, curr_program)
        # If the program gives an error during evaluation, this program should
        # be 'discarded/bad' -> set victory to 0.
        except Exception as e:
            #print('erro - ', str(e))
            victories = 0
        

        best_score = victories

        curr_temp = self.init_temp
        for i in range(2, self.n_SA_iterations + 2):
            #print('sa - ', i)
            # Make a copy of curr_tree
            mutated_tree = pickle.loads(pickle.dumps(curr_tree, -1))
            # Mutate it
            mutated_tree.mutate_tree()
            # Get the object of the mutated program
            mutated_program = Sketch(mutated_tree.generate_program()).get_object()
            # Evaluates the mutated program against best_program
            try:
                victories_mut, _, _ = self.evaluate(mutated_program, best_program)
            except Exception as e:
                #print('erro - ', str(e))
                victories_mut = 0

            new_score = victories_mut

            # Update score given the temperature parameters
            updated_score_best, updated_score_mutated = self.update_score(
                                                                best_score, 
                                                                new_score, 
                                                                curr_temp
                                                            )
            
            if updated_score_mutated > updated_score_best:
                print('foi melhor - best, new = ', best_score, new_score)
                exit()
                best_score = new_score
                # Copy the trees
                best_tree = mutated_tree
                # Copy the script
                best_program = mutated_program
            # update temperature according to schedule
            curr_temp = self.temperature_schedule(i)
        return best_tree, best_program

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