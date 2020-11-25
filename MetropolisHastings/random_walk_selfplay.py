import sys
import pickle
import time
import os
import matplotlib.pyplot as plt
sys.path.insert(0,'..')
from MetropolisHastings.parse_tree import ParseTree
from game import Game
from MetropolisHastings.experimental_DSL import ExperimentalDSL
from experimental_sketch import Sketch
#from MetropolisHastings.DSL import DSL
#from sketch import Sketch
from algorithm import Algorithm
from play_game_template import simplified_play_single_game
from play_game_template import play_single_game

class RandomWalkSelfplay(Algorithm):
    """
    Simulated Annealing but instead of keeping a score on how many actions this
    algorithm got it correctly (when compared to an oracle), the score is now
    computed on how many victories the mutated get against the current program.
    The mutated program is accepted if it gets more victories than the current
    program playing against itself.
    """
    def __init__(self, algo_id, n_iterations, tree_max_nodes, n_games, 
        n_games_glenn, n_games_uct, n_games_solitaire, uct_playouts, eval_step, 
        max_game_rounds, iteration_run, yes_no_dsl, column_dsl, validate,
        scrips_to_collect):
        """
        Metropolis Hastings with temperature schedule. This allows the 
        algorithm to explore more the space search.
        - n_games is the number of games played in selfplay.
        - n_games_glenn is the number of games played against Glenn's heuristic.
        - max_game_rounds is the number of rounds necessary in a game to
        consider it a draw. This is necessary because Can't Stop games can
        theoretically last forever.
        """

        self.algo_id = algo_id
        self.n_games = n_games
        self.eval_step = eval_step
        self.iteration_run = iteration_run

        super().__init__(tree_max_nodes, n_iterations, n_games_glenn, 
                            n_games_uct, n_games_solitaire, uct_playouts,
                            max_game_rounds, yes_no_dsl, column_dsl, validate,
                            scrips_to_collect
                        )

        self.filename = str(self.algo_id) + '_' + \
                        str(self.n_iterations) + 'ite_' + \
                        str(self.n_games) + 'selfplay_' + \
                        str(self.n_games_glenn) + 'glenn_' + \
                        str(self.n_games_uct) + 'uct_' + \
                        str(self.iteration_run) + 'run'

        if not os.path.exists(self.filename):
            os.makedirs(self.filename)

        self.tree_string = ParseTree(self.yes_no_dsl, self.tree_max_nodes)
        self.tree_column = ParseTree(self.column_dsl, self.tree_max_nodes)

    def run(self):
        """ Main routine of the SA algorithm. """

        full_run = time.time()

        self.tree_string.build_glenn_tree(True)
        self.tree_column.build_glenn_tree(False)
        #self.tree_string.build_tree(self.tree_string.root)
        #self.tree_column.build_tree(self.tree_column.root)

        #print('tree')
        #self.tree_string.print_tree(self.tree_string.root, '  ')
        #current_program_string = self.tree_string.generate_program()
        #current_program_column = self.tree_column.generate_program()
        #print('current_program_string')
        #print(current_program_string)
        #print('current_program_column')
        #print(current_program_column)
        #exit()
        # Main loop
        for i in range(self.n_iterations):
            start = time.time()
            # Make a copy of the tree for future mutation
            new_tree_string = pickle.loads(pickle.dumps(self.tree_string, -1))
            new_tree_column = pickle.loads(pickle.dumps(self.tree_column, -1))

            new_tree_string.mutate_tree()
            new_tree_column.mutate_tree()

            current_program_string = self.tree_string.generate_program()
            current_program_column = self.tree_column.generate_program()

            #print('current_program_string')
            #print(current_program_string)
            #print('current_program_column')
            #print(current_program_column)

            #print('tree current')
            #self.tree_string.print_tree(self.tree_string.root, '  ')
            #exit()
            '''
            folder = self.filename + '/data/' 
            if not os.path.exists(folder):
                os.makedirs(folder)
            dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.filename + '/data/' 
            script = Sketch(
                            current_program_string, 
                            current_program_column, 
                            self.n_iterations, 
                            self.tree_max_nodes
                        )      
            script.save_file_custom(dir_path, self.filename + '_myglennnnnn')
            
            exit()
            '''

            script_best_player = self.generate_player(
                                                        current_program_string,
                                                        current_program_column,
                                                        i
                                                        )

            mutated_program_string = new_tree_string.generate_program()
            mutated_program_column = new_tree_column.generate_program()

            #print('mutated_program_string')
            #print(mutated_program_string)
            #print('mutated_program_column')
            #print(mutated_program_column)

            #print('tree mutated')
            #new_tree_string.print_tree(new_tree_string.root, '  ')
            #exit()

            script_mutated_player = self.generate_player(
                                                        mutated_program_string,
                                                        mutated_program_column,
                                                        i
                                                        )
            victories, losses, draws = self.selfplay(
                                                    script_mutated_player, 
                                                    script_best_player
                                                    )

            self.games_played += self.n_games
            self.games_played_all.append(self.games_played)

            # If the new synthesized program is better
            if self.accept_new_program(victories, losses):
                self.victories.append(victories)
                self.losses.append(losses)
                self.draws.append(draws)

                self.games_played_successful.append(self.games_played)

                # Save checkpoint based on number of games played
                if self.scripts_to_collect and self.games_played >= self.scripts_to_collect[0]:
                    self.save_checkpoint_script(mutated_program_string, mutated_program_column)


                # Assign the mutated tree as the best one
                self.tree_string = new_tree_string
                self.tree_column = new_tree_column

                # Validade against Glenn's heuristic
                v_glenn = None
                l_glenn = None
                d_glenn = None
                start_glenn = time.time()
                if self.validate:
                    v_glenn, l_glenn, d_glenn = self.validate_against_glenn(script_mutated_player)
                    self.victories_against_glenn.append(v_glenn)
                    self.losses_against_glenn.append(l_glenn)
                    self.draws_against_glenn.append(d_glenn)
                elapsed_time_glenn = time.time() - start_glenn

                # Validade against UCT
                v_uct = None 
                l_uct = None 
                d_uct = None
                start_uct = time.time()
                if self.validate:
                    # Only play games against UCT every eval_step successful iterations
                    if len(self.victories_against_glenn) % self.eval_step == 0:
                        self.games_played_uct.append(self.games_played)
                        v_uct, l_uct, d_uct = self.validate_against_UCT(script_mutated_player)
                        self.victories_against_UCT.append(v_uct)
                        self.losses_against_UCT.append(l_uct)
                        self.draws_against_UCT.append(d_uct)
                elapsed_time_uct = time.time() - start_uct

                elapsed_time = time.time() - start

                # Save data file
                iteration_data = (
                                    self.victories,
                                    self.losses,
                                    self.draws,
                                    self.victories_against_glenn,
                                    self.losses_against_glenn,
                                    self.draws_against_glenn,
                                    self.victories_against_UCT,
                                    self.losses_against_UCT,
                                    self.draws_against_UCT,
                                    self.avg_rounds_solitaire,
                                    self.std_rounds_solitaire,
                                    self.games_played,
                                    self.games_played_successful,
                                    self.games_played_all,
                                    self.games_played_uct,
                                    self.tree_string, self.tree_column
                                )
                folder = self.filename + '/data/' 
                if not os.path.exists(folder):
                    os.makedirs(folder)
                with open(folder + 'datafile_iteration_' + str(i) , 'wb') as file:
                    pickle.dump(iteration_data, file)
                # Save current script
                dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.filename + '/data/' 
                script = Sketch(
                                mutated_program_string, 
                                mutated_program_column, 
                                self.n_iterations, 
                                self.tree_max_nodes
                            )      
                script.save_file_custom(dir_path, self.filename + '_iteration_' + str(i))

                # Generate the graphs with current data
                self.generate_report()

                with open(self.filename + '/' + 'log_' + self.filename + '.txt', 'a') as f:
                    print('Iteration -', i, 'New program accepted - ', 
                        'V/L/D Selfplay = ', victories, losses, draws, 
                        'V/L/D against Glenn = ', v_glenn, l_glenn, d_glenn, 
                        'V/L/D against UCT', self.uct_playouts, 'playouts = ', v_uct, l_uct, d_uct,
                        'Games played = ', self.games_played,
                        file=f)
                    print('Iteration -', i, 
                        'Glenn elapsed time = ', elapsed_time_glenn, 
                        'UCT elapsed time = ', elapsed_time_uct,
                        'Total elapsed time = ', elapsed_time, file=f)
            else:
                elapsed_time = time.time() - start
                with open(self.filename + '/' + 'log_' + self.filename + '.txt', 'a') as f:
                    print('Iteration -', i, '- Elapsed time: ', elapsed_time, 
                            'V/L/D Selfplay = ', victories, losses, draws,
                            'Games played = ', self.games_played, file=f)
        
        best_program_string = self.tree_string.generate_program()
        best_program_column = self.tree_column.generate_program()

        dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.filename + '/'
        script = Sketch(
                        best_program_string, 
                        best_program_column, 
                        self.n_iterations, 
                        self.tree_max_nodes
                    )      
        script.save_file_custom(dir_path, self.filename + '_best_script')

        full_run_elapsed_time = time.time() - full_run
        with open(self.filename + '/' + 'log_' + self.filename + '.txt', 'a') as f:
            print('Full program elapsed time = ', full_run_elapsed_time, file=f)

        return best_program_string, best_program_column, script_best_player, self.tree_string, self.tree_column

    def accept_new_program(self, victories, losses):
        return victories > losses

    def selfplay(self, mutated_player, current_player):
        victories = 0
        losses = 0
        draws = 0
        for i in range(self.n_games):
            game = game = Game(2, 4, 6, [2,12], 2, 2)
            if i%2 == 0:
                    who_won = simplified_play_single_game(
                                                        mutated_player, 
                                                        current_player, 
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
                                                    current_player, 
                                                    mutated_player, 
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

    def generate_report(self):
        
        dir_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.filename + '/' 
        filename = dir_path + self.filename

        y_victories = [vic / self.n_games for vic in self.victories]
        y_losses = [loss / self.n_games for loss in self.losses]
        plt.plot(self.games_played_successful, y_victories, color='green', label='Victory')
        plt.plot(self.games_played_successful, y_losses, color='red', label='Loss')
        plt.legend(loc="best")
        plt.title(str(self.algo_id) + " - Generated script against previous script")
        plt.xlabel('Games played')
        plt.ylabel('Rate - ' + str(self.n_games) + ' games')
        plt.grid()
        plt.yticks([x / 10.0 for x in range(0,11,1)])
        axes = plt.gca()
        axes.set_ylim([-0.05, 1.05])
        plt.savefig(filename + '_vs_previous_script.png')
        plt.close()

        if self.validate:
            y_victories = [vic / self.n_games_glenn for vic in self.victories_against_glenn]
            plt.plot(self.games_played_successful, y_victories, color='green')
            plt.title(str(self.algo_id) + " - Games against Glenn")
            plt.xlabel('Games played')
            plt.ylabel('Victory rate - ' + str(self.n_games_glenn) + ' games')
            plt.grid()
            plt.yticks([x / 10.0 for x in range(0,11,1)])
            axes = plt.gca()
            axes.set_ylim([-0.05, 1.05])
            plt.savefig(filename + '_vs_glenn.png')
            plt.close()

            for i in range(len(self.uct_playouts)):
                victories = [vic[i] / self.n_games_uct for vic in self.victories_against_UCT]  
                plt.plot(self.games_played_uct, victories, label=str(self.uct_playouts[i]) + " playouts")
            plt.legend(loc="best")
            plt.title(str(self.algo_id) + " - Games against UCT")
            plt.xlabel('Games played')
            plt.ylabel('Victory rate - ' + str(self.n_games_uct) + ' games')
            plt.grid()
            plt.yticks([x / 10.0 for x in range(0,11,1)])
            axes = plt.gca()
            axes.set_ylim([-0.05, 1.05])
            plt.savefig(filename + '_vs_UCT.png')
            plt.close()

if __name__ == "__main__":
    algo_id = 'RWSP'
    n_iterations = 200
    tree_max_nodes = 150
    n_games = 1000
    n_games_glenn = 1000
    n_games_uct = 5
    n_games_solitaire = 1000
    uct_playouts = [2, 3, 4]
    eval_step = 5
    max_game_rounds = 1000
    iteration_run = 0
    validate = True
    scripts_to_collect = [100, 200, 500, 1000, 1500, 2000, 5000]
    '''
    yes_no_dsl = DSL('S')
    yes_no_dsl.set_type_action(True)
    column_dsl = DSL('S')
    column_dsl.set_type_action(False)
    '''
    
    yes_no_dsl = ExperimentalDSL('S')
    yes_no_dsl.set_type_action(True)
    column_dsl = ExperimentalDSL('S')
    column_dsl.set_type_action(False)
    
    random_walk_selfplay = RandomWalkSelfplay(
                            algo_id,
                            n_iterations,
                            tree_max_nodes,
                            n_games,
                            n_games_glenn,
                            n_games_uct,
                            n_games_solitaire,
                            uct_playouts,
                            eval_step,
                            max_game_rounds,
                            iteration_run,
                            yes_no_dsl,
                            column_dsl,
                            validate,
                            scripts_to_collect
                        )
    random_walk_selfplay.run()