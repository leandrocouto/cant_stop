import math
import sys
import pickle
import time
sys.path.insert(0,'..')
from MetropolisHastings.metropolis_hastings import MetropolisHastings

class SimulatedAnnealing(MetropolisHastings):
    def __init__(self, beta, player_1, player_2, n_games, n_iterations, k,
        threshold, tree_max_nodes, 
        dataset_name, n_cores, d, init_temp):
        """
        Metropolis Hastings with temperature schedule. This allows the 
        algorithm to explore more the space search.
        - d is a constant for the temperature schedule.
        - init_temp is the temperature used for the first iteration. Following
          temperatures are calculated following self.temperature_schedule().
        """

        super().__init__(beta, player_1, player_2, n_games, n_iterations, k,
        threshold, tree_max_nodes, dataset_name, n_cores)
        self.d = d
        self.temperature = init_temp


    def run(self):
        """ Main routine of the SA algorithm. """

        # Read the dataset
        with open(self.dataset_name, "rb") as f:
            while True:
                try:
                    self.data.append(pickle.load(f))
                except EOFError:
                    break
        full_run = time.time()
        # Sample k data instances to be used in the evaluation
        if self.k == -1:
            self.k = len(self.data)
        #initial_data = self.sample_from_data()
        initial_data = self.sample_data_from_importance_threshold()

        self.tree.build_tree(self.tree.root)

        # Main loop
        for i in range(2, self.n_iterations + 2):
            start = time.time()
            # Make a copy of the tree for future mutation
            new_tree = pickle.loads(pickle.dumps(self.tree, -1))

            new_tree.mutate_tree()

            current_program = self.tree.generate_program()
            mutated_program = new_tree.generate_program()

            #print('current')
            #print(current_program)
            #print()

            script_best_player = self.tree.generate_player(
                                                        current_program, 
                                                        self.k, 
                                                        self.n_iterations, 
                                                        self.tree_max_nodes
                                                        )
            script_mutated_player = new_tree.generate_player(
                                                        mutated_program, 
                                                        self.k, 
                                                        self.n_iterations, 
                                                        self.tree_max_nodes
                                                        )

            score_best, _, _, _ = self.calculate_score_function(
                                                        script_best_player, 
                                                        initial_data
                                                        )
            score_mutated, errors_mutated, errors_rate_mutated, data_distribution = self.calculate_score_function(
                                                        script_mutated_player, 
                                                        initial_data
                                                        )
            n_errors = errors_mutated[0]
            n_errors_yes_action = errors_mutated[1]
            n_errors_no_action = errors_mutated[2]
            n_errors_numeric_action = errors_mutated[3]
            total_errors_rate = errors_rate_mutated[0]
            total_yes_errors_rate = errors_rate_mutated[1]
            total_no_errors_rate = errors_rate_mutated[2]
            total_numeric_errors_rate = errors_rate_mutated[3]

            self.data_distribution = data_distribution

            # Update score given the SA parameters
            new_score_mutated = score_mutated**(1 / self.temperature)
            new_score_best = score_best**(1 / self.temperature)

            # Accept program only if new score is higher.
            accept = min(1, new_score_mutated/new_score_best)

            # Adjust the temperature accordingly.
            self.temperature = self.temperature_schedule(i)

            self.all_results.append(
                                        (
                                            n_errors,
                                            n_errors_yes_action,
                                            n_errors_no_action,
                                            n_errors_numeric_action,
                                            total_errors_rate,
                                            total_yes_errors_rate,
                                            total_no_errors_rate,
                                            total_numeric_errors_rate
                                        )
                                    )
            # If the new synthesized program is better
            if accept == 1:
                self.tree = new_tree
                self.passed_results.append(
                                            (
                                                n_errors,
                                                n_errors_yes_action,
                                                n_errors_no_action,
                                                n_errors_numeric_action,
                                                total_errors_rate,
                                                total_yes_errors_rate,
                                                total_no_errors_rate,
                                                total_numeric_errors_rate
                                            )
                                        )
                print('Iteration -', i, 'New program accepted - Score = ', 
                        score_mutated,'Error rate = ', errors_rate_mutated, 
                        'n_errors = ', n_errors)

            elapsed_time = time.time() - start
            print('Iteration -', i, '- Elapsed time: ', elapsed_time)
            
        best_program = self.tree.generate_program()
        script_best_player = self.tree.generate_player(
                                                        best_program, 
                                                        self.k, 
                                                        self.n_iterations, 
                                                        self.tree_max_nodes
                                                        )

        full_run_elapsed_time = time.time() - full_run
        print('Full program elapsed time = ', full_run_elapsed_time)

        return best_program, script_best_player

    def temperature_schedule(self, iteration):
        """ Calculate the next temperature used for the score calculation. """

        return self.d/math.log(iteration)