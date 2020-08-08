import math
import sys
import pickle
import time
sys.path.insert(0,'..')
from MetropolisHastings.metropolis_hastings import MetropolisHastings

class SimulatedAnnealing(MetropolisHastings):
    def __init__(self, beta, n_iterations, threshold, tree_max_nodes, 
        string_dataset, column_dataset, sub_folder_batch, file_name, d, init_temp):
        """
        Metropolis Hastings with temperature schedule. This allows the 
        algorithm to explore more the space search.
        - d is a constant for the temperature schedule.
        - init_temp is the temperature used for the first iteration. Following
          temperatures are calculated following self.temperature_schedule().
        """

        super().__init__(beta, n_iterations, threshold, tree_max_nodes, 
            string_dataset, column_dataset, sub_folder_batch, file_name)
        self.d = d
        self.temperature = init_temp


    def run(self):
        """ Main routine of the SA algorithm. """

        # Read the dataset
        with open(self.string_dataset, "rb") as f:
            while True:
                try:
                    self.string_data.append(pickle.load(f))
                except EOFError:
                    break
        with open(self.column_dataset, "rb") as f:
            while True:
                try:
                    self.column_data.append(pickle.load(f))
                except EOFError:
                    break
        full_run = time.time()

        data_string, data_column = self.sample_data_from_importance_threshold()

        self.tree_string.build_tree(self.tree_string.root)
        self.tree_column.build_tree(self.tree_column.root)

        # Main loop
        for i in range(2, self.n_iterations + 2):
            start = time.time()
            # Make a copy of the tree for future mutation
            new_tree_string = pickle.loads(pickle.dumps(self.tree_string, -1))
            new_tree_column = pickle.loads(pickle.dumps(self.tree_column, -1))

            new_tree_string.mutate_tree()
            new_tree_column.mutate_tree()

            current_program_string = self.tree_string.generate_program()
            mutated_program_string = new_tree_string.generate_program()

            current_program_column = self.tree_column.generate_program()
            mutated_program_column = new_tree_column.generate_program()

            script_best_player = self.generate_player(
                                                current_program_string, 
                                                current_program_column
                                                )
            script_mutated_player = self.generate_player(
                                                mutated_program_string,
                                                mutated_program_column
                                                )

            score_best, _, _, _ = self.calculate_score_function(
                                                        script_best_player, 
                                                        data_string,
                                                        data_column
                                                        )
            score_mutated, errors_mutated, errors_rate_mutated, data_distribution = self.calculate_score_function(
                                                        script_mutated_player, 
                                                        data_string,
                                                        data_column
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
                self.tree_string = new_tree_string
                self.tree_column = new_tree_column
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
                with open(self.sub_folder_batch + '/' + self.file_name + '.txt', 'a') as f:
                    print('Iteration -', i, 'New program accepted - Score = ', 
                            score_mutated,'Error rate = ', errors_rate_mutated, 
                            'n_errors = ', n_errors, file=f)

            elapsed_time = time.time() - start
            with open(self.sub_folder_batch + '/' + self.file_name + '.txt', 'a') as f:
                print('Iteration -', i, '- Elapsed time: ', elapsed_time, file=f)
        
        best_program_string = self.tree_string.generate_program()
        best_program_column = self.tree_column.generate_program()
        script_best_player = self.generate_player(
                                                best_program_string,
                                                best_program_column
                                                )

        full_run_elapsed_time = time.time() - full_run
        with open(self.sub_folder_batch + '/' + self.file_name + '.txt', 'a') as f:
            print('Full program elapsed time = ', full_run_elapsed_time, file=f)

        return best_program_string, best_program_column, script_best_player, self.tree_string, self.tree_column

    def temperature_schedule(self, iteration):
        """ Calculate the next temperature used for the score calculation. """

        return self.d/math.log(iteration)