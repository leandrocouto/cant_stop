import math
import sys
import pickle
import time
sys.path.insert(0,'..')
from MetropolisHastings.metropolis_hastings import MetropolisHastings

class SimulatedAnnealing(MetropolisHastings):
    def __init__(self, beta, n_iterations, threshold, init_temp, tree_max_nodes, 
        string_dataset, column_dataset, sub_folder_batch, file_name, d):
        """
        Metropolis Hastings with temperature schedule. This allows the 
        algorithm to explore more the space search.
        - d is a constant for the temperature schedule.
        """

        super().__init__(beta, n_iterations, threshold, init_temp, tree_max_nodes, 
            string_dataset, column_dataset, sub_folder_batch, file_name)
        self.d = d

    def update_score(self, score_best, score_mutated):
        """ 
        Update the score according to the current temperature. 
        """
        
        new_score_best = score_best**(1 / self.temperature)
        new_score_mutated = score_mutated**(1 / self.temperature)
        return new_score_best, new_score_mutated

    def temperature_schedule(self, iteration):
        """ Calculate the next temperature used for the score calculation. """

        return self.d/math.log(iteration)