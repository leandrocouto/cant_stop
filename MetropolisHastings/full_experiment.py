import sys
sys.path.insert(0,'..')
from MetropolisHastings.levi_selfplay_simulated_annealing import LeviSelfplaySimulatedAnnealing
from MetropolisHastings.selfplay_simulated_annealing import SimulatedAnnealingSelfplay
from MetropolisHastings.glenn_simulated_annealing import GlennSimulatedAnnealing
from MetropolisHastings.standard_selfplay import StandardSelfplay
from MetropolisHastings.hybrid_simulated_annealing import HybridSimulatedAnnealing
from MetropolisHastings.simulated_annealing import SimulatedAnnealing

if __name__ == "__main__":
	# Cluster configurations
	if int(sys.argv[1]) == 0: 
		beta = 0.5
		n_iterations = 10000
		tree_max_nodes = 100
		d = 1
		init_temp = 1
		n_games = 1000
		n_games_glenn = 1000
		n_games_uct = 50
		n_uct_playouts = 50
		max_game_rounds = 500

		standard_selfplay = StandardSelfplay(
		                        beta,
		                        n_iterations,
		                        tree_max_nodes,
		                        n_games,
		                        n_games_glenn,
		                        n_games_uct,
		                        n_uct_playouts,
		                        max_game_rounds
		                    )
		standard_selfplay.run()
	elif int(sys.argv[1]) == 1: 
		beta = 0.5
		n_iterations = 10000
		tree_max_nodes = 100
		d = 1
		init_temp = 1
		n_games = 1000
		n_games_glenn = 1000
		n_games_uct = 50
		n_uct_playouts = 50
		max_game_rounds = 500

		selfplay_SA = SimulatedAnnealingSelfplay(
		                                    beta,
		                                    n_iterations,
		                                    tree_max_nodes,
		                                    d,
		                                    init_temp,
		                                    n_games,
		                                    n_games_glenn,
		                                    n_games_uct,
		                                    n_uct_playouts,
		                                    max_game_rounds
		                                )
		selfplay_SA.run()
	elif int(sys.argv[1]) == 2: 
		beta = 0.5
		n_iterations = 10000
		tree_max_nodes = 100
		d = 1
		init_temp = 1
		n_games = 1000
		n_games_glenn = 1000
		n_games_uct = 50
		n_uct_playouts = 50
		max_game_rounds = 500

		glenn_SA = GlennSimulatedAnnealing(
		                                    beta,
		                                    n_iterations,
		                                    tree_max_nodes,
		                                    d,
		                                    init_temp,
		                                    n_games,
		                                    n_games_glenn,
		                                    n_games_uct,
		                                    n_uct_playouts,
		                                    max_game_rounds
		                                )

		glenn_SA.run()
	elif int(sys.argv[1]) == 3: 
		beta = 0.5
		n_iterations = 10000
		tree_max_nodes = 100
		d = 1
		init_temp = 1
		n_games = 1000
		n_games_glenn = 1000
		n_games_uct = 50
		n_uct_playouts = 50
		max_game_rounds = 500
		threshold = 0
		string_dataset = 'fulldata_sorted_string'
		column_dataset = 'fulldata_sorted_column'

		hybrid_SA = HybridSimulatedAnnealing(
		                                    beta,
		                                    n_iterations,
		                                    tree_max_nodes,
		                                    d,
		                                    init_temp,
		                                    n_games,
		                                    n_games_glenn,
		                                    n_games_uct,
		                                    n_uct_playouts,
		                                    max_game_rounds,
		                                    string_dataset,
		                                    column_dataset,
		                                    threshold
		                                )

		hybrid_SA.run()
	elif int(sys.argv[1]) == 4: 
		beta = 0.5
		n_iterations = 10000
		n_games_glenn = 1000
		n_games_uct = 50
		n_uct_playouts = 50
		threshold = 0
		tree_max_nodes = 100
		init_temp = 1
		d = 1
		string_dataset = 'fulldata_sorted_string'
		column_dataset = 'fulldata_sorted_column'
		max_game_rounds = 500

		SA = SimulatedAnnealing(
		                        beta,
		                        n_iterations,
		                        n_games_glenn,
		                        n_games_uct,
		                        n_uct_playouts,
		                        threshold,
		                        init_temp,
		                        d, 
		                        tree_max_nodes,
		                        string_dataset,
		                        column_dataset,
		                        max_game_rounds
		                    )

		SA.run()
	elif int(sys.argv[1]) == 5: 
		n_selfplay_iterations = 200
		n_SA_iterations = 250
		tree_max_nodes = 100
		d = 1
		init_temp = 1
		n_games_evaluate = 100
		n_games_glenn = 1000
		n_games_uct = 50
		n_uct_playouts = 50
		max_game_rounds = 500

		levi_selfplay_SA = LeviSelfplaySimulatedAnnealing(
		                                    n_selfplay_iterations,
		                                    n_SA_iterations,
		                                    tree_max_nodes,
		                                    d,
		                                    init_temp,
		                                    n_games_evaluate,
		                                    n_games_glenn,
		                                    n_games_uct,
		                                    n_uct_playouts,
		                                    max_game_rounds
		                                )
		levi_selfplay_SA.selfplay()
