import sys
sys.path.insert(0,'..')
from MetropolisHastings.selfplay_simulated_annealing import SimulatedAnnealingSelfplay
from MetropolisHastings.fictitious_play import FictitiousPlay
from MetropolisHastings.new_hybrid_simulated_annealing import NewHybridSimulatedAnnealing

if __name__ == "__main__":
    # Cluster configurations
    if int(sys.argv[1]) == 0: 
        beta = 0.5
        n_iterations = 10000
        tree_max_nodes = 100
        d = 1
        init_temp = 1
        temp_decrease = 0.95
        n_games = 1000
        n_games_glenn = 1000
        n_games_uct = 50
        n_uct_playouts = 50
        max_game_rounds = 500
        threshold = 0
        string_dataset = 'fulldata_sorted_string'
        column_dataset = 'fulldata_sorted_column'

        hybrid_SA = NewHybridSimulatedAnnealing(
                                            beta,
                                            n_iterations,
                                            tree_max_nodes,
                                            d,
                                            init_temp,
                                            temp_decrease,
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
    elif int(sys.argv[1]) == 1: 
        n_selfplay_iterations = 10000
        n_SA_iterations = 200
        tree_max_nodes = 100
        d = 1
        init_temp = 1
        n_games_evaluate = 100
        n_games_glenn = 1000
        n_games_uct = 50
        n_uct_playouts = 50
        max_game_rounds = 500

        fictitious_play = FictitiousPlay(
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
        fictitious_play.selfplay()
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
