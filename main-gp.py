import sys

from players.scripts.GP import GP

def main():
    if len(sys.argv[1:]) < 8:
        print('Usage for synthesizing scripts: main-gp generations mutation_rate population_size elite tournament_size number_matches invaders run_id')
        return
    
    generations = int(sys.argv[1])
    mutation_rate = float(sys.argv[2])
    population_size = int(sys.argv[3])
    elite = int(sys.argv[4])
    tournament_size = int(sys.argv[5])
    number_matches = int(sys.argv[6])
    number_invaders = int(sys.argv[7])
    run_id = int(sys.argv[8])
    
    gp = GP(generations, mutation_rate, population_size, elite, tournament_size, number_matches, number_invaders, run_id)
    gp.evolve()
    
if __name__ == "__main__":
    main()