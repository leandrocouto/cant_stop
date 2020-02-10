import importlib
import random
import glob
import copy
import os

from players.scripts.DSL import DSL
from players.scripts.Script import Script
from game import Game

class GP:
    def __init__(self, generations, mutation_rate, population_size, elite, tournament_size, number_matches, invaders, run_id = 0):
        self._generations = generations
        self._mutation_rate = mutation_rate
        self._population_size = population_size
        self._elite = elite
        self._number_matches = number_matches
        self._tournament_size = tournament_size
        self._invaders = invaders
        self._id_counter = 0
        self._scripts_instances = {}
        
        parameters_str = str(generations) + '_' + str(mutation_rate).replace('.', '') + '_' + str(population_size) + '_' + str(elite) + '_' \
        + str(tournament_size) + '_' + str(number_matches) + '_' + str(invaders)
        
        self._path_run = 'players/scripts/generated/' + parameters_str + '/' + str(run_id) + '/' 
        
        os.makedirs(self._path_run)
        
        self._population = []
        
        self._dsl = DSL()
        
        for _ in range(self._population_size):
            script = self._dsl.generateRandomScript(self._id_counter)
            script.saveFile(self._path_run)
            self._population.append(script)
            self._id_counter += 1
            
    def _computeFitness(self):
        for k in range(self._number_matches):
            for i in range(len(self._population)):
                for j in range(i + 1, len(self._population)):
                    result, is_over = self._play_match(self._population[i], self._population[j])
                    self._population[i].incrementMatchesPlayed()
                    self._population[j].incrementMatchesPlayed()
                                                            
                    if result == 1 and is_over:
                        self._population[i].addFitness(1)
                        self._population[j].addFitness(-1)
                    elif result == 2 and is_over:
                        self._population[i].addFitness(-1)
                        self._population[j].addFitness(1)
                    else:
                        self._population[i].addFitness(-1)
                        self._population[j].addFitness(-1)
                        
                    result, is_over = self._play_match(self._population[j], self._population[i])
                    self._population[i].incrementMatchesPlayed()
                    self._population[j].incrementMatchesPlayed()
                     
                    if result == 1 and is_over:
                        self._population[i].addFitness(-1)
                        self._population[j].addFitness(1)
                    elif result == 2 and is_over:
                        self._population[i].addFitness(1)
                        self._population[j].addFitness(-1)
                    else:
                        self._population[i].addFitness(-1)
                        self._population[j].addFitness(-1)
    
    def _tournament(self):
        best_individual = self._population[random.randint(0, self._population_size - 1)]
        for _ in range(0, self._tournament_size - 1):
            rand_individual = self._population[random.randint(0, self._population_size - 1)]
            if best_individual.getFitness() < rand_individual.getFitness():
                best_individual = rand_individual
        return best_individual
    
    def _crossover(self, parent1, parent2):
        children = []
        
        child1_1, child2_1 = parent1.generateSplit()
        child1_2, child2_2 = parent2.generateSplit()
        
        ind1 = child1_1 + child2_2
        ind2 = child1_2 + child2_1
        if len(ind1) > 0:
            child = Script(ind1)
            children.append(child)
        if len(ind2) > 0:
            child = Script(ind2)
            children.append(child)
        return children
    
    def evolve(self, crossover=True):
        for k in range(self._generations):
            print('***Generation ', k+1, '***')
            next_population = []
            
            self._computeFitness()
            
            self._population.sort(reverse=True, key=lambda ind: ind.getFitness())
            
            print('ID, Fitness, Matches Played')
            for i in range(self._population_size):
                print('===================================================================================')
                print('Rank: ', i + 1, '\t Id: ', self._population[i].getId(), '\t Fitness: ', self._population[i].getFitness(), 
                      '\t Matches Played: ', self._population[i].getMatchesPlayed())
                
                #Removing unsued rules in the scripts
                script_name = 'Script' + str(self._population[i].getId())
                counter_calls = self._scripts_instances[script_name].get_counter_calls()
                self._population[i].remove_unused_rules(counter_calls)
                self._population[i].print()
                print()
            print()
            
            #removing from memory instances of scripts of the previous generation
            self._scripts_instances = {} 
            
            #elite individuals
            for i in range(self._elite):
#                 print('Adding elite: ', i)
#                 self._population[i].print()
                next_population.append(copy.deepcopy(self._population[i]))
#             print()

            #adding invaders
            for _ in range(self._invaders):
                script = self._dsl.generateRandomScript()
                next_population.append(script)
            
            #reproduction
            while len(next_population) < len(self._population):
                if crossover:
                    parent1 = self._tournament()
                    parent2 = self._tournament()
                    
                    children = self._crossover(parent1, parent2)
                    
                    for c in children:
                        c.mutate(self._mutation_rate, self._dsl)
                        next_population.append(c)
                        
                        if len(next_population) == self._population_size:
                            break
                else:
                    random_elite = next_population[random.randint(0, self._elite - 1)]
                    copy_random_elite = copy.deepcopy(random_elite)
                    copy_random_elite.forcedMutation(self._mutation_rate, self._dsl)
                    next_population.append(copy_random_elite)
            
            self._population = next_population
            
            #Cleaning-up folder with scripts 
            self._clean_folder()
            
            for ind in self._population:
                ind.clearAttributes()
                ind.setId(self._id_counter)
                
                ind.saveFile(self._path_run)
                self._id_counter += 1
                
    def _clean_folder(self):
        files = glob.glob(self._path_run + '*.py')
        for f in files:
            os.remove(f)
                
    def _play_match(self, script1, script2):
        
        script1_name = 'Script' + str(script1.getId())
        script2_name = 'Script' + str(script2.getId())
        
        if script1_name not in self._scripts_instances:
            module = importlib.import_module(self._path_run.replace('/', '.') + script1_name)
            class_ = getattr(module, 'Script' + str(script1.getId()))
            instance_script1 = class_()
            self._scripts_instances[script1_name] = instance_script1
        else:
            instance_script1 = self._scripts_instances[script1_name]
        
        if script2_name not in self._scripts_instances:
            module = importlib.import_module(self._path_run.replace('/', '.') + script2_name)
            class_ = getattr(module, 'Script' + str(script2.getId()))
            instance_script2 = class_()
            self._scripts_instances[script2_name] = instance_script2
        else:
            instance_script2 = self._scripts_instances[script2_name]
        
#         game = Game(n_players = 2, dice_number = 4, dice_value = 6, column_range = [2, 12], offset = 2, initial_height = 2)
        game = Game(n_players = 2, dice_number = 4, dice_value = 3, column_range = [2, 6], offset = 2, initial_height = 1)
        
        is_over = False
        who_won = None
    
        number_of_moves = 0
        while not is_over:
            moves = game.available_moves()
            if game.is_player_busted(moves):
                continue
            else:
                if game.player_turn == 1:
                    chosen_play = instance_script1.get_action(game)
                else:
                    chosen_play = instance_script2.get_action(game)
                game.play(chosen_play)
                number_of_moves += 1
            
            if number_of_moves >= 200:
                return -1, False
            
            who_won, is_over = game.is_finished()
        return who_won, is_over
        
        