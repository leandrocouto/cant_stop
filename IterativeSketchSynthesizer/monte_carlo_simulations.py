import time
import sys
import pickle
import math
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from DSL import *
from rule_of_28_sketch import Rule_of_28_Player_PS

sys.path.insert(0,'..')
from game import Game
from play_game_template import simplified_play_single_game

class MonteCarloSimulation:
	def __init__(self, hole_program, n_simulations, n_games, max_game_rounds, to_parallel, hole_number, to_log):
		self.hole_program = hole_program
		self.n_simulations = n_simulations
		self.n_games = n_games
		self.max_game_rounds = max_game_rounds
		self.to_parallel = to_parallel
		self.hole_number = hole_number
		self.folder = str(self.hole_number) + 'hole_MCsim' + str(self.n_simulations) + '_Ngames' + str(self.n_games) + '/'
		self.to_log = to_log
		if to_log:
			if not os.path.exists(self.folder):
				os.makedirs(self.folder)
		self.log_file = self.folder + 'log.txt'
		self.curr_id = 0

	def id_tree_nodes(self, node):

		node.id = self.curr_id
		self.curr_id += 1
		for child in node.children:
			self.id_tree_nodes(child)

	def get_object(self, program):

		program_yes_no = Sum(Map(Function(Times(Plus(NumberAdvancedThisRound(), Constant(1)), VarScalarFromArray('progress_value'))), VarList('neutrals')))
		return Rule_of_28_Player_PS(program_yes_no, program)

	def get_glenn_player(self):

		program_yes_no = Sum(Map(Function(Times(Plus(NumberAdvancedThisRound(), Constant(1)), VarScalarFromArray('progress_value'))), VarList('neutrals')))
		program_decide_column = Argmax(Map(Function(Sum(Map(Function(Minus(Times(NumberAdvancedByAction(), VarScalarFromArray('move_value')), Times(VarScalar('marker'), IsNewNeutral()))), NoneNode()))), VarList('actions')))
		return Rule_of_28_Player_PS(program_yes_no, program_decide_column)

	def get_traversal(self, program):
		list_of_nodes = []
		self._get_traversal(program, list_of_nodes)
		return list_of_nodes

	def _get_traversal(self, program, list_of_nodes):
		list_of_nodes.append(program)
		for child in program.children:
			self._get_traversal(child, list_of_nodes)

	def update_parent(self, program, parent):
		program.add_parent(parent)
		for child in program.children:
			self.update_parent(child, program)

	def update_children(self, program):
		children = program.children
		if children:
			program.add_children(children)
		for child in children:
			self.update_children(child)

	def finish_tree(self, node, chosen_node):
		"""
		chosen_node is the DSL term (a string) that will substitute node. The 
		remaining children will be built recursively.
		"""
		parent = node.parent
		if chosen_node == 'VarScalar':
			acceptable_nodes = [VarScalar('marker')]
			new_node = random.choice(acceptable_nodes)
			return new_node
		elif chosen_node == 'VarScalarFromArray':
			acceptable_nodes = ['progress_value', 'move_value']
			chosen = random.choice(acceptable_nodes)
			if chosen == 'progress_value':
				new_node = VarScalarFromArray('progress_value')
			else:
				new_node = VarScalarFromArray('move_value')
			return new_node
		elif chosen_node == 'VarList':
			acceptable_nodes = ['actions', 'neutrals', 'None']
			chosen = random.choice(acceptable_nodes)
			if chosen == 'actions':
				new_node =  VarList('actions')
			elif chosen == 'neutrals':
				new_node = VarList('neutrals')
			else:
				new_node = NoneNode()
			return new_node
		elif chosen_node == 'functions_scalars':
			acceptable_nodes = ['NumberAdvancedThisRound', 'NumberAdvancedByAction', 'IsNewNeutral', 'PlayerColumnAdvance', 'OpponentColumnAdvance']
			chosen = random.choice(acceptable_nodes)
			if chosen == 'NumberAdvancedThisRound':
				new_node = NumberAdvancedThisRound()
			elif chosen == 'NumberAdvancedByAction':
				new_node = NumberAdvancedByAction()
			elif chosen == 'IsNewNeutral':
				new_node = IsNewNeutral()
			elif chosen == 'PlayerColumnAdvance':
				new_node = PlayerColumnAdvance()
			elif chosen == 'OpponentColumnAdvance':
				new_node = OpponentColumnAdvance()
			return new_node
		elif chosen_node == 'Times':
			acceptable_nodes = ['VarScalar', 'VarScalarFromArray', 'functions_scalars']
			chosen_left = random.choice(acceptable_nodes)
			chosen_right = random.choice(acceptable_nodes)
			# Left
			if chosen_left == 'VarScalar':
				chosen_node_left = self.finish_tree(node, 'VarScalar')
			elif chosen_left == 'VarScalarFromArray':
				chosen_node_left = self.finish_tree(node, 'VarScalarFromArray')
			else:
				chosen_node_left = self.finish_tree(node, 'functions_scalars')
			# Right
			if chosen_right == 'VarScalar':
				chosen_node_right = self.finish_tree(node, 'VarScalar')
			elif chosen_right == 'VarScalarFromArray':
				chosen_node_right = self.finish_tree(node, 'VarScalarFromArray')
			else:
				chosen_node_right = self.finish_tree(node, 'functions_scalars')
			new_node = Times(chosen_node_left, chosen_node_right)
			return new_node
		elif chosen_node == 'Plus':
			acceptable_nodes = ['VarScalar', 'VarScalarFromArray', 'functions_scalars',
								'Times', 'Plus', 'Minus',
							]
			chosen_left = random.choice(acceptable_nodes)
			chosen_right = random.choice(acceptable_nodes)
			# Left
			if chosen_left == 'VarScalar':
				chosen_node_left = self.finish_tree(node, 'VarScalar')
			elif chosen_left == 'VarScalarFromArray':
				chosen_node_left = self.finish_tree(node, 'VarScalarFromArray')
			elif chosen_left == 'functions_scalars':
				chosen_node_left = self.finish_tree(node, 'functions_scalars')
			elif chosen_left == 'Times':
				chosen_node_left = self.finish_tree(node, 'Times')
			elif chosen_left == 'Plus':
				chosen_node_left = self.finish_tree(node, 'Plus')
			else:
				chosen_node_left = self.finish_tree(node, 'Minus')
			# Right
			if chosen_right == 'VarScalar':
				chosen_node_right = self.finish_tree(node, 'VarScalar')
			elif chosen_right == 'VarScalarFromArray':
				chosen_node_right = self.finish_tree(node, 'VarScalarFromArray')
			elif chosen_right == 'functions_scalars':
				chosen_node_right = self.finish_tree(node, 'functions_scalars')
			elif chosen_right == 'Times':
				chosen_node_right = self.finish_tree(node, 'Times')
			elif chosen_right == 'Plus':
				chosen_node_right = self.finish_tree(node, 'Plus')
			else:
				chosen_node_right = self.finish_tree(node, 'Minus')
			new_node = Plus(chosen_node_left, chosen_node_right)
			return new_node
		elif chosen_node == 'Minus':
			acceptable_nodes = ['VarScalar', 'VarScalarFromArray', 'functions_scalars',
								'Times', 'Plus', 'Minus',
							]
			chosen_left = random.choice(acceptable_nodes)
			chosen_right = random.choice(acceptable_nodes)
			# Left
			if chosen_left == 'VarScalar':
				chosen_node_left = self.finish_tree(node, 'VarScalar')
			elif chosen_left == 'VarScalarFromArray':
				chosen_node_left = self.finish_tree(node, 'VarScalarFromArray')
			elif chosen_left == 'functions_scalars':
				chosen_node_left = self.finish_tree(node, 'functions_scalars')
			elif chosen_left == 'Times':
				chosen_node_left = self.finish_tree(node, 'Times')
			elif chosen_left == 'Plus':
				chosen_node_left = self.finish_tree(node, 'Plus')
			else:
				chosen_node_left = self.finish_tree(node, 'Minus')
			# Right
			if chosen_right == 'VarScalar':
				chosen_node_right = self.finish_tree(node, 'VarScalar')
			elif chosen_right == 'VarScalarFromArray':
				chosen_node_right = self.finish_tree(node, 'VarScalarFromArray')
			elif chosen_right == 'functions_scalars':
				chosen_node_right = self.finish_tree(node, 'functions_scalars')
			elif chosen_right == 'Times':
				chosen_node_right = self.finish_tree(node, 'Times')
			elif chosen_right == 'Plus':
				chosen_node_right = self.finish_tree(node, 'Plus')
			else:
				chosen_node_right = self.finish_tree(node, 'Minus')
			new_node = Minus(chosen_node_left, chosen_node_right)
			return new_node
		elif chosen_node == 'Sum':
			acceptable_nodes = ['VarList', 'Map']
			chosen = random.choice(acceptable_nodes)
			if chosen == 'VarList':
				chosen_node = self.finish_tree(node, 'VarList')
			else:
				chosen_node = self.finish_tree(node, 'Map')
			new_node = Sum(chosen_node)
			return new_node
		elif chosen_node == 'Map':
			# Function
			acceptable_nodes_1 = ['Function']
			# VarList
			acceptable_nodes_2 = ['VarList']
			chosen_left = random.choice(acceptable_nodes_1)
			chosen_right = random.choice(acceptable_nodes_2)
			if chosen_left == 'Function':
				chosen_node_left = self.finish_tree(node, 'Function')
			if chosen_right == 'VarList':
				chosen_node_right = self.finish_tree(node, 'VarList')
			new_node = Map(chosen_node_left, chosen_node_right)
			return new_node
		elif chosen_node == 'Function':
			acceptable_nodes = ['Times', 'Plus', 'Minus', 'Sum', 'Map', 'Function']
			chosen = random.choice(acceptable_nodes)
			if chosen == 'Times':
				chosen_node = self.finish_tree(node, 'Times')
			elif chosen == 'Plus':
				chosen_node = self.finish_tree(node, 'Plus')
			elif chosen == 'Minus':
				chosen_node = self.finish_tree(node, 'Minus')
			elif chosen == 'Sum':
				chosen_node = self.finish_tree(node, 'Sum')
			elif chosen == 'Map':
				chosen_node = self.finish_tree(node, 'Map')
			else:
				chosen_node = self.finish_tree(node, 'Function')
			new_node = Function(chosen_node)
			return new_node
		elif chosen_node == 'Argmax':
			acceptable_nodes = ['VarList', 'Map']
			chosen = random.choice(acceptable_nodes)
			if chosen == 'VarList':
				chosen_node = self.finish_tree(node, 'VarList')
			elif chosen == 'Map':
				chosen_node = self.finish_tree(node, 'Map')
			new_node = Argmax(chosen_node)
			return new_node
		else:
			raise Exception('Unhandled DSL term at finish_tree. DSL term = ', chosen_node)

	def find_replacement(self, node):
		"""
		Find a replacement for node according to the DSL. In this implementation,
		this node will also be deleted; therefore, it is needed to look at this
		node's parent and apply the appropriate changes. 
		"""

		parent = node.parent

		# Check if HoleNode is the root
		if node.parent is None:
			acceptable_nodes = ['Argmax',
								'Map',
								'Sum', 
								'Function',
								'Plus',
								'Minus',
								'Times'
							]
			chosen_node = random.choice(acceptable_nodes)
			# Finish the tree with the chosen substitute
			new_node = self.finish_tree(node, chosen_node)
		# Check for Times
		elif parent.className() == 'Times':
			acceptable_nodes = ['VarScalar', 
								'VarScalarFromArray',
								'functions_scalars'
							]
			chosen_node = random.choice(acceptable_nodes)
			# Finish the tree with the chosen substitute
			new_node = self.finish_tree(node, chosen_node)
		elif parent.className() == 'Plus' or parent.className() == 'Minus':
			acceptable_nodes = ['VarScalar', 
								'VarScalarFromArray', 
								'functions_scalars',
								#'Constant',
								'Times',
								'Plus',
								'Minus'
								]
			chosen_node = random.choice(acceptable_nodes)
			# Finish the tree with the chosen substitute
			new_node = self.finish_tree(node, chosen_node)
		elif parent.className() == 'Function':
			acceptable_nodes = ['Times', 
								'Plus', 
								'Minus', 
								'Sum', 
								'Map', 
								'Function'
							]
			chosen_node = random.choice(acceptable_nodes)
			# Finish the tree with the chosen substitute
			new_node = self.finish_tree(node, chosen_node)
		elif parent.className() == 'Argmax' or parent.className() == 'Sum':
			acceptable_nodes = ['VarList', 
								'Map'
							]
			chosen_node = random.choice(acceptable_nodes)
			# Finish the tree with the chosen substitute
			new_node = self.finish_tree(node, chosen_node)
		elif parent.className() == 'Map':
			# Map is a little different because it has two distinctive children,
			# so it must be done checked separately

			# Check the special case "HoleNode"
			if node.className() == 'HoleNode':
				chosen_node = random.choice(['Function', 'VarList'])
				if chosen_node == 'Function':
					# Finish the tree with the chosen substitute
					new_node = self.finish_tree(node, 'Function')
				else:
					# It is a VarList
					new_node = self.finish_tree(node, 'VarList')
			elif node.className() == 'Function':
				# Finish the tree with the chosen substitute
				new_node = self.finish_tree(node, 'Function')
			else:
				# It is a VarList
				new_node = self.finish_tree(node, 'VarList')
		else:
			raise Exception('Unhandled parent at find_replacement. parent = ', parent)

		# Add replacement to the original tree
		self.update_parent(new_node, None)
		return new_node

	def finish_holes(self, initial_node):
		""" Finish HoleNodes randomly. """

		# Special case: root is a HoleNode
		if isinstance(initial_node, HoleNode):
			replacement = self.find_replacement(initial_node)
			replacement.parent = None
			return replacement
		else:
			list_of_nodes = self.get_traversal(initial_node)
			for node in list_of_nodes:
				if isinstance(node, HoleNode):
					replacement = self.find_replacement(node)
					# Add replacement to the original tree
					node.parent.children[:] = [replacement if child==node else child for child in node.parent.children]
				else:
					node.can_mutate = False
			return initial_node

	def run(self):

		start = time.time()

		average_victories = []
		average_losses = []
		average_draws = []
		self.update_parent(self.hole_program, None)
		self.curr_id = 0
		self.id_tree_nodes(self.hole_program)
		if self.to_log:
			with open(self.log_file, 'a') as f:
				print('Program to be MC simulated:', self.hole_program.to_string(), file=f)
				print(file=f)
		for i in range(self.n_simulations):
			start_sim = time.time()
			curr_hole_program = pickle.loads(pickle.dumps(self.hole_program, -1))
			# Fill HoleNodes
			curr_hole_program = self.finish_holes(curr_hole_program)
			# Update the children of the nodes in a top-down approach
			self.update_children(curr_hole_program)
			self.curr_id = 0
			self.id_tree_nodes(curr_hole_program)
			# Update the parent of the nodes
			#self.update_parent(curr_hole_program, None)
			if self.to_log:
				with open(self.log_file, 'a') as f:
					print('MC simulations - Iteration -', i, file=f)
					print('Program finished randomly:', curr_hole_program.to_string(), file=f)
			# Transform into a "playable" player
			curr_player = self.get_object(curr_hole_program)
			glenn_player = self.get_glenn_player()
			try:
				if self.to_parallel:
					v, l, d = self.evaluate_parallel(curr_player, glenn_player)
				else:
					v, l, d = self.evaluate(curr_player, glenn_player)
			# If program generates an exception, do not 'accept' it by
			# giving a 0 score
			except Exception as e:
				if self.to_log:
					with open(self.log_file, 'a') as f:
						print('Invalid program (exception).', file=f)
						print('Exception = ', str(e), file=f)
				v, l, d = 0, 0, 0
			average_victories.append(v)
			average_losses.append(l)
			average_draws.append(d)

			if self.to_log:
				# Average
				final_average_vic = sum(average_victories)/len(average_victories)
				final_average_loss = sum(average_losses)/len(average_losses)
				final_average_draw = sum(average_draws)/len(average_draws)
				# Standard deviation
				final_std_vic = math.sqrt( sum([abs(value - final_average_vic) for value in average_victories]) / len(average_victories))
				final_std_loss = math.sqrt( sum([abs(value - final_average_loss) for value in average_losses]) / len(average_losses))
				final_std_draw = math.sqrt( sum([abs(value - final_average_draw) for value in average_draws]) / len(average_draws))

				end_sim = time.time() - start_sim

				with open(self.log_file, 'a') as f:
					print('End of simulation.', file=f)
					print('Info on this simulation', file=f)
					print('V/L/D against Glenn = ', v, l, d, file=f)
					print('Info so far', file=f)
					print('Avg vics = ', average_victories, file=f)
					print('Avg losses = ', average_losses, file=f)
					print('Avg draws = ', average_draws, file=f)
					print('Final victories avg and std = ', final_average_vic, final_std_vic, file=f)
					print('Final losses avg and std = ', final_average_loss, final_std_loss, file=f)
					print('Final draws avg and std = ', final_average_draw, final_std_draw, file=f)
					print('Time elapsed of this simulation = ', end_sim, file=f)
					print('Time elapsed so far = ', time.time() - start, file=f)
					print(file=f)




		# End of MC

		# Average
		final_average_vic = sum(average_victories)/len(average_victories)
		final_average_loss = sum(average_losses)/len(average_losses)
		final_average_draw = sum(average_draws)/len(average_draws)
		# Standard deviation
		final_std_vic = math.sqrt( sum([abs(value - final_average_vic) for value in average_victories]) / len(average_victories))
		final_std_loss = math.sqrt( sum([abs(value - final_average_loss) for value in average_losses]) / len(average_losses))
		final_std_draw = math.sqrt( sum([abs(value - final_average_draw) for value in average_draws]) / len(average_draws))

		end = time.time() - start

		if self.to_log:
			with open(self.log_file, 'a') as f:
				print('End of MC simulations of the program = ', self.hole_program.to_string(), file=f)
				print(file=f)
				print('Avg vics = ', average_victories, file=f)
				print('Avg losses = ', average_losses, file=f)
				print('Avg draws = ', average_draws, file=f)
				print(file=f)
				print('Final victories avg and std = ', final_average_vic, final_std_vic, file=f)
				print('Final losses avg and std = ', final_average_loss, final_std_loss, file=f)
				print('Final draws avg and std = ', final_average_draw, final_std_draw, file=f)
				print(file=f)
				print('Time elapsed = ', end, file=f)

		return final_average_vic, final_average_draw, final_average_loss

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

	def evaluate_helper(self, args):
		return simplified_play_single_game(args[0], args[1], args[2], args[3])

	def evaluate_parallel(self, player_1, player_2):
		""" 
		Play self.n_games Can't Stop games between player_1 and player_2. The
		players swap who is the first player per iteration because Can't Stop
		is biased towards the player who plays the first move.
		"""

		victories = 0
		losses = 0
		draws = 0

		# First with the current script as first player, then the opposite

		# ProcessPoolExecutor() will take care of joining() and closing()
		# the processes after they are finished.
		with ProcessPoolExecutor(max_workers=8) as executor:
			# Specify which arguments will be used for each parallel call
			args_1 = (
						(
						player_1, player_2, Game(2, 4, 6, [2,12], 2, 2), 
						self.max_game_rounds
						) 
					for _ in range(self.n_games // 2)
					)
			results_1 = executor.map(self.evaluate_helper, args_1)

		# Current script is now the second player
		with ProcessPoolExecutor(max_workers=8) as executor:
			# Specify which arguments will be used for each parallel call
			args_2 = (
						(
						player_2, player_1, Game(2, 4, 6, [2,12], 2, 2), 
						self.max_game_rounds
						) 
					for _ in range(self.n_games // 2)
					)
			results_2 = executor.map(self.evaluate_helper, args_2)

		for result in results_1:
			if result == 1:
				victories += 1
			elif result == 2:
				losses += 1
			else:
				draws += 1 

		for result in results_2:
			if result == 1:
				losses += 1
			elif result == 2:
				victories += 1
			else:
				draws += 1 
		return victories, losses, draws

if __name__ == "__main__":
	
	hole_program = [
					HoleNode(),
					Argmax(HoleNode()),
					Argmax(Map(HoleNode(), HoleNode())),
					Argmax(Map(Function(HoleNode()), VarList('actions'))),
					Argmax(Map(Function(Sum(HoleNode())), VarList('actions'))),
					Argmax(Map(Function(Sum(Map(Function(HoleNode()), NoneNode()))), VarList('actions'))),
					Argmax(Map(Function(Sum(Map(Function(Minus(Times(HoleNode(), HoleNode()), HoleNode())), NoneNode()))), VarList('actions'))),
				]
	
	chosen = int(sys.argv[1])

	n_simulations = 10
	n_games = 100
	max_game_rounds = 500
	to_parallel = False
	to_log = True
	start_MC = time.time()
	MC = MonteCarloSimulation(hole_program[chosen], n_simulations, n_games, max_game_rounds, to_parallel, chosen, to_log)
	v, l, d = MC.run()
	end_MC = time.time() - start_MC
	print('Time elapsed = ', end_MC)
