
import random
from board import Board

class Game:
	def __init__(self, n_players):
		self.board = Board()
		self.n_players = n_players
	def update_board(self, player_id, dice_combination):
		"""player_id refers to either 1, 2, 3 or 4
		dice_combination is 2-tuple chosen by the player
		"""
		print()
	def is_player_busted(self, player_id, dice):
		"""player_id refers to either 1, 2, 3 or 4
		dice is a 2-tuple representing the dice values
		Returns a boolean
		"""
		print()
	def roll_dice(self):
		return (random.randrange(7), random.randrange(7))
	def get_possible_moves(self, player_id, dice):
		"""player_id refers to either 1, 2, 3 or 4
		dice is a 2-tuple representing the dice values
		Returns a list of possible combinations player_id can play 
		based on the current board schematic
		"""
		print()

if __name__ == "__main__":
    game = Game(2)
    print(game.roll_dice())