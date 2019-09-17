from collections import defaultdict
import random
#from game import Board, Game

if __name__ == "__main__":
    #game = Game(2)
    #print(game.roll_dice())
    n = {'a':1, 'b':2, 'c':3}
    #These will initialize value = 0 for whichever keys yet to be added.
    n = defaultdict(lambda: 0, n)
    print(n['a'])
    print(n['b'])
    print(n['c'])
    n['d'] += 1
    print(n['d'])