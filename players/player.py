from abc import ABC, abstractmethod

class Player(ABC):
    @abstractmethod
    def get_action(self, game):
        """
        Return the action to be made by the player given the 
        game state passed.
        Concrete classes must implement this method.
        """
        pass