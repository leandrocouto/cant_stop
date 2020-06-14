import sys
sys.path.insert(0,'..')
from game import Game
from MetropolisHastings.DSL import DSL
import pandas as pd
import os.path
import pickle
from sklearn.preprocessing import LabelEncoder
#from sklearn.tree.export import export_graphviz
from sklearn import tree
import sklearn.ensemble

from io import StringIO


def get_features(data):
    features = {}
    features['target'] = []
    features['dice_0'] = []
    features['dice_1'] = []
    features['dice_2'] = []
    features['dice_3'] = []
    features['opponent_score'] = []
    features['player_score'] = []
    for i in range(2, 13):
        features[f'neutral_marker_position_{i}'] = []
        features[f'player_1_marker_position_{i}'] = []
        features[f'player_2_marker_position_{i}'] = []
        features[f'won_column_{i}'] = []
        features[f'finished_column_{i}'] = []
    
    for instance in data:
        state = instance[0]
        action_chosen = instance[1]
        board = state.board_game
        #print(state.print_board())
        #exit()
        for i in range(2, 13):
            list_of_cells = board.board[i]
            neutral_position = -1
            player_1_position = -1
            player_2_position = -1
            for j in range(len(list_of_cells)):
                if 0 in list_of_cells[j].markers:
                    neutral_position = j
                if 1 in list_of_cells[j].markers:
                    player_1_position = j
                if 2 in list_of_cells[j].markers:
                    player_2_position = j
            features[f'neutral_marker_position_{i}'].append(neutral_position)
            features[f'player_1_marker_position_{i}'].append(player_1_position)
            features[f'player_2_marker_position_{i}'].append(player_2_position)

        features['dice_0'].append(state.current_roll[0])
        features['dice_1'].append(state.current_roll[1])
        features['dice_2'].append(state.current_roll[2])
        features['dice_3'].append(state.current_roll[3])
        
        won_indexes = []
        finished_indexes = []
        for won_column in state.player_won_column:
            features[f'won_column_{won_column[0]}'].append(won_column[1])
            won_indexes.append(won_column[0])
        for finished_column in state.finished_columns:
            features[f'finished_column_{finished_column[0]}'].append(finished_column[1])
            finished_indexes.append(finished_column[0])
        # Fill the noncompleted rows with -1
        
        for i in range(2, 13):
            if i not in won_indexes:
                features[f'won_column_{i}'].append(-1)
            if i not in finished_indexes:
                features[f'finished_column_{i}'].append(-1)
        
        features['opponent_score'].append(DSL.get_opponent_score(state))
        features['player_score'].append(DSL.get_player_score(state))

        features['target'].append(action_chosen)

    return features

data = []
print('Reading data...')
with open('dataset', "rb") as f:
    while True:
        try:
            data.append(pickle.load(f))
        except EOFError:
            break
print('Building the data frame...')
features = get_features(data)
# Convert the dictionary into DataFrame 
df = pd.DataFrame(features)

# Transform the target actions to numbers
le_target = LabelEncoder()
df['target_n'] = le_target.fit_transform(df['target'].astype(str))
inputs = df.drop(['target', 'target_n'], axis='columns')
target = df['target_n']

print(df)
model = tree.DecisionTreeClassifier()
#model = sklearn.ensemble.BaggingClassifier()
print('Training now')
model.fit(inputs, target)
print('Score')
print(model.score(inputs, target))

x = model.tree_

print('node count = ', x.node_count)
#out = StringIO()
#out = export_graphviz(model, out_file='tree.dot')
