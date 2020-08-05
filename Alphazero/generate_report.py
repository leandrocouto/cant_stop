from statistics import Statistic
import tkinter.filedialog
from os import listdir
from os.path import isfile, join
import re



def main():

    # User should specify the root folder of where all the Statistic files are
    # located.

    # The report is generated in the same directory of where this file is.
    root = tkinter.Tk()
    root.withdraw()
    file_path = tkinter.filedialog.askdirectory()
    print(file_path)
    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    valid_files = []
    for file in files:
        if 'h5' not in file and '_player' not in file:
            valid_files.append(file)

    # Sort the files using natural sorting
    # Source: https://stackoverflow.com/questions/5967500
    #            /how-to-correctly-sort-a-string-with-a-number-inside

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]

    valid_files.sort(key=natural_keys)

    data_net_vs_net_training = [] 
    data_net_vs_net_eval = [] 
    data_net_vs_uct = []
    n_simulations = None
    n_games = None
    alphazero_iterations = None
    use_UCT_playout = None
    conv_number = None
    for file in valid_files:
        file = file_path + '/' + file
        stats = Statistic()
        stats.load_from_file(file)
        data_net_vs_net_training.append(stats.data_net_vs_net_training[0]) 
        data_net_vs_net_eval.append(stats.data_net_vs_net_eval[0])
        # In case the new network was worse than the old one, the UCT data
        # does not exist, therefore there's no data
        if stats.data_net_vs_uct:
            data_net_vs_uct.append(stats.data_net_vs_uct[0]) 
        n_simulations = stats.n_simulations
        n_games = stats.n_games
        alphazero_iterations = stats.alphazero_iterations
        use_UCT_playout = stats.use_UCT_playout
        conv_number = stats.conv_number
    stats = Statistic(
            data_net_vs_net_training, data_net_vs_net_eval, 
            data_net_vs_uct, n_simulations, n_games,
            alphazero_iterations, use_UCT_playout, conv_number
            )
    stats.generate_report()

if __name__ == "__main__":
    main()