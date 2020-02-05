import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pickle
import os

class Statistic:
    def __init__(self, eval_net_vs_net = None, eval_net_vs_uct = None, n_simulations = None, n_games = None,
                    alphazero_iterations = None, use_UCT_playout = None, conv_number = None):
        self.eval_net_vs_net = eval_net_vs_net
        self.eval_net_vs_uct = eval_net_vs_uct
        self.n_simulations = n_simulations
        self.n_games = n_games
        self.alphazero_iterations = alphazero_iterations
        self.use_UCT_playout = use_UCT_playout
        self.conv_number = conv_number
        if n_simulations != None and n_games != None and alphazero_iterations != None and \
            use_UCT_playout != None and conv_number != None:
            self.path = 'data_'+str(self.n_simulations)+'_'+str(self.n_games) \
                + '_' + str(self.alphazero_iterations) + '_' + str(self.conv_number) + \
                '_' + str(self.use_UCT_playout) + '/'
                

    def save_to_file(self, alphazero_iteration):
        """Save the analysis data of this iteration to file."""
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        filename = self.path + '/' + str(self.n_simulations)+'_'+str(self.n_games) \
                + '_' + str(self.alphazero_iterations) + '_' + str(self.conv_number) + \
                '_' + str(self.use_UCT_playout) + '_' + str(alphazero_iteration)
        with open(filename, 'wb') as file:
            pickle.dump(self.__dict__, file)

    def load_from_file(self, path):
        """
        Load data from file to self object.
        path is a dictionary where key is Statistic attributes and value is the data.
        """
        with open(path, 'rb') as file:
            stats = pickle.load(file)
            self.eval_net_vs_net = stats['eval_net_vs_net']
            self.eval_net_vs_uct = stats['eval_net_vs_uct']
            self.n_simulations = stats['n_simulations']
            self.n_games = stats['n_games']
            self.alphazero_iterations = stats['alphazero_iterations']
            self.use_UCT_playout = stats['use_UCT_playout']
            self.conv_number = stats['conv_number']
            self.path = stats['path']


    def save_model_to_file(self, model, alphazero_iteration):
        """Save the networkmodel of this iteration to file."""
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        filename = self.path + '/' + str(self.n_simulations)+'_'+str(self.n_games) \
                + '_' + str(self.alphazero_iterations) + '_' + str(self.conv_number) + \
                '_' + str(self.use_UCT_playout) + '_' + str(alphazero_iteration) + '_model.h5'
        model.save(filename)

    def generate_graphs(self):
        """Generate graphs based on alphazero iterations."""
        save_path = 'graphs_'+str(self.n_simulations)+'_'+str(self.n_games) \
                    + '_' + str(self.alphazero_iterations) + '_' + str(self.conv_number) + \
                    '_' + str(self.use_UCT_playout) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # Data preparation

        total_loss = []
        output_dist_loss_history = []
        output_value_loss_history = []
        dist_metric_history = [] 
        value_metric_history = [] 
        victory_1 = [] 
        victory_2 = [] 
        loss_eval = [] 
        dist_loss_eval = [] 
        value_loss_eval = [] 
        dist_metric_eval = [] 
        value_metric_eval = [] 
        victory_1_eval = [] 
        victory_2_eval = []
        victory_1_eval_net = [] 
        victory_2_eval_net = []
        for analysis in self.eval_net_vs_uct:
            total_loss.append(analysis[0])
            output_dist_loss_history.append(analysis[1])
            output_value_loss_history.append(analysis[2])
            dist_metric_history.append(analysis[3])
            value_metric_history.append(analysis[4])
            victory_1.append(analysis[5])
            victory_2.append(analysis[6])
            loss_eval.append(analysis[7])
            dist_loss_eval.append(analysis[8])
            value_loss_eval.append(analysis[9])
            dist_metric_eval.append(analysis[10])
            value_metric_eval.append(analysis[11])
            victory_1_eval.append(analysis[12])
            victory_2_eval.append(analysis[13])
        for analysis in self.eval_net_vs_net:
            victory_1_eval_net.append(analysis[0])
            victory_2_eval_net.append(analysis[1])

        # Total loss
        x = np.array(range(1, len(total_loss) + 1))
        y = np.array(total_loss)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='Loss', 
            title='Total loss in training')
        ax.grid()
        fig.savefig(save_path + "1_total_loss.png")

        # Probability distribution loss
        x = np.array(range(1, len(output_dist_loss_history) + 1))
        y = np.array(output_dist_loss_history)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='Loss',
                title='Probability distribution loss in training')
        ax.grid()
        fig.savefig(save_path + "2_output_dist_loss_history.png")

        # Value loss
        x = np.array(range(1, len(output_value_loss_history) + 1))
        y = np.array(output_value_loss_history)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='Loss', 
                title='Value loss in training')
        ax.grid()
        fig.savefig(save_path + "3_output_value_loss_history.png")

        # Probability Distribution CE error
        x = np.array(range(1, len(dist_metric_history) + 1))
        y = np.array(dist_metric_history)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='Cross entropy error',
                title='Probability Distribution CE error in training')
        ax.grid()
        fig.savefig(save_path + "4_dist_metric_history.png")

        # Value MSE error
        x = np.array(range(1, len(value_metric_history) + 1))
        y = np.array(value_metric_history)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='MSE error',
                title='Value MSE error in training')
        ax.grid()
        fig.savefig(save_path + "5_value_metric_history.png")

        # Victory of player 1 in training
        x = np.array(range(1, len(victory_1) + 1))
        y = np.array(victory_1)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='Number of victories',
                title='Victory of player 1 in training')
        ax.grid()
        fig.savefig(save_path + "6_victory_1.png")

        # Victory of player 2 in training
        x = np.array(range(1, len(victory_2) + 1))
        y = np.array(victory_2)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='Number of victories',
                title='Victory of player 2 in training')
        ax.grid()
        fig.savefig(save_path + "7_victory_2.png")

        # Total loss in evaluation
        x = np.array(range(1, len(loss_eval) + 1))
        y = np.array(loss_eval)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='Loss',
                title='Total loss in evaluation')
        ax.grid()
        fig.savefig(save_path + "8_loss_eval.png")

        # Probability distribution loss in evaluation
        x = np.array(range(1, len(dist_loss_eval) + 1))
        y = np.array(dist_loss_eval)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='Loss',
                title='Probability distribution loss in evaluation')
        ax.grid()
        fig.savefig(save_path + "9_dist_loss_eval.png")

        # Value loss in evaluation
        x = np.array(range(1, len(value_loss_eval) + 1))
        y = np.array(value_loss_eval)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='Loss',
                title='Value loss in evaluation')
        ax.grid()
        fig.savefig(save_path + "10_value_loss_eval.png")

        # Probability Distribution CE error in evaluation
        x = np.array(range(1, len(dist_metric_history) + 1))
        y = np.array(dist_metric_history)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='Crossentropy error',
                title='Probability Distribution CE error in evaluation')
        ax.grid()
        fig.savefig(save_path + "11_dist_metric_history.png")

        # Value MSE error in evaluation
        x = np.array(range(1, len(value_metric_eval) + 1))
        y = np.array(value_metric_eval)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='MSE error',
                title='Value MSE error in evaluation')
        ax.grid()
        fig.savefig(save_path + "12_value_metric_eval.png")

        # Victory of player 1 in evaluation
        x = np.array(range(1, len(victory_1_eval) + 1))
        y = np.array(victory_1_eval)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='Number of victories',
                title='Victory of player 1 in evaluation - Net vs. UCT')
        ax.grid()
        fig.savefig(save_path + "13_victory_1_eval.png")

        # Victory of player 2 in evaluation
        x = np.array(range(1, len(victory_2_eval) + 1))
        y = np.array(victory_2_eval)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Iterations', ylabel='Number of victories',
                title='Victory of player 2 in evaluation - Net vs. UCT')
        ax.grid()
        fig.savefig(save_path + "14_victory_2_eval.png")

        # This can happen if the new network always lose to the
        # previous one.
        if len(victory_1_eval_net) != 0:
            # Victory of player 1 in evaluation - net vs net
            x = np.array(range(1, len(self.eval_net_vs_net) + 1))
            y = np.array(victory_1_eval_net)
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set(xlabel='Iterations', ylabel='Number of victories',
                    title='Victory of player 1 in evaluation - Net vs. Net')
            ax.grid()
            fig.savefig(save_path + "15_victory_1_eval_net_vs_net.png")

            # Victory of player 2 in evaluation - net vs net
            x = np.array(range(1, len(self.eval_net_vs_net) + 1))
            y = np.array(victory_2_eval_net)
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set(xlabel='Iterations', ylabel='Number of victories',
                    title='Victory of player 2 in evaluation - Net vs. Net')
            ax.grid()
            fig.savefig(save_path + "16_victory_2_eval_net_vs_net.png")