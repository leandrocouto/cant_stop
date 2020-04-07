import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pylatex import Document, Section, Figure, NoEscape
from pylatex.utils import bold
import numpy as np
import pickle
import os

class Statistic:
    def __init__(self, data_net_vs_net_training = None, 
                    data_net_vs_net_eval = None, data_net_vs_uct = None, 
                    n_simulations = None, n_games = None, 
                    alphazero_iterations = None, use_UCT_playout = None, 
                    conv_number = None
                    ):
        self.data_net_vs_net_training = data_net_vs_net_training
        self.data_net_vs_net_eval = data_net_vs_net_eval
        self.data_net_vs_uct = data_net_vs_uct
        self.n_simulations = n_simulations
        self.n_games = n_games
        self.alphazero_iterations = alphazero_iterations
        self.use_UCT_playout = use_UCT_playout
        self.conv_number = conv_number
        if n_simulations != None and n_games != None \
            and alphazero_iterations != None \
            and use_UCT_playout != None and conv_number != None:
            self.path = 'data_' + str(self.n_simulations) + '_' \
                + str(self.n_games) + '_' + str(self.alphazero_iterations) \
                + '_' + str(self.conv_number) + '_' \
                + str(self.use_UCT_playout) + '/'
                

    def save_to_file(self, alphazero_iteration):
        """Save the analysis data of this iteration to file."""

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        filename = self.path + '/' + str(self.n_simulations) + '_' \
                + str(self.n_games) + '_' + str(self.alphazero_iterations) \
                + '_' + str(self.conv_number) + '_' \
                + str(self.use_UCT_playout) + '_' + str(alphazero_iteration)
        with open(filename, 'wb') as file:
            pickle.dump(self.__dict__, file)

    def load_from_file(self, path):
        """
        Load data from file to self object.
        path is a dictionary where key is Statistic attributes and 
        value is the data.
        """

        with open(path, 'rb') as file:
            stats = pickle.load(file)
            self.data_net_vs_net_training = stats['data_net_vs_net_training']
            self.data_net_vs_net_eval = stats['data_net_vs_net_eval']
            self.data_net_vs_uct = stats['data_net_vs_uct']
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
        filename = self.path + '/' + str(self.n_simulations) + '_' \
                + str(self.n_games) + '_' + str(self.alphazero_iterations) \
                + '_' + str(self.conv_number) + '_' \
                + str(self.use_UCT_playout) + '_' \
                + str(alphazero_iteration) + '_model.h5'
        model.save(filename)

    def info_header_latex(self, doc):
        """Create a header used before each graph in the report."""

        doc.append(bold('Number of UCT simulations: '))
        doc.append(str(self.n_simulations))
        doc.append(bold('\nNumber of games in selfplay: '))
        doc.append(str(self.n_games))
        doc.append(bold('\nUsing standard rollout in UCT: '))
        doc.append(str(self.use_UCT_playout))
        doc.append(bold('\nNumber of convolutional layers: '))
        doc.append(str(self.conv_number))
        doc.append(bold('\nNumber of AZ iterations: '))
        doc.append(str(self.alphazero_iterations))

    def generate_report(self):
        """Generate and export the report."""

        matplotlib.use('Agg')

        # Data preparation
        
        training_total_loss = []
        training_output_dist_loss = []
        training_output_value_loss = []
        training_dist_metric = [] 
        training_value_metric = []
        training_victory_0 = [] # Selfplay - Tie 
        training_victory_1 = [] # Selfplay
        training_victory_2 = [] # Selfplay

        victory_0_eval_net = [] # vs. Net - Tie
        victory_1_eval_net = [] # vs. Net
        victory_2_eval_net = [] # vs. Net

        victory_0_eval_uct_temp = [] 
        victory_1_eval_uct_temp = [] 
        victory_2_eval_uct_temp = [] 
        list_of_n_simulations = []
        
        for analysis in self.data_net_vs_net_training:
            training_total_loss.append(analysis[0])
            training_dist_metric.append(analysis[1])
            training_value_metric.append(analysis[2])
            training_victory_0.append(analysis[3])
            training_victory_1.append(analysis[4])
            training_victory_2.append(analysis[5])
        for analysis in self.data_net_vs_net_eval:
            victory_0_eval_net.append(analysis[0])
            victory_1_eval_net.append(analysis[1])
            victory_2_eval_net.append(analysis[2])
        for analysis in self.data_net_vs_uct:
            victory_0_eval_uct_temp.append(analysis[0])
            victory_1_eval_uct_temp.append(analysis[1])
            victory_2_eval_uct_temp.append(analysis[2])
            list_of_n_simulations.append(analysis[3])

        victory_0_eval_uct = [] # vs. UCT - Tie
        victory_1_eval_uct = [] # vs. UCT
        victory_2_eval_uct = [] # vs. UCT

        # It can happen that every iteration, new network lost to old one.
        if victory_0_eval_uct_temp:
            for j in range(len(victory_0_eval_uct_temp[0])):
                temp_list_0 = []
                temp_list_1 = []
                temp_list_2 = []
                for i in range(len(victory_0_eval_uct_temp)):
                    temp_list_0.append(victory_0_eval_uct_temp[i][j])
                    temp_list_1.append(victory_1_eval_uct_temp[i][j])
                    temp_list_2.append(victory_2_eval_uct_temp[i][j])
                victory_0_eval_uct.append(temp_list_0)
                victory_1_eval_uct.append(temp_list_1)
                victory_2_eval_uct.append(temp_list_2)       

        geometry_options = {"right": "2cm", "left": "2cm"}
        pdf_name = str(self.n_simulations) + '_' + str(self.n_games) \
                + '_' + str(self.alphazero_iterations) + '_' \
                + str(self.conv_number) + '_' + str(self.use_UCT_playout)
        doc = Document(pdf_name, geometry_options=geometry_options)
        
        # Total loss
        with doc.create(Section('Total loss')):
            self.info_header_latex(doc)
            x = np.array(range(1, len(training_total_loss) + 1))
            y = np.array(training_total_loss)
            _, ax = plt.subplots()
            ax.plot(x, y)
            ax.set(xlabel='AZ Iterations', ylabel='Loss', 
            title='Total loss')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

        # Probability Distribution Crossentropy error
        with doc.create(Section('Probability Distribution Crossentropy error')):
            self.info_header_latex(doc)
            x = np.array(range(1, len(training_dist_metric) + 1))
            y = np.array(training_dist_metric)

            _, ax = plt.subplots()
            ax.plot(x, y)
            ax.set(xlabel='AZ Iterations', ylabel='Crossentropy error', 
            title='Probability Distribution Crossentropy error')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

            plt.close()
            doc.append(NoEscape(r'\newpage'))

        # Value MSE Error
        with doc.create(Section('Value Mean Squared Error')):
            self.info_header_latex(doc)
            x = np.array(range(1, len(training_value_metric) + 1))
            y = np.array(training_value_metric)

            _, ax = plt.subplots()
            ax.plot(x, y)
            ax.set(xlabel='AZ Iterations', ylabel='MSE', 
            title='Value Mean Squared Error')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

            plt.close()
            doc.append(NoEscape(r'\newpage'))

        # Player victories in selfplay
        with doc.create(Section('Player victories in selfplay')):
            self.info_header_latex(doc)
            x = np.array(range(1, len(training_victory_1) + 1))
            y = np.array(training_victory_1)
            a = np.array(range(1, len(training_victory_2) + 1))
            b = np.array(training_victory_2)
            m = np.array(range(1, len(training_victory_0) + 1))
            n = np.array(training_victory_0)

            _, ax = plt.subplots()
            ax.plot(x, y, label='Player 1')
            ax.plot(a, b, label='Player 2')
            ax.plot(m, n, label='Draw')
            ax.legend(loc="upper right")
            ax.set(xlabel='AZ Iterations', ylabel='Number of victories', 
            title='Player victories in selfplay')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

            plt.close()
            doc.append(NoEscape(r'\newpage'))

        # This can happen if the new network always lose to the
        # previous one.
        if len(victory_1_eval_net) != 0:
            # Player victories in evaluation - Net vs. Net
            with doc.create(
                    Section('Player victories in evaluation - Net vs. Net')
                    ):
                self.info_header_latex(doc)   
                x = np.array(range(1, len(self.data_net_vs_net_eval) + 1))
                y = np.array(victory_1_eval_net)
                a = np.array(range(1, len(self.data_net_vs_net_eval) + 1))
                b = np.array(victory_2_eval_net)
                m = np.array(range(1, len(self.data_net_vs_net_eval) + 1))
                n = np.array(victory_0_eval_net)

                _, ax = plt.subplots()
                ax.plot(x, y, label='New Network')
                ax.plot(a, b, label='Old Network')
                ax.plot(m, n, label='Draw')
                ax.legend(loc="upper right")
                ax.set(xlabel='AZ Iterations', ylabel='Number of victories', 
                title='Player victories in evaluation - Net vs. Net')

                with doc.create(Figure(position='htbp')) as plot:
                    plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

                plt.close()
                doc.append(NoEscape(r'\newpage'))

            # Generate these graphs only if there's data available
            if list_of_n_simulations:
                for i in range(len(list_of_n_simulations[0])):
                    # Player victories in evaluation - Net vs. UCT
                    title = 'Player victories in evaluation - Net vs. UCT - '\
                          +  str(list_of_n_simulations[0][i]) + ' simulations'
                    with doc.create(Section(title)):
                        self.info_header_latex(doc)  
                        x = np.array(range(1, len(victory_1_eval_uct[i]) + 1))
                        y = np.array(victory_1_eval_uct[i])
                        a = np.array(range(1, len(victory_2_eval_uct[i]) + 1))
                        b = np.array(victory_2_eval_uct[i])
                        m = np.array(range(1, len(victory_0_eval_uct[i]) + 1))
                        n = np.array(victory_0_eval_uct[i])

                        _, ax = plt.subplots()
                        ax.plot(x, y, label='Network')
                        ax.plot(a, b, label='UCT')
                        ax.plot(m, n, label='Draw')
                        ax.legend(loc="upper right")
                        ax.set(xlabel='AZ Iterations', 
                                ylabel='Number of victories', 
                                title=title
                                )

                        with doc.create(Figure(position='htbp')) as plot:
                            plot.add_plot(width=NoEscape(r'1\textwidth'), \
                                            dpi=300)

                        plt.close()
                        doc.append(NoEscape(r'\newpage'))

        doc.generate_pdf(clean_tex=False)