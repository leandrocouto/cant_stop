import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pylatex import Document, Section, Figure, NoEscape
from pylatex.utils import bold
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

    def info_header_latex(self, doc):
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
        matplotlib.use('Agg')  # Not to use X server. For TravisCI.

        # Data preparation
        
        total_loss = []
        output_dist_loss_history = []
        output_value_loss_history = []
        dist_metric_history = [] 
        value_metric_history = []
        victory_0 = [] # Selfplay - Tie 
        victory_1 = [] # Selfplay
        victory_2 = [] # Selfplay
        loss_eval = [] 
        dist_loss_eval = [] 
        value_loss_eval = [] 
        dist_metric_eval = [] 
        value_metric_eval = [] 
        victory_0_eval = [] # vs. UCT - Tie
        victory_1_eval = [] # vs. UCT
        victory_2_eval = [] # vs. UCT
        victory_0_eval_net = [] # vs. Net - Tie
        victory_1_eval_net = [] # vs. Net
        victory_2_eval_net = [] # vs. Net
        for analysis in self.eval_net_vs_uct:
            total_loss.append(analysis[0])
            output_dist_loss_history.append(analysis[1])
            output_value_loss_history.append(analysis[2])
            dist_metric_history.append(analysis[3])
            value_metric_history.append(analysis[4])
            victory_0.append(analysis[5])
            victory_1.append(analysis[6])
            victory_2.append(analysis[7])
            loss_eval.append(analysis[8])
            dist_loss_eval.append(analysis[9])
            value_loss_eval.append(analysis[10])
            dist_metric_eval.append(analysis[11])
            value_metric_eval.append(analysis[12])
            victory_0_eval.append(analysis[13])
            victory_1_eval.append(analysis[14])
            victory_2_eval.append(analysis[15])
        for analysis in self.eval_net_vs_net:
            victory_0_eval_net.append(analysis[0])
            victory_1_eval_net.append(analysis[1])
            victory_2_eval_net.append(analysis[2])
        

        geometry_options = {"right": "2cm", "left": "2cm"}
        pdf_name = str(self.n_simulations)+'_'+str(self.n_games) \
                + '_' + str(self.alphazero_iterations) + '_' + str(self.conv_number) + \
                '_' + str(self.use_UCT_playout)
        doc = Document(pdf_name, geometry_options=geometry_options)
        
        # Total loss
        with doc.create(Section('Total loss')):
            self.info_header_latex(doc)

            x = np.array(range(1, len(total_loss) + 1))
            y = np.array(total_loss)
            a = np.array(range(1, len(loss_eval) + 1))
            b = np.array(loss_eval)
            _, ax = plt.subplots()
            ax.plot(x, y, label='Training')
            ax.plot(a, b, label='vs. UCT')
            ax.legend(loc="upper right")
            ax.set(xlabel='AZ Iterations', ylabel='Loss', 
            title='Total loss')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

        # Probability distribution loss
        with doc.create(Section('Probability distribution loss')):
            self.info_header_latex(doc)
            x = np.array(range(1, len(output_dist_loss_history) + 1))
            y = np.array(output_dist_loss_history)
            a = np.array(range(1, len(dist_loss_eval) + 1))
            b = np.array(dist_loss_eval)

            _, ax = plt.subplots()
            ax.plot(x, y, label='Training')
            ax.plot(a, b, label='vs. UCT')
            ax.legend(loc="upper right")
            ax.set(xlabel='AZ Iterations', ylabel='Loss', 
            title='Probability distribution loss')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

        plt.close()
        doc.append(NoEscape(r'\newpage'))

        # Value loss
        with doc.create(Section('Value loss')):
            self.info_header_latex(doc)
            x = np.array(range(1, len(output_value_loss_history) + 1))
            y = np.array(output_value_loss_history)
            a = np.array(range(1, len(value_loss_eval) + 1))
            b = np.array(value_loss_eval)

            _, ax = plt.subplots()
            ax.plot(x, y, label='Training')
            ax.plot(a, b, label='vs. UCT')
            ax.legend(loc="upper right")
            ax.set(xlabel='AZ Iterations', ylabel='Loss', 
            title='Value loss')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

            plt.close()
            doc.append(NoEscape(r'\newpage'))

        # Probability Distribution Crossentropy error
        with doc.create(Section('Probability Distribution Crossentropy error')):
            self.info_header_latex(doc)
            x = np.array(range(1, len(dist_metric_history) + 1))
            y = np.array(dist_metric_history)
            a = np.array(range(1, len(dist_metric_eval) + 1))
            b = np.array(dist_metric_eval)

            _, ax = plt.subplots()
            ax.plot(x, y, label='Training')
            ax.plot(a, b, label='vs. UCT')
            ax.legend(loc="upper right")
            ax.set(xlabel='AZ Iterations', ylabel='Crossentropy error', 
            title='Probability Distribution Crossentropy error')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

            plt.close()
            doc.append(NoEscape(r'\newpage'))

        # Value MSE Error
        with doc.create(Section('Value Mean Squared Error')):
            self.info_header_latex(doc)
            x = np.array(range(1, len(value_metric_history) + 1))
            y = np.array(value_metric_history)
            a = np.array(range(1, len(value_metric_eval) + 1))
            b = np.array(value_metric_eval)

            _, ax = plt.subplots()
            ax.plot(x, y, label='Training')
            ax.plot(a, b, label='vs. UCT')
            ax.legend(loc="upper right")
            ax.set(xlabel='AZ Iterations', ylabel='MSE', 
            title='Value Mean Squared Error')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

            plt.close()
            doc.append(NoEscape(r'\newpage'))

        # Player victories in selfplay
        with doc.create(Section('Player victories in selfplay')):
            self.info_header_latex(doc)
            x = np.array(range(1, len(victory_1) + 1))
            y = np.array(victory_1)
            a = np.array(range(1, len(victory_2) + 1))
            b = np.array(victory_2)
            m = np.array(range(1, len(victory_0) + 1))
            n = np.array(victory_0)

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
            with doc.create(Section('Player victories in evaluation - Net vs. Net')):
                self.info_header_latex(doc)   
                x = np.array(range(1, len(self.eval_net_vs_net) + 1))
                y = np.array(victory_1_eval_net)
                a = np.array(range(1, len(self.eval_net_vs_net) + 1))
                b = np.array(victory_2_eval_net)
                m = np.array(range(1, len(self.eval_net_vs_net) + 1))
                n = np.array(victory_0_eval_net)

                _, ax = plt.subplots()
                ax.plot(x, y, label='Player 1')
                ax.plot(a, b, label='Player 2')
                ax.plot(m, n, label='Draw')
                ax.legend(loc="upper right")
                ax.set(xlabel='AZ Iterations', ylabel='Number of victories', 
                title='Player victories in evaluation - Net vs. Net')

                with doc.create(Figure(position='htbp')) as plot:
                    plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

                plt.close()
                doc.append(NoEscape(r'\newpage'))

        # Player victories in evaluation - Net vs. UCT
        with doc.create(Section('Player victories in evaluation - Net vs. UCT')):
            self.info_header_latex(doc)  
            x = np.array(range(1, len(victory_1_eval) + 1))
            y = np.array(victory_1_eval)
            a = np.array(range(1, len(victory_2_eval) + 1))
            b = np.array(victory_2_eval)
            m = np.array(range(1, len(victory_0_eval) + 1))
            n = np.array(victory_0_eval)

            _, ax = plt.subplots()
            ax.plot(x, y, label='Player 1')
            ax.plot(a, b, label='Player 2')
            ax.plot(m, n, label='Draw')
            ax.legend(loc="upper right")
            ax.set(xlabel='AZ Iterations', ylabel='Number of victories', 
            title='Player victories in evaluation - Net vs. UCT')

            with doc.create(Figure(position='htbp')) as plot:
                plot.add_plot(width=NoEscape(r'1\textwidth'), dpi=300)

            plt.close()
            doc.append(NoEscape(r'\newpage'))

        doc.generate_pdf(clean_tex=False)