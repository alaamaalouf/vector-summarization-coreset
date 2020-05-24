import matplotlib.pyplot as plt
import os
import numpy as np
import Utils
import seaborn
import mpltex


class GraphPlotter(object):
    """
    ################# Grapher #################
    Functions:
        - __init__: instructor
        -saveFigure
        - errorFill
        -Graph
    """

    def __init__(self):
        """
        :param directoryName:  A directory that the graphs will be saved at
        """
        self.saveFigWidth = 20  # width of the figure
        self.saveFigHeight = 13  # height of the figure
        self.fontsize = 50  # font size of the letters at the axes
        self.legendfontsize = self.fontsize  # font size of the letters at the legend
        self.labelpad = 10  # padding with respect to the labels of the axes
        self.linewidth = 6  # line width of the graphs
        self.colors = ['blue', '#20B2AA', 'red', 'cyan', 'green', 'magenta', 'darkorange', 'navy']  # used colors
        self.color_matching = Utils.COLOR_MATCHING  # dictionary of colors per each line
        self.line_styles = ['--', '-.', '-', '--', '-.', '-']
        self.markers = ['D', 's', 'D', 'o', 'p', '*']
        self.marker_size = 20

        self.OPEN_FIG = True  # automatically open figure after saving

        # updating plot parameters
        plt.rcParams.update({'font.size': self.fontsize})
        plt.rcParams['xtick.major.pad'] = '{}'.format(self.labelpad * 3)
        plt.rcParams['ytick.major.pad'] = '{}'.format(self.labelpad)
        plt.rcParams['xtick.labelsize'] = self.legendfontsize
        plt.rcParams['ytick.labelsize'] = self.legendfontsize
        seaborn.set_style("whitegrid")


    def saveFigure(self, fileName):
        """

        :param fileName: A string containing the name of the figure
        :return: None
        """
        figure = plt.gcf()
        figure.set_size_inches(self.saveFigWidth, self.saveFigHeight)
        plt.savefig(fileName, bbox_inches='tight')

    def plotGraph(self, x_values, y_values, legend, figure_title, xlabel, ylabel, save_path):
        """
        :param x_values: A numpy array of values at the X axis.
        :param y_values: A numpy array of values at the Y axis with respect to different lines.
        :param legend: A list of names for different lines.
        :param figure_title: The title of the figure.
        :param xlabel: The label of the X axis.
        :param ylabel: The label of the Y axis.
        :param save_path: The physical path to which the file will be saved at.
        :return: None.
        """
        # linestyles = mpltex.linestyle_generator()
        for i in range(y_values.shape[0]):
            # style = next(linestyles)
            # print(style)
            plt.plot(x_values, y_values[i, :], linewidth=self.linewidth,
                     linestyle=self.line_styles[i], color=self.color_matching[legend[i]], marker=self.markers[i],
                     markersize=self.marker_size, markerfacecolor=self.color_matching[legend[i]])

        # set the title of the figure
        plt.title(figure_title, fontsize=self.fontsize)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # set the legend
        plt.legend(legend, loc='best')
        seaborn.set_style("whitegrid")

        # save the figure and open it if needed
        self.saveFigure(save_path)

        # show the plot
        plt.clf()


if __name__ == '__main__':
    A = np.load(r'results/Synthetic/Results_Synthetic.npz')
    x_values = Utils.generateSampleSizes(1000)
    errors = A['mean_error']
    times = A['mean_time']
    legend = ['Uniform Sampling', 'Sensitivity Sampling', 'Caratheodory', 'Dan ICML2016',
                       'Our slow coreset', 'Our fast Coreset']


    graph_plotter = GraphPlotter()
    file_path = r'results/{}/{}-{}.pdf'.format('Synthetic', 'Synthetic', 'error')
    graph_plotter.plotGraph(x_values, errors, legend, 'Synthetic', 'sample size', r'$\varepsilon$', file_path)

    file_path = r'results/{}/{}-{}.pdf'.format('Synthetic', 'Synthetic', 'time')
    graph_plotter.plotGraph(x_values, times, legend, 'Synthetic', 'sample size', 'Overall Time (secs)', file_path)
