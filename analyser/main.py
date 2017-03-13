#! py -2


# from models.models import models_handeller
import matplotlib.pylab as plt

from analyser.analysis.analysis import data_loader


# TODO:
# get a gui or sudo gui working.
# this page currently does not work.

# model_handeller = models_handeller()
def go(folder, sample):
    data = data_loader(folder, sample)

    data.check_sample_properties()

    # print(data.attr['Fs'], data.attr['Ai'])
    # data.attr = {'Fs': 4e20}
    # data.attr = {'Ai': 3.8e22}

    # fs = data.self_consistant_generation(0, 'PC')

    # data.attr = {'Fs': fs}
    data.calculate()
    print(data.attr['Fs'])
    # ax = data.plot()

    # Ai = data.self_consistant_PL(0)
    # print(Ai)

    # data.attr = {'Ai': Ai}
    data.calculate()
    ax = data.plot()
    plt.show()

if __name__ == '__main__':

    folder = input('Please enter the folder:')
    sample = input('Please enter the sample name:')

    go(folder, sample)

    # print(fs, data.attr['thickness'], data.attr['doping'], data.attr['reflection'])
    #
    # # plt.ion()
    # data.save()
plt.show()
