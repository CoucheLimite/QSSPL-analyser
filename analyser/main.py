#! py -2


from models.models import models_handeller
import matplotlib.pylab as plt

from analysis.analysis import data_loader


model_handeller = models_handeller()


if __name__ == '__main__':

    folder = input('Please enter the folder:')
    sample = input('Please enter the sample name:')

    data = data_loader(folder, sample)

    for i in ['doping_type', 'doping', 'thickness', 'reflection']:
        if data.attr[i] is None:
            var = input('Please enter the ' + i + ':')
            try:
                data.attr = {i: float(var)}
            except:
                data.attr = {i: var}

    print(data.attr['Fs'], data.attr['Ai'])
    data.attr = {'Fs': 4e20}
    data.attr = {'Ai': 3.8e22}

    fs = data.self_consistant_generation(0, 'PC')

    data.attr = {'Fs': fs}
    data.calculate()
    ax = data.plot()

    Ai = data.self_consistant_PL(0)
    print(Ai)

    data.attr = {'Ai': Ai}
    data.calculate()
    ax = data.plot()
    # print(fs, data.attr['thickness'], data.attr['doping'], data.attr['reflection'])
    #
    # # plt.ion()
    # data.save()
plt.show()
