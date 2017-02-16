#! py -2


from models.models import models_handeller
import matplotlib.pylab as plt

from analysis.analysis import data_loader


model_handeller = models_handeller()


if __name__ == '__main__':

    # folder = input('Please enter the folder:')
    # sample = input('Please enter the sample name:')

    folder = r'C:\Users\z3186867'
    sample = 'hamid_oxide_'
    data = data_loader(folder, sample)

    for i in ['doping_type', 'doping', 'thickness', 'reflection']:
        if data.attr[i] is None:
            var = input('Please enter the ' + i + ':')
            try:
                data.attr = {i: float(var)}
            except:
                data.attr = {i: var}
        else:
            print(data.attr[i], i)

    data.calculate()

    plt.ion()
    ax = data.plot()
    data.save()
# plt.show()
