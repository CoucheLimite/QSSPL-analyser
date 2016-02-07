

from semiconductor.electrical.mobility import Mobility
from semiconductor.electrical.ionisation import Ionisation as Ion
from semiconductor.matterial.ni import IntrinsicCarrierDensity as NI
from semiconductor.recombination.Intrinsic import Radiative


class models_handeller():
    '''

    '''

    available_models = {}

    def __init__(self):
        self._get_available_models()
        self.selected_model = {'ni': 'Couderc2014',
                               'mobility': 'klaassen1992',
                               'ionisation': 'Altermatt2006_table1',
                               'B': 'Altermatt2005'
                               }
        self.use_models = {'ni': NI,
                           'mobility': Mobility,
                           'ionisation': Ion,
                           'B': Radiative
                           }

    def access(self, model, matterial, author, **kwargs):

        return self.use_models[model](
            matterial=matterial, author=author
        ).update(kwargs)

    def _get_available_models(self):
        ni = NI().available_models()
        mobility = Mobility().available_models()
        ionisation = Ion().available_models()
        B = Radiative().available_models()

        values = locals()
        del values['self']
        self.available_models = values
        return self.available_models

    def _auto_select_models(self):
        values = self._get_available_models()

        self.selected_model = {}
        for i, k in zip(values.keys(), values.values()):
            self.selected_model[i] = k[0]
