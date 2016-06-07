

from semiconductor.electrical.mobility import Mobility
from semiconductor.electrical.ionisation import Ionisation as Ion
from semiconductor.matterial.ni import IntrinsicCarrierDensity as NI
from semiconductor.recombination.Intrinsic import Radiative


class models_handeller():
    '''

    '''

    available_models = {}
    matterial = 'Si'

    def __init__(self):
        self._get_available_models()
        self.selected_model = {'ni': 'Couderc_2014',
                               'mobility': 'klaassen1992',
                               'ionisation': 'Altermatt2006_table1',
                               'B': 'Altermatt2005'
                               }
        self._update_update()

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

    def _update_update(self):
        '''
        creates a dictionary that holds
        the required semiconudctor models for easy
        calling
        '''

        self.update = {
            'ni': NI(
                matterial=self.matterial,
                author=self.selected_model['ni']
                ).update_ni,
            'mobility': Mobility(
                matterial=self.matterial,
                author=self.selected_model['mobility']
                ).mobility_sum,
            'ionisation': Ion(
                matterial=self.matterial,
                author=self.selected_model['ionisation']
                ).update_dopant_ionisation,
            'B': Radiative(
                matterial=self.matterial,
                author=self.selected_model['B']
                ).B
        }
    def _auto_select_models(self):
        values = self._get_available_models()

        self.selected_model = {}
        for i, k in zip(values.keys(), values.values()):
            self.selected_model[i] = k[0]
