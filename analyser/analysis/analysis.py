

import numpy as np
import matplotlib.pylab as plt
import os

from glob import glob

from models.NumericalDifferentiation_windows import Finite_Difference
from models.models import models_handeller

from utils.importexport import LoadData
import scipy.constants as C

from . import caculate_quantities as CQ
import semiconductor.electrical as el


def find_nearest(array, value):
    idx = (abs(array - value)).argmin()
    return idx


class data_loader():

    def __init__(self, folder=None, sample=None):
        self.folder = folder
        self.sample = sample

        self.model_handeller = models_handeller()
        if self.folder is not None and self.sample is not None:
            self.load()

    def load(self, folder=None, sample=None):
        self.folder = folder or self.folder
        self.sample = sample or self.sample

        self.datas = []

        for fname in glob(os.path.join(self.folder, self.sample) + '*_Raw_Data.dat'):
            self.datas.append(Data(fname))
            self.datas[-1].ChoosingDefultCropValues()

    def calculate(self):
        for i in self.datas:
            i.CalculateLifetime(
                model_handeller=self.model_handeller)

    def plot(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1)

        j = 0
        for i in self.datas:
            ax.plot(i.DeltaN_PC, i.Tau_PC, '-', label=str(j))
            ax.plot(i.DeltaN_PL, i.Tau_PL, '-')
            j += 1

        ax.semilogx()
        ax.semilogy()
        ax.set_xlabel('Excess carrier density (cm^-3)')
        ax.set_ylabel('Lifetime (s^-1)')

        return ax

    def plot_raw(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1)

        j = 0
        for i in self.datas:
            print(i.RawData2.shape, i.RawData2.dtype.names)
            print(i.RawData.shape)
            ax.plot(i.RawData['Time'], i.RawData['PL'], '-')
            ax.plot(i.RawData['Time'], i.RawData['Gen'], '--')
            ax.plot(i.RawData['Time'], i.RawData['PC'], '.-')

        ax.semilogx()
        ax.semilogy()
        ax.set_xlabel('time')
        ax.set_ylabel('voltage')

        return ax

    @property
    def attr(self):
        return self.datas[0].wafer_inf

    def adjust(self, var, percent):
        dic = {var: self.datas[0].wafer_inf[var] * (1 + percent / 100.)}
        self.set_attr(**dic)
        self.calculate()

    @attr.setter
    def attr(self, kwargs):

        for key, value in kwargs.items():
            if key in self.datas[0].wafer_inf:
                for i in self.datas:
                    i.wafer_inf[key] = value

    def adjust(self, var, percent):
        dic = {var: self.datas[0].wafer_inf[var] * (1 + percent / 100.)}
        self.attr = dic
        self.calculate()

    def save(self):
        for data in self.datas:
            data.save()


class Data():

    Derivitive = 'Finite Difference'
    Analysis = 'Generalised'
    Temp = -1

    def __init__(self, fname=None):
        self.LD = LoadData()

        if fname is not None:
            self.Load_Measurements(fname)

    def _update_ni(self, model_handeller):
        if self.Temp != self.wafer_inf['Temp']:
            self.ni = model_handeller.update['ni'](temp=self.wafer_inf['Temp'])
            self.Temp = self.wafer_inf['Temp']
            self.Vt = C.k * self.Temp / C.e
        pass

    def save(self):

        data = np.vstack((
            self.DeltaN_PC,
            self.Tau_PC,
            self.DeltaN_PL,
            self.Tau_PL
        )).T

        self.LD.save(data, self.wafer_inf)

    def background_doping(self, model_handeller):
        if (self.wafer_inf['doping_type'] == 'p'):
            self.Na = self.wafer_inf['doping']
            self.Nd = 0
            self.wafer_inf['dopant'] = 'boron'

        elif(self.wafer_inf['doping_type'] == 'n'):
            self.Nd = self.wafer_inf['doping']
            self.Na = 0
            self.wafer_inf['dopant'] = 'phosphorous'

        self.ne0, self.nh0 = el.get_carriers(Na=self.Na,
                                             Nd=self.Nd,
                                             nxc=0,
                                             temp=self.wafer_inf['Temp'],
                                             ni_author=model_handeller.selected_model[
                                                 'ni'],
                                             ionisation_author=model_handeller.selected_model['ionisation'])

    def UpdateInfData(self):
        self.LD.WriteTo_Inf_File(self.wafer_inf)

    def dndt(self, Deltan):

        if (self.Derivitive == 'Finite Difference'):
            dn_dt = Finite_Difference().FourPointCentral(
                self.Data['Time'], Deltan)

        else:
            print('You fucked up.... again')

        return dn_dt

    def Load_Measurements(self, fname):

        if os.sep in fname:
            self.LD.fname = fname

        self.RawData = self.LD.Load_RawData_File()
        self.RawData2 = np.copy(self.RawData)
        self.wafer_inf = self.LD.Load_InfData_File()

    def ChoosingDefultCropValues(self):

        _cropping_index(self.RawData, self.wafer_inf)

    def CalculateLifetime(self, BackGroundShow=False, model_handeller=None):

        # make sure the ni is updated
        self._update_ni(model_handeller)

        # determine the background concentration of carriers
        self.background_doping(model_handeller)

        # Background correction stuff

        self.Data, self.DarkConductance = _background_correction(
            self.RawData, self.wafer_inf)

        self.Data = crop_data(self.Data, self.wafer_inf)

        self.Data = bin_named_array(self.Data, self.wafer_inf['binning_pp'])

        model_handeller._update_update()

        self.DeltaN_PC = CQ.nxc_from_photoconductance(
            self.Data['PC'],
            self.wafer_inf['thickness'],
            self.wafer_inf['Temp'],
            self.ne0,
            self.nh0,
            model_handeller)

        self.DeltaN_PL = CQ.nxc_from_photoluminescence(
            self.Data['PL'] / self.wafer_inf['gain_pl'],
            self.wafer_inf['Ai'],
            self.wafer_inf['dopant'],
            self.Na,
            self.Nd,
            self.wafer_inf['Temp'],
            model_handeller)

        self.Tau_PC = self.DeltaN_PC / self.Generation('PC')
        self.Tau_PL = self.DeltaN_PL / self.Generation('PL')

        # CQ.doping_from_pc(self.DarkConductance,
        #                   self.wafer_inf, model_handeller)

        return self.Tau_PC, self.Tau_PL

    def Generation(self, PCorPL, suns=False):
        try:
            Gen = getattr(
                self, 'Generation_' + self.Analysis.replace(' ', '_'))
        except:
            print('Choice of generation doesn\'t exist: You fucked up')

        if suns == True:
            scale = self.wafer_inf['thickness'] / 2.5e17
        else:
            scale = 1.

        return Gen(PCorPL) * scale

    def Generation_Steady_State(self, PCorPL):
        Trans = (1 - self.wafer_inf['reflection'] / 100.)
        return self.Data['Gen'] * self.wafer_inf['Fs'] /self.wafer_inf['gain_gen'] * Trans / self.wafer_inf['thickness']

    def Generation_Generalised(self, PCorPL):
        Trans = (1 - self.wafer_inf['reflection'] / 100.)

        gen = self.Generation_Steady_State(PCorPL) + self.Generation_Transient(PCorPL)

        return gen

    def Generation_Transient(self, PCorPL):
        if PCorPL == 'PC':
            gen = -self.dndt(self.DeltaN_PC)
        elif PCorPL == 'PL':
            gen = -self.dndt(self.DeltaN_PL)
        else:
            gen = 0
            print('You fucked up the Generation')

        return gen


def crop_data(data, wafer_inf):
    # this just uses that when points are negitive they should no longer be
    # used
    maxindex = np.amax(data.shape)

    index = np.arange(int(wafer_inf[
        'cropStart'] / 100 * maxindex), int(wafer_inf['cropEnd'] / 100 * maxindex), 1)

    data = data[index]
    return data


def bin_named_array(data, no_pts_bnd):
    '''
    Binnes a named around the provided amount
    '''

    if len(data.dtype.names) != 1:
        data2 = np.copy(data)[::no_pts_bnd]

    for i in data.dtype.names:
        for j in range(data.shape[0] // no_pts_bnd):

            data2[i][j] = np.mean(
                data[i][j * no_pts_bnd:(j + 1) * no_pts_bnd], axis=0)

    return data2


def _background_correction(data_in, wafer_inf, bkg_percent=0.95):
    '''
    Background corrects the data
    '''
    Data = np.copy(data_in)
    if 'bkh_pl' in wafer_inf:

        for name in data_in.dtype.names:
            if name in ['PL', 'Gen']:
                Data[name] -= wafer_inf['bkg_' + name.lower()]
            elif name in ['PC']:
                Data[name] = CQ.pc_from_voltage(Data[name], wafer_inf['Quad'],
                                                wafer_inf['Lin'], wafer_inf['Const'])

                DarkConductance = CQ.pc_from_voltage(wafer_inf['bkg_' + name.lower()],
                                                     wafer_inf['Quad'],
                                                     wafer_inf['Lin'], wafer_inf['Const'])

                Data[name] -= DarkConductance

    else:
        bkg_index = int(data_in.shape[0] * bkg_percent)
        for name in data_in.dtype.names:
            if name in ['PL', 'Gen']:
                Data[name] -= np.average(Data[name][bkg_index:])

            elif name in ['PC']:
                Data[name] = CQ.pc_from_voltage(Data[name], wafer_inf['Quad'],
                                                wafer_inf['Lin'], wafer_inf['Const'])

                DarkConductance = np.average(Data['PC'][bkg_index:])
                Data[name] -= DarkConductance

        return Data, DarkConductance


def _cropping_index(data, wafer_inf):
    '''
    Tries to find the cropping based on the noise in the PL channel
    '''

    if wafer_inf['cropStart'] == None and wafer_inf['cropStart'] == None:
        if 'bkg_pl_std' in wafer_inf:
            index = data['PL'] > wafer_inf['bkg_pl_std'] * 7

            indexs = np.linspace(0, 1, data['PL'].shape[0])
            wafer_inf['cropStart'], wafer_inf[
                'cropEnd'] = indexs[index][[0, -1]] * 100
        else:

            wafer_inf['cropStart'], wafer_inf['cropEnd'] = 5, 95
