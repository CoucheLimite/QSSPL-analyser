

import numpy as np
import matplotlib.pylab as plt
import os

from glob import glob

from analyser.models.models import models_handeller

from analyser.utils.importexport import LoadData
import scipy.constants as C

from . import calculate_quantities as CQ
from . import selfconsistent as SC
import semiconductor.electrical as el


def find_nearest(array, value):
    idx = (abs(array - value)).argmin()
    return idx


# class data_loader():
#       # the idea behind this class is to have a handeller for a single wafer.
#        # lets get there a bit later though
#     def __init__(self, folder=None, sample=None):
#         self.folder = folder
#         self.sample = sample
#
#         self.model_handeller = models_handeller()
#         if self.folder is not None and self.sample is not None:
#             self.load()
#
#     def load(self, folder=None, sample=None):
#         self.folder = folder or self.folder
#         self.sample = sample or self.sample
#
#         self.datas = []
#
#         for fname in glob(os.path.join(self.folder, self.sample) + '*_Raw_Data.dat'):
#             self.datas.append(Data(fname))
#             self.datas[-1].ChoosingDefultCropValues()
#
#     def calculate(self):
#         for i in self.datas:
#             i.calculate_lifetime(
#                 model_handeller=self.model_handeller)
#
#     def plot(self, ax=None):
#
#         if ax is None:
#             fig, ax = plt.subplots(1)
#
#         j = 0
#         for i in self.datas:
#             ax.plot(i.nxc_pc, i.Tau_PC, '-', label=str(j))
#             ax.plot(i.nxc_pl, i.Tau_PL, '-')
#             j += 1
#
#         ax.semilogx()
#         ax.semilogy()
#         ax.set_xlabel('Excess carrier density (cm^-3)')
#         ax.set_ylabel('Lifetime (s^-1)')
#
#         return ax
#
#     def plot_raw(self, ax=None):
#
#         if ax is None:
#             fig, ax = plt.subplots(1)
#
#         j = 0
#         for i in self.datas:
#             ax.plot(i.RawData['time'], i.RawData['PL'], '-')
#             ax.plot(i.RawData['time'], i.RawData['gen'], '--')
#             ax.plot(i.RawData['time'], i.RawData['PC'], '.-')
#
#         ax.semilogx()
#         ax.semilogy()
#         ax.set_xlabel('time')
#         ax.set_ylabel('voltage')
#
#         return ax
#
#     @property
#     def attr(self):
#         return self.datas[0].wafer_inf
#
#     def adjust(self, var, percent):
#         dic = {var: self.datas[0].wafer_inf[var] * (1 + percent / 100.)}
#         self.set_attr(**dic)
#         self.calculate()
#
#     @attr.setter
#     def attr(self, kwargs):
#
#         for key, value in kwargs.items():
#             if key in self.datas[0].wafer_inf:
#                 for i in self.datas:
#                     i.wafer_inf[key] = value
#
#     def adjust(self, var, percent):
#         dic = {var: self.datas[0].wafer_inf[var] * (1 + percent / 100.)}
#         self.attr = dic
#         self.calculate()
#
#     def save(self):
#         for data in self.datas:
#             data.save()
#
#     def self_consistant_generation(self, index, data):
#         temp = dict(self.datas[index].wafer_inf)
#
#         self.datas[index].wafer_inf['cropStart'] = None
#         self.datas[index].wafer_inf['cropEnd'] = None
#         self.datas[index].wafer_inf['signal_cutoff'] = 100
#
#         self.datas[index].calculate_lifetime(
#             model_handeller=self.model_handeller)
#
#         value = self.datas[index].self_consistant_generation(data)
#         self.datas[index].wafer_inf = dict(temp)
#
#         return value
#
#     def self_consistant_PL(self, index):
#         temp = dict(self.datas[index].wafer_inf)
#
#         self.datas[index].wafer_inf['cropStart'] = None
#         self.datas[index].wafer_inf['cropEnd'] = None
#         self.datas[index].wafer_inf['signal_cutoff'] = 100
#
#         self.datas[index].calculate_lifetime(
#             model_handeller=self.model_handeller)
#
#         value = self.datas[index].self_consistant_PL(self.model_handeller)
#         self.datas[index].wafer_inf = dict(temp)
#
#         return value
#
#     def check_sample_properties(self):
#         for i in ['doping_type', 'doping', 'thickness', 'reflection']:
#             if self.attr[i] is None:
#                 var = input('Please enter the ' + i + ':')
#                 try:
#                     self.attr = {i: float(var)}
#                 except:
#                     self.attr = {i: var}


class Data():

    Derivitive = 'Finite Difference'

    def __init__(self, fname):
        load_measurements(fname)

    def save(self):
        '''
        Saves all the data
        '''

        data = np.vstack((
            self.nxc_pc,
            self.Tau_PC,
            self.nxc_pl,
            self.Tau_PL
        )).T

        # TODO: Fix the saving.

        np.savetxt(self.fname + 'something.dat', data)
        self.save_inf()

    def save_inf(self):
        '''
        Gets the data in a format to send for saving.
        Then sends to save.
        '''

        settings['sample'] = self.sample_inf
        settings['measurement'] = self.mmt_inf
        settings['analysis'] = self.anl_inf

        self.LoadData(self.fname).save(settings)

    def load_measurements(self, fname=None):
        '''
        Grab the data from the loader
        '''
        fname = fname or self.fname
        self.fname = fname

        self.RawData, settings = LoadData(fname).load()
        # this just prevents reloading of data
        self.sample_inf = settings['sample']
        self.mmt_inf = settings['measurement']
        self.anl_inf = settings['analysis']

    def choosing_defult_crop_values(self):

        _cropping_index(self.RawData, self.mmt_inf)

    def calculate_lifetime(self, BackGroundShow=False, model_handeller=None):
        '''
        Calculates the lifetimes
        '''

        # TODO: seperate out the PC and the PL lifetime calculation.
        # then make this class detect and return the lifetimes based on the data
        # in the provided columns

        # update the models
        model_handeller._update_update()
        # make sure the ni is updated
        # self.ni = CQ.ni(self.sample_inf['temp'], model_handeller)

        self.Na, self.Nd, self.sample_inf['dopant'] = background_doping(
            doping_type=self.sample_inf['doping_type'],
            doping=self.sample_inf['doping'],
            temp=self.sample_inf['temp']
        )
        # determine the background concentration of carriers
        self.ne0, self.nh0 = CQ.number_of_carriers(
            Na=Na, Nd=Nd, nxc=0, temp=self.sample_inf['temp'],
            model_handeller=model_handeller)

        # Background correction stuff
        self.Data, self.DarkConductance = _background_correction(
            self.RawData, self.mmt_inf)

        # crop the data
        self.ChoosingDefultCropValues()
        self.Data = crop_data(self.Data, self.anl_inf[
                              'cropStart'], self.anl_inf['cropEnd'])

        self.Data = bin_named_array(self.Data, self.anl_inf['binning'])

        self.nxc_pc = CQ.nxc_from_photoconductance(
            self.Data['PC'],
            self.sample_inf['thickness'],
            self.sample_inf['temp'],
            self.ne0,
            self.nh0,
            model_handeller)

        self.nxc_pl = CQ.nxc_from_photoluminescence(
            self.Data['PL'] / self.mmt_inf['gain_pl'],
            self.sample_inf['Ai'],
            self.sample_inf['dopant'],
            self.Na,
            self.Nd,
            self.sample_inf['temp'],
            model_handeller)

        gen_pc = CQ.generation(self.Data['time'],
                               self.Data['gen'] /
                               self.mmt_inf['gain_gen'],
                               self.nxc_pc, self.anl_inf['generation'], thickness=self.sample_inf[
                                   'thickness'],
                               reflection=self.sample_inf['reflection'],
                               Fs=self.mmt_inf['Fs'])

        gen_pl = CQ.generation(self.Data['time'],
                               self.Data['gen'] /
                               self.mmt_inf['gain_gen'],
                               self.nxc_pl, self.anl_inf['generation'], thickness=self.sample_inf[
                                   'thickness'],
                               reflection=self.sample_inf['reflection'],
                               Fs=self.mmt_inf['Fs'])

        self.Tau_PC = self.nxc_pc / gen_pc

        self.Tau_PL = self.nxc_pl / gen_pl

        return self.Tau_PC, self.Tau_PL

    def self_consistant_generation(self, data):
        nxc = getattr(self, 'nxc_' + data.lower())
        Fs = SC.generation(self.Data['time'],
                           self.Data['gen'] / self.mmt_inf['gain_gen'],
                           nxc,
                           self.wafer_inf)

        return Fs

    def self_consistant_PL(self, model_handeller):
        # nxc = getattr(self, 'nxc_' + 'pl')
        Ai = SC.photoluminescence(time=self.Data['time'],
                                  gen_norm=self.Data['gen'] /
                                  self.mmt_inf['gain_gen'],
                                  pl_norm=self.Data['PL'] /
                                  self.mmt_inf['gain_pl'],
                                  Ai=self.sample_inf['Ai'],
                                  thickness=self.sample_inf['thickness'],
                                  reflection=self.sample_inf['reflection'],
                                  Fs=self.mmt_inf['Fs'],
                                  Na=self.Na,
                                  Nd=self.Nd,
                                  dopant=self.sample_inf['dopant']
                                  temp=self.sample_inf['temp']
                                  model_handeller=model_handeller)

        return Ai


def crop_data(data, crop_start, crop_end):
    '''
     Crops the data based on provided values
    '''
    maxindex = np.amax(data.shape)

    index = np.arange(int(crop_start / 100 * maxindex),
                      int(crop_end / 100 * maxindex), 1)

    data = data[index]
    return data


def bin_named_array(data, no_pts_bnd):
    '''
    Binns a named around the provided amount
    '''

    # create a new array
    if len(data.dtype.names) != 1:
        # ensures that the array has the right number of values
        num = data.shape[0] // BinAmount
        data2 = copy(data)[0:num * BinAmount:BinAmount]

    # make sure that the lengths aren't too long
    assert data.shape[0] >= data2.shape[0] * BinAmount

    for i in data.dtype.names:
        for j in range(num):
            data2[i][j] = mean(
                data[i][j * BinAmount:(j + 1) * BinAmount])

    return data2


def _background_correction(data_in, measurement_inf, bkg_percent=0.95):
    '''
    Background corrects the data
    '''
    Data = np.copy(data_in)
    if 'bkh_pl' in measurement_inf:

        for name in data_in.dtype.names:
            if name in ['PL', 'gen']:
                Data[name] -= measurement_inf['bkg_' + name.lower()]

            elif name in ['PC']:
                Data[name] = CQ.pc_from_voltage(Data[name], measurement_inf['Quad'],
                                                measurement_inf['Lin'], measurement_inf['Const'])

                DarkConductance = CQ.pc_from_voltage(measurement_inf['bkg_' + name.lower()],
                                                     measurement_inf['Quad'],
                                                     measurement_inf['Lin'], measurement_inf['Const'])

                Data[name] -= DarkConductance

    else:
        bkg_index = int(data_in.shape[0] * bkg_percent)
        for name in data_in.dtype.names:
            if name in ['PL', 'gen']:
                Data[name] -= np.average(Data[name][bkg_index:])

            elif name in ['PC']:
                Data[name] = CQ.pc_from_voltage(Data[name], measurement_inf['Quad'],
                                                measurement_inf['Lin'], measurement_inf['Const'])

                DarkConductance = np.average(Data['PC'][bkg_index:])
                Data[name] -= DarkConductance

        return Data, DarkConductance


def _cropping_index(data, analysis_inf):
    '''
    Tries to find the cropping based on the noise in the PL channel
    '''
    if 'signal_cutoff' not in analysis_inf:
        analysis_inf['signal_cutoff'] = 10

    if analysis_inf['cropStart'] == None and analysis_inf['cropStart'] == None:
        if 'bkg_pl_std' in analysis_inf:
            index = data['PL'] > analysis_inf[
                'bkg_pl_std'] * analysis_inf['signal_cutoff']

            indexs = np.linspace(0, 1, data['PL'].shape[0])
            analysis_inf['cropStart'], analysis_inf[
                'cropEnd'] = indexs[index][[0, -1]] * 100
        else:

            analysis_inf['cropStart'], analysis_inf[
                'cropEnd'] = 5, 95
