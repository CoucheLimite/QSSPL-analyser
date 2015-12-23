

from numpy import *
import matplotlib.pylab as plt

from models.ConstantsClass import *

from models.MobilityV2 import Mobility_Klassen as Mobility
from models.NumericalDifferentiation_windows import Finite_Difference, Regularisation
from models.Recombination_v2_01 import Radiative

from utils.importexport import LoadData


def B(deltan, Nd):
    return Radiative().B(deltan, Nd)


def find_nearest(array, value):
    idx = (abs(array - value)).argmin()
    return idx


class Data(Constants):

    Derivitive = 'Regularised'
    Analysis = 'Generalised'
    Type = 'p'
    CropStart = 0
    CropEnd = 100
    BackgroundSubtraction = 0.95
    Used = False

    def __init__(self):
        self.LD = LoadData()

    def BackgrounConcentration(self):
        if (self.Type == 'p'):
            self.p0 = self.Wafer['Doping']
            self.n0 = self.ni**2 / self.Wafer['Doping']
            # print 'ptype'
        elif(self.Type == 'n'):
            self.n0 = self.Wafer['Doping']
            self.p0 = self.ni**2 / self.Wafer['Doping']
            # print 'ntype'
        else:
            self.n0 = 1e20
            self.p0 = 1e20

    def ProvideRawDataFile(self, Directory, RawDataFile):
        self.Directory = Directory + '/'
        self.RawDataFile = RawDataFile

        self.Used = True

        self.Load_Measurements()

    def UpdateInfData(self):

        self.LD.WriteTo_Inf_File(self.Wafer)

    def dndt(self, Deltan):

        if (self.Derivitive == 'Regularised'):
            dn_dt = Regularisation().FirstDerivative(
                self.Data['Time'], Deltan, 1e-20)

        elif (self.Derivitive == 'Finite Difference'):
            dn_dt = Finite_Difference().FourPointCentral(
                self.Data['Time'], Deltan)

        else:
            print 'You fucked up.... again'

        return dn_dt

    def Load_Measurements(self):

        self.LD.Directory = self.Directory
        self.LD.RawDataFile = self.RawDataFile

        self.RawData = self.LD.Load_RawData_File()
        self.Wafer = self.LD.Load_InfData_File()

    def ChoosingDefultCropValues(self):

        Waveform = self.Wafer['Waveform']

        if Waveform == 'Triangle':
            self.Wafer['CropStart'], self.Wafer['CropEnd'] = 35, 55
        elif Waveform == 'Square':
            self.Wafer['CropStart'], self.Wafer['CropEnd'] = 13, 50
        elif Waveform == 'Sawtooth':
            self.Wafer['CropStart'], self.Wafer['CropEnd'] = 12, 79

    def iVoc(self):
        return self.Vt * log((self.n0 + self.DeltaN_PC) * (self.p0 + self.DeltaN_PC) / self.ni / self.ni), self.Vt * log((self.DeltaN_PL + self.n0) * (self.DeltaN_PL + self.p0) / self.ni / self.ni)

    def CalculateLifetime(self, BackGroundShow=False):
        """changing Votlage to conductance"""
        self.BackgrounConcentration()

        BackgroundIndex = int(self.RawData['Time'].shape[0] * .95)
        self.Data = copy(self.RawData)

        for i in ['PL', 'Gen']:
            self.Data[i] -= average(self.Data[i][BackgroundIndex:])

        self.Data['PC'] = self.Quad * \
            (self.Data['PC']) * (self.Data['PC']) + \
            self.Lin * (self.Data['PC']) + self.Const

        '''for now just use the 95% limit to cal background'''
        """background subtraction"""
        self.DarkConductance = average(self.Data['PC'][BackgroundIndex:])
        for i in ['PC']:

            self.Data[i] -= average(self.Data[i][BackgroundIndex:])

        self.Cropping_Percentage()

        """For checking background subtraction"""
        if BackGroundShow == True:
            fig = plt.figure('BackGround Check')
            plt.title('BackGround Check')
            for i in ['PC', 'PL', 'Gen']:
                plt.plot(self.Data['Time'], self.Data[i], label=i)
            plt.xlim(0, max(self.Raw_Time))
            plt.legend(loc=0)
            plt.ylim(-1, 11)
            plt.show()

        self.Data = self.Binning_Named(self.Data, self.Wafer['Binning'])

        self.DeltaN_PC = ones(self.Data['Time'].shape[0]) * 1e10
        self.DeltaN_PL = ones(self.Data['Time'].shape[0]) * 1e10

        while (i > 0.001):

            temp = self.Data[
                'PC'] / self.q / Mobility().mobility_sum(self.DeltaN_PC, self.n0, self.p0, self.Wafer['Temp']) / self.Wafer['Thickness']
            i = average(abs(temp - self.DeltaN_PC) / self.DeltaN_PC)

            self.DeltaN_PC = temp

        for i in range(5):
            self.DeltaN_PL = (-self.Wafer['Doping'] + sqrt(abs(self.Wafer['Doping']**2 + 4 * self.Data[
                              'PL'] * self.Wafer['Ai'] / B(self.DeltaN_PL, self.Wafer['Doping'])))) / 2

        self.Tau_PC = self.DeltaN_PC / self.Generation('PC')
        self.Tau_PL = self.DeltaN_PL / self.Generation('PL')

    def Generation(self, PCorPL, suns=False):
        try:
            Gen = getattr(
                self, 'Generation_' + self.Analysis.replace(' ', '_'))
        except:
            print 'Choice of generation doesn\'t exist: You fucked up'

        if suns == True:
            scale = self.Wafer['Thickness'] / 2.5e17
        else:
            scale = 1.

        return Gen(PCorPL) * scale

    def Generation_Steady_State(self, PCorPL):
        Trans = (1 - self.Wafer['Reflection'] / 100.)
        return self.Data['Gen'] * self.Wafer['Fs'] * Trans / self.Wafer['Thickness']

    def Generation_Generalised(self, PCorPL):
        Trans = (1 - self.Wafer['Reflection'] / 100.)
        if PCorPL == 'PC':
            return self.Data['Gen'] * self.Wafer['Fs'] * Trans / self.Wafer['Thickness'] - self.dndt(self.DeltaN_PC)
        elif PCorPL == 'PL':
            return self.Data['Gen'] * self.Wafer['Fs'] * Trans / self.Wafer['Thickness'] - self.dndt(self.DeltaN_PL)
        else:
            print 'You fucked up the Generation'

    def Generation_Transient(self, PCorPL):
        if PCorPL == 'PC':
            return -self.dndt(self.DeltaN_PC)
        elif PCorPL == 'PL':
            return -self.dndt(self.DeltaN_PL)
        else:
            print 'You fucked up the Generation'

    def Local_IdealityFactor(self):
        # Generation scale doesn't matter so Generation is used
        # print 'in'
        iVocPC, iVocPL = self.iVoc()

        # print iVocPC
        # print (self.SS_Generation-self.dndt(self.DeltaN_PC)),Regularisation().FirstDerivative(iVocPC,self.SS_Generation-self.dndt(self.DeltaN_PC),1e-20),Finite_Difference().FourPointCentral(iVocPC,self.SS_Generation-self.dndt(self.DeltaN_PC)),(self.SS_Generation-self.dndt(self.DeltaN_PC))
        if (self.Derivitive == 'Regularised'):
            return  (self.Generation('PC')) / (self.Vt * Regularisation().FirstDerivative       (self.Data['Time'], self.Generation('PC'), 1e-20) / Regularisation().FirstDerivative  (self.Data['Time'], iVocPC, 1e-20)),\
                    (self.Generation('PL')) / (self.Vt * Regularisation().FirstDerivative(self.Data['Time'], self.Generation(
                        'PL'), 1e-20) / Regularisation().FirstDerivative(self.Data['Time'], iVocPL, 1e-20))

        elif (self.Derivitive == 'Finite Difference'):
            return  (self.Generation('PC')) / (self.Vt * Finite_Difference().FourPointCentral   (self.Data['Time'], self.Generation('PC'))     / Finite_Difference().FourPointCentral(self.Data['Time'], iVocPC)),\
                    (self.Generation('PL')) / (self.Vt * Finite_Difference().FourPointCentral(self.Data[
                        'Time'], self.Generation('PL')) / Finite_Difference().FourPointCentral(self.Data['Time'], iVocPL))

    def Cropping_negitives(self):
        # this just uses that when points are negitive they should no longer be
        # used
        maxindex = argmax(self.SS_Generation)

        # print self.DeltaN_PL.shape
        # print 'Cropping Negitives'

        IndexOfNegitives = where(self.SS_Generation < 0)[0]

        firstnegtive = where(IndexOfNegitives < maxindex)[0][-1]
        lastnegtive = where(IndexOfNegitives > maxindex)[0][0]
        self.index = arange(
            IndexOfNegitives[firstnegtive], IndexOfNegitives[lastnegtive], 1)
        # plot(self.Time,self.DeltaN_PL)
        # print self.Time[IndexOfNegitives[firstnegtive]],self.Time[maxindex],self.Time[IndexOfNegitives[lastnegtive]]
        # show()

        # print
        # IndexOfNegitives,maxindex,self.DeltaN_PL[IndexOfNegitives[firstnegtive]]

        #self.DeltaN_PL = self.DeltaN_PL [ self.index]
        #self.DeltaN_PC = self.DeltaN_PC [ self.index]
        self.SS_Generation = self.SS_Generation[self.index]
        self.Time = self.Time[self.index]

        self.RawPCDataEdited = self.RawPCDataEdited[self.index]
        self.Raw_PLEdited = self.Raw_PLEdited[self.index]

    def Cropping_Percentage(self):
        # this just uses that when points are negitive they should no longer be used
        # print 'Cropping Percentage'
        maxindex = self.Data.shape[-1]

        self.index = arange(int(self.Wafer[
                            'CropStart'] / 100 * maxindex), int(self.Wafer['CropEnd'] / 100 * maxindex), 1)

        #self.DeltaN_PL = self.DeltaN_PL [ self.index]
        # self.DeltaN_PC = self.DeltaN_PC [ self.index]
        # self.SS_Generation = self.SS_Generation [ self.index]
        # self.Time = self.Time [ self.index]
        self.Data = self.Data[self.index]
        # self.RawPCDataEdited = self.RawPCDataEdited [ self.index]
        # self.Raw_PLEdited = self.Raw_PLEdited [ self.index]

    def EQE(self):

        return self.RawPCDataEdited / self.Generation('PC'), self.Raw_PLEdited / self.Generation('PL') * self.Ai

    def IQE_SingleIntensity(self):

        idx = find_nearest(self.Generation('PL'), 2.5e17 / self.Thickness / 10)
        return self.RawPCDataEdited[idx] / self.Generation('PC')[idx], self.Raw_PLEdited[idx] / self.Generation('PL')[idx] * self.Ai
