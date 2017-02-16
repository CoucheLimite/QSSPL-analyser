
import numpy as np
import scipy.constants as C


def pc_from_voltage(V, a, b, c):
    return a * V * V + b * V + c


def doping_from_pc(darkconductance, wafer_inf, model_handeller):
    '''
    This doesn't work as we don't know the air value
    '''
    if 'air_conductance' in wafer_inf:
        wafer_inf['doping_pc'] = 1e16
        for i in range(5):
            if wafer_inf['doping_type'] == 'p':
                Na = wafer_inf['doping_pc']
                Nd = 0
            else:
                Nd = wafer_inf['doping_pc']
                Na = 0

            wafer_inf['doping_pc'] = darkconductance / C.e / model_handeller.update['mobility'](
                nxc=0,
                Na=Na, Nd=Nd,
                temp=wafer_inf['Temp'])


def nxc_from_photoconductance(conductance,
                              wafer_thickness,
                              wafer_temp,
                              ne0,
                              nh0,
                              model_handeller):
    '''
    Calculates the excess carrier density per cm^-3 from a photoconductance
    '''

    #
    nxc = np.ones(conductance.shape[0]) * 1e10
    Na, Nd = ne0, nh0

    error = 1
    while (error > 0.01):
        # assumption: mobility only matters on the ionised dopants

        # iNa = Ion('Si').update_dopant_ionisation(Na, self.DeltaN_PC, 'phosphorous',
        #                      temp=self.Wafer['Temp'], author=None)
        # iNd = Ion('Si').update_dopant_ionisation(Nd, self.DeltaN_PC, 'boron',
        #                      temp=self.Wafer['Temp'], author=None)

        # current just on the number of dopants

        iNa = model_handeller.update['ionisation'](
            N_dop=Na, nxc=nxc, impurity='boron',
            temp=wafer_temp)

        iNd = model_handeller.update['ionisation'](
            N_dop=Nd, nxc=nxc, impurity='phosphorous',
            temp=wafer_temp)

        temp = conductance / C.e / wafer_thickness\
            / model_handeller.update['mobility'](
                nxc=nxc,
                Na=iNa, Nd=iNd,
                temp=wafer_temp)

        error = np.average(np.absolute(temp - nxc) / nxc)

        nxc = temp

    return nxc


def nxc_from_photoluminescence(photoluminescence,
                               Ai,
                               dopant,
                               Na,
                               Nd,
                               wafer_temp,
                               model_handeller):
    '''
    Calculates the excess carrier density per cm^-3 from a photoluminescence data
    '''

    if np.all(photoluminescence == 0):
        return photoluminescence
    else:
        #
        nxc = np.ones(photoluminescence.shape[0]) * 1e10
        i = 1
        while (i > 0.001):

            idop = model_handeller.update['ionisation'](
                N_dop=abs(Na - Nd), nxc=nxc, impurity=dopant,
                temp=wafer_temp)

            B = model_handeller.update['B'](
                nxc=nxc, Na=Na, Nd=Nd, temp=wafer_temp)

            temp = (-idop +
                    np.sqrt(np.absolute(
                        (idop)**2 + 4 * photoluminescence * Ai / B))) / 2

            i = np.average(np.absolute(temp - nxc) / nxc)
            nxc = temp

        return nxc


def iVoc_from_carriers(ne0, nh0, nxc, temp, ni):
    '''
    calculates the implied voltage from the number of carriers
    '''
    ne = ne0 + nxc
    nh = nh0 + nxc
    return C.k * temp / C.e * np.log(ne * nh / np.power(ni, 2))
