
import numpy as np
import scipy.constants as C

from semiconductor.electrical.mobility import Mobility
from semiconductor.electrical.ionisation import Ionisation as Ion
from semiconductor.matterial.ni import IntrinsicCarrierDensity as NI
from semiconductor.recombination.Intrinsic import Radiative


def min_car_den_from_photoconductance(conductance,
                                        wafer_thickness,
                                        wafer_temp,
                                        ne0,
                                        nh0,
                                        mobility_model='klaassen1992'):
    '''
    Calculates the excess carrier density per cm^-3 from a photoconductance
    '''
    print mobility_model
    #
    min_car_den = np.ones(conductance.shape[0]) * 1e10
    Na, Nd = ne0, nh0

    # as the mobility is a function of carrier density,
    # this needs to be iterated a few times (usually less than 5 times)
    error = 1
    while (error > 0.01):

        # TO DO
        # the below line takes is provide n0 and p0 and not the doping of
        # each. This needs to be fixed.
        # assumption: mobility only matters on the ionised dopants

        # iNa = Ion('Si').update_dopant_ionisation(Na, self.DeltaN_PC, 'phosphorous',
        #                      temp=self.Wafer['Temp'], author=None)
        # iNd = Ion('Si').update_dopant_ionisation(Nd, self.DeltaN_PC, 'boron',
        #                      temp=self.Wafer['Temp'], author=None)

        # current just on the number of dopants
        iNa = Na
        iNd = Nd

        temp = conductance / C.e / wafer_thickness\
            / Mobility('Si',
                       author=mobility_model
                       ).mobility_sum(
                min_car_den=min_car_den,
                Na=iNa, Nd=iNd,
                temp=wafer_temp)

        error = np.average(np.absolute(temp - min_car_den) / min_car_den)

        min_car_den = temp

    return min_car_den


def min_car_den_from_photoluminescence(photoluminescence,
                                         Ai,
                                         dopant,
                                         net_dopants,
                                         wafer_temp,
                                         B_model='Altermatt2005',
                                         ionisation_model='Altermatt2006_table1'):
    '''
    Calculates the excess carrier density per cm^-3 from a photoluminescence data
    '''

    #
    min_car_den = np.ones(photoluminescence.shape[0]) * 1e10
    i = 1
    while (i > 0.01):

        idop = Ion('Si',
                   author=ionisation_model
                   ).update_dopant_ionisation(
            net_dopants, min_car_den, dopant,
            temp=wafer_temp)

        maj_car_den = idop + min_car_den

        # TODO
        B = Radiative('Si',
                      author=B_model).B(
            min_car_den, idop, temp=wafer_temp)

        temp = (-maj_car_den +
                np.sqrt(np.absolute(
                    (maj_car_den)**2 + 4 * photoluminescence * Ai / B))) / 2

        i = np.average(np.absolute(temp - min_car_den) / min_car_den)
        min_car_den = temp

    return min_car_den


def iVoc_from_carriers(ne0, nh0, min_car_den, temp, ni):
    '''
    calculates the implied voltage from the number of carriers 
    '''
    ne = ne0 + min_car_den
    nh = nh0+ min_car_den
    return C.k*temp/C.e * np.log(ne*nh / np.power(ni,2))