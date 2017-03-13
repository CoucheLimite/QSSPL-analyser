
import numpy as np
import scipy.constants as C
from .NumericalDifferentiation_windows import FourPointCentral
import semiconductor.electrical as el


def background_doping(doping_type, doping, temp):
    '''
    Determines the number acceptors and donars in the sample
        inputs:
            doping_type: (str)
                'p' or 'n'
            doping: (float)
                The number of substitutional dopants per cm^3
            temp: (float)
                The temperature in Kelvin
    output:
        The number of substitutional acceptor and donar atoms per cm^3 and the dopant.
    '''

    if (doping_type == 'p'):
        Na = doping
        Nd = 0
        dopant = 'boron'

    elif(doping_type == 'n'):
        Nd = doping
        Na = 0
        dopant = 'phosphorous'

    return Na, Nd, dopant


def number_of_carriers(Na, Nd, nxc, temp, model_handeller):
    '''
    Determines the number of carriers in the sample
        inputs:
            Na: (float)
                The number of substitutional acceptor atoms per cm^3
            Na: (float)
                The number of substitutional donar atoms  per cm^3
            nxc: (array like)
                The number of excess carriers per cm^3
            temp: (float)
                The temperature in Kelvin
            model_handeller: (object)
                A instance of models.models.models_handeller
    output:
        the number of electrons and holes
    '''
    ne0, nh0 = el.get_carriers(Na=Na,
                               Nd=Nd,
                               nxc=nxc,
                               temp=temp,
                               ni_author=model_handeller.selected_model[
                                   'ni'],
                               ionisation_author=model_handeller.selected_model[

    return ne0, nh0

def pc_from_voltage(V, quad, lin, constant):
    '''
    Tranforms measured PC voltage into conductance

    inputs:
        voltage: (array)
            voltage meaured on the photoconductance coil in volts
        quad: (float)
            The quadratic coefficient for the coil
        lin: (float)
            The linear coefficient for the coil
        constant: (float)
            The linear coefficient for the coil
    output:
        the conductance in semens
    '''
    return quad * V * V + lin * V + constant


def doping_from_pc(darkconductance, dark_voltage, temp, quad, lin, constant, doping_type, model_handeller):
    '''
    Calculates the doping from a dark conductance meaurement

        inputs:
            darkconductance: (float)
                The conductance of the sample in the dark on the coil
            dark_voltage: (float)
                The voltage when no sample was on the coil
            temp: (float)
                The temperature in Kelvin
            quad: (float)
                The quadratic coefficient for the coil
            lin: (float)
                The linear coefficient for the coil
            constant: (float)
                The linear coefficient for the coil
            doping_type: (str)
                'p' or 'n'
            model_handeller: (object)
                A instance of models.models.models_handeller

        output:
            The doping of the wafer
    '''

    DC=darkconductance - pc_from_voltage(dark_voltage,
                                           quad,
                                           lin,
                                           const)

    # assume a doping
    doping_DC=1e16

    # get a mobility class ready.
    mob=Mob(material='Si',
            author=model_handeller.selected_model['mobility'],
            temp=temp)


    for i in range(5):

        # get the number of dopants
        Na, Nd, dopant=background_doping(doping_type, doping_DC, temp)
        # then the number of carriers in the dark
        ne0, nh0=number_of_carriers(Na, Nd, 0, temp, model_handeller)

        # calculate the mobility
        mob_e=mob.electron_mobility(nxc=0,
                                           Na=Na,
                                           Nd=Nd
                                           )
        mob_h=mob.hole_mobility(nxc=0,
                                       Na=Na,
                                       Nd=Nd)

        # The conductivity is
        # cond=C.e * (mob_e * ne + mob_h * nh)
        # so rarranging provides
        if dopant == 'boron':
            doping_DC=(DC / C.e - mob_e * ne0) / mob_h
        elif dopant == 'phosphorous':
            doping_DC=(DC / C.e - mob_h * nh0) / mob_h

    return doping_DC



def nxc_from_photoconductance(conductance,
                              thickness,
                              temp,
                              dopant,
                              Ndop,
                              model_handeller):
    '''
    Calculates the excess carrier density per cm^-3 from a photoconductance
    '''

    nxc=np.ones(conductance.shape[0]) * 1e10

    error=1
    while (error > 0.01):
        # assumption: no co-doping

        if dopant == 'boron':
            iNa=model_handeller.update['ionisation'](
                N_dop=Ndop, nxc=nxc, impurity='boron',
                temp=temp)

            iNd=0

        elif dopant == 'phosphorous':
            iNd=model_handeller.update['ionisation'](
                N_dop=Ndop, nxc=nxc, impurity='phosphorous',
                temp=temp)

            iNa=0

        temp=conductance / C.e / thickness\
            / model_handeller.update['mobility'](
                nxc=nxc,
                Na=iNa, Nd=iNd,
                temp=temp)

        error=np.average(np.absolute(temp - nxc) / nxc)

        nxc=temp

    return nxc


def nxc_from_photoluminescence(photoluminescence,
                               Ai,
                               dopant,
                               Na,
                               Nd,
                               temp,
                               model_handeller):
    '''
    Calculates the excess carrier density per cm^-3 from a photoluminescence data
    '''

    if np.all(photoluminescence == 0):
        return photoluminescence
    else:
        #
        nxc=np.ones(photoluminescence.shape[0]) * 1e10
        i=1
        while (i > 0.001):

            idop=model_handeller.update['ionisation'](
                N_dop=abs(Na - Nd), nxc=nxc, impurity=dopant,
                temp=temp)

            B=model_handeller.update['B'](
                nxc=nxc, Na=Na, Nd=Nd, temp=temp)

            _temp=(-idop +
                    np.sqrt(np.absolute(
                        (idop)**2 + 4 * photoluminescence * Ai / B))) / 2

            i=np.average(np.absolute(_temp - nxc) / nxc)
            nxc=_temp

        return nxc

def ni(temp, model_handeller):
    '''
    Calculates the intrinsic carrier density given a temperature

    inputs:
        temp:(float)
            The temperature in Kelvin
        model_handeller: (object)
                A instance of models.models.models_handeller
    '''
        ni=model_handeller.update[
            'ni'](temp=temp)
    return ni


def iVoc_from_carriers(ne0, nh0, nxc, temp, ni):
    '''
    calculates the implied voltage from the number of carriers
    '''
    ne=ne0 + nxc
    nh=nh0 + nxc
    return C.k * temp / C.e * np.log(ne * nh / np.power(ni, 2))


def generation(time, gen_norm, nxc, analysis_type, thickness=None, reflection=None, Fs=None):

    function={
        'generalised': _generation_generalised,
        'ss': _generation_steady_state,
        'trans': _generation_transient,
    }

    assert analysis_type in function
    if anal_type == 'trans':
        gen=function[anal_type](time, gen_norm, nxc)
    else:
        gen=function[anal_type](
            time, gen_norm, nxc, thickness, reflection, Fs)

    return gen


def _generation_steady_state(time, gen_norm, nxc, thickness, reflection, Fs):
    transmission=(1 - wafer_inf['reflection'] / 100.)
    return gen_V * wafer_inf['Fs'] * transmission / wafer_inf['thickness']


def _generation_generalised(time, gen_norm, nxc, thickness, reflection, Fs):

    gen=_generation_steady_state(time, gen_V, nxc, thickness, reflection, Fs
                                   ) + _generation_transient(time, gen_V, nxc)

    return gen


def _generation_transient(time, gen_norm, nxc, **args):
    gen=-FourPointCentral(time,
                            nxc)
    return gen
