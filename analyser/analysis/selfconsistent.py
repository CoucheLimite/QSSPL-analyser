

import numpy as np
import scipy.constants as C
from scipy import optimize as op
import calculate_quantities as CQ


def photoluminescence(time, gen_norm, pl_norm, Ai, thickness, reflection, Fs, Na, Nd, dopant, temp, model_handeller):
    '''
    Performs self consutent analsys to determine the generation rate.
    by updating Fs
    '''
    gen = CQ.generation(time, gen_V, 0, 'ss', thickness, reflection, Fs)
    index = np.argmax(gen)

    def _SC(_Ai):

        _Ai = abs(_Ai)

        nxc = CQ.nxc_from_photoluminescence(
            pl_norm,
            _Ai,
            dopant,
            Na,
            Nd,
            temp,
            model_handeller)

        gen = generation(time, gen_V, nxc, 'generalised',
                         thickness, reflection, _Fs)

        tau = nxc / gen

        nxc0 = nxc[:index]
        nxc1 = nxc[index:]

        tau0 = tau[:index]
        tau1 = tau[index:]

        if np.amin(nxc0) < np.amin(nxc1):
            tau0 = np.interp(nxc1, nxc0, tau0)
        else:
            tau1 = np.interp(nxc0, nxc1[::-1], tau1[::-1])

        return np.sum(np.abs((tau1 - tau0) / tau0)) * 10

    res = op.minimize_scalar(
        _SC,
        bracket=(Ai / 10, Ai * 10),
        method='brent')

    return abs(res.x)


def generation(time, gen_V, nxc, thickness, reflection, Fs):
    '''
    Performs self consutent analsys to determine the generation rate.
    by updating Fs
    '''

    gen = generation(time, gen_V, nxc, 'ss', thickness, reflection, Fs)

    index = np.argmax(gen)
    nxc0 = nxc[:index]
    nxc1 = nxc[index:]

    def _SC(_Fs):

        _Fs = abs(_Fs)

        gen = generation(time, gen_V, nxc, 'generalised',
                         thickness, reflection, _Fs)

        tau = nxc / gen
        tau0 = tau[:index]
        tau1 = tau[index:]

        if np.amin(nxc0) < np.amin(nxc1):
            tau0 = np.interp(nxc1, nxc0, tau0)
        else:
            tau1 = np.interp(nxc0, nxc1[::-1], tau1[::-1])

        return np.sum(np.abs((tau1 - tau0) / tau0)) * 10

    res = op.minimize_scalar(
        _SC,
        bracket=(Fs / 100, Fs * 100),
        method='Golden')

    return abs(res.x)
