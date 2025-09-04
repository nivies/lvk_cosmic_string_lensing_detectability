from utils.library import *
def inject_noise_to_signal(signal, psd, duration=128, seed=None):
    """
    Adds gaussian noise to a given signal using a given PSD.

    Parameters
    ----------
    signal : pycbc.timeseries.BaseTimeseries
        Input timeseries.
    psd : callable
        Power spectral density to generate noise.
    duration : float, optional
        Duration of the noise signal in seconds, default is 128.
    seed : int, optional
        Random seed for the noise realisation, default is None.

    Returns
    -------
    noise_signal : pycbc.timeseries.BaseTimeseries
        Timeseries with added noise of duration 8 seconds.
    """

    delta_t = 1.0 / signal.sample_rate
    tsamples = int(duration / delta_t)

    ts = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=seed)
    ts.start_time += signal.end_time - duration / 2
    noise_signal = ts.add_into(signal)

    return noise_signal.time_slice(signal.end_time - 8, signal.end_time + 0)

def read_parfile(filepath):
    """
    Reads a parameter file at the given filepath and returns a dictionary of parameters.

    Parameters:
    filepath : str
        The path to the parameter file to be read.

    Returns:
    dict
        A dictionary containing the parameters read from the file.
    """
    params = {}
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                key, value = line.split('=')
                params[key.strip()] = eval(value.strip())
    return params

def pl_wave(R):
    """
    Plane wave function exp(-1j * R).

    Parameters
    ----------
    R : float
        Argument of the plane wave.

    Returns
    -------
    plane_wave : complex
        Complex value of the plane wave.
    """
    return np.exp(-1j * R)


def phase_delay(f, tM, y):
    """
    Phase delay of the wave due to a time delay and a gravitational lensing angle.

    Parameters
    ----------
    f : float
        Frequency of the wave.
    tM : float
        Time delay.
    y : float
        Gravitational lensing angle normalized by the angular diameter of the lens.

    Returns
    -------
    phase_delay : float
        Phase delay of the wave.
    """
    return np.pi * f * tM * (1 + y)**2


def diffr_coef(y):
    """
    Diffraction coefficient for GTD, draft:Eq.(12).

    Parameters
    ----------
    y : float
        Gravitational lensing angle normalized by the angular Diameter of the lens.

    Returns
    -------
    diffraction_coeff : complex
        Diffraction coefficient.
    """
    return -np.exp(1j * np.pi / 4) / (np.pi * (1 - y**2))


def argF(f, tM, y):
    """
    Argument in the Fresnel function, draft:Eq.(3).

    Parameters
    ----------
    f : float
        Frequency of the wave.
    tM : float
        Time delay.
    y : float
        Gravitational lensing angle normalized by the angular Diameter of the lens.

    Returns
    -------
    arg_F : float
        Argument of the Fresnel function.
    """
    tdd = phase_delay(f, tM ,y)
    return np.sign(1 + y) * np.sqrt(tdd)


def Fresnel_funct(x):
    """
    Fresnel integral, normalized to 1 at infinity, draft:Eq.(7).

    Parameters
    ----------
    x : float
        Argument of the Fresnel integral.

    Returns
    -------
    fresnel_integral : complex
        Complex value of the Fresnel integral.
    """
    ss, cc = fresnel(np.sqrt(2 / np.pi) * x)
    return 1 / np.sqrt(2j) * (1j * ss + cc + (1 + 1j) / 2)


def Sommerf(f, tM, y):
    """
    Full-wave Sommerfeld function, draft:Eq.(8).

    Parameters
    ----------
    f : float
        Frequency of the wave.
    tM : float
        Time delay.
    y : float
        Gravitational lensing angle normalized by the angular Diameter of the lens.

    Returns
    -------
    F : complex
        Complex value of the full-wave Sommerfeld function.
    """
    sp = phase_delay(f, tM, y)
    sm = phase_delay(f, tM, -y)
    wp = argF(f, tM, y)
    wm = argF(f, tM, -y)
    return pl_wave(sp) * Fresnel_funct(wp) + pl_wave(sm) * Fresnel_funct(wm)


def go(f, tM, y):
    """
    Geometrical optics (GO), draft:Eq.(11), 1st + 2nd terms.

    Parameters
    ----------
    f : float
        Frequency of the wave.
    tM : float
        Time delay.
    y : float
        Gravitational lensing angle normalized by the angular Diameter of the lens.

    Returns
    -------
    GOp : complex
        First and second terms of the geometrical optics contribution.
    """
    sp = phase_delay(f, tM, y)
    sm = phase_delay(f, tM, -y)
    GOp = pl_wave(sp) * 0.5 * (np.sign(1 + y) + 1)
    GOm = pl_wave(sm) * 0.5 * (np.sign(1 - y) + 1)
    return GOp + GOm


def gtd(f, tM, y):
    """
    Geometrical optics (GO) + diffraction, draft:Eq.(12).

    Parameters
    ----------
    f : float
        Frequency of the wave.
    tM : float
        Time delay.
    y : float
        Gravitational lensing angle normalized by the angular Diameter of the lens.

    Returns
    -------
    GOp : complex
        Geometrical optics + diffraction contribution.
    """
    return go(f, tM, y) + diffr_coef(y) / np.sqrt(f * tM)


''' 
PML transmission factor:
'''
from scipy.special import gamma,fresnel
import mpmath as mpm

def go_factor(f, tM, y):

    # Geometric optics approximation of the lensing factor
    v = y / np.sqrt(y*y+4)
    tau21 = 2.*v/(1-v*v) + np.log((1+v)/(1-v))
    mu1 = 0.25*(v + 1./v + 2.)
    mu2 = 0.25*(v + 1./v - 2.)
    smu1 = np.sqrt(mu1)
    smu2 = np.sqrt(mu2)
    x1_go = 0.5*(y+np.sqrt(y*y+4))
    tau_x1 = 0.5*(x1_go-y)**2 - np.log(np.abs(x1_go))
    eph1 = np.exp(-1j*2*np.pi*f*tM*tau_x1)
    # for phase, include psi_0? see article
    factor = eph1*( smu1 + smu2 * np.exp( -1j*2*(np.pi*f*tM*tau21 - np.pi/4.) ) )
    return factor

def full_factor(f, tM, y):

    # Full wave optics lensing factor (confluent hypergeometric form)
    ww = np.pi*f*tM
    hyp = np.array([complex(mpm.hyp1f1(-1j*wi, 1, -1j*wi*y**2)) for wi in ww])
    fact = np.exp(np.pi*ww/2)*gamma(1.0 + 1j*ww)*hyp*ww**(-1j*ww)
    # with phase, include second exponential? see article
    return fact

def hybrid_factor(f, tM, y):

    # Hybrid lensing factor combining wave and geometric optics
    v = y / np.sqrt(y*y+4)
    tau21 = 2.*v/(1-v*v) + np.log((1+v)/(1-v))
    #idx = np.searchsorted(f, 0.5/tau21/tM) # nu_G
    idx = np.searchsorted(f, 2.25/tau21/tM) # match at 2nd interference max
    factor1 = full_factor(f[:idx], tM, y)
    factor2 = go_factor(f[idx:], tM, y)
    return np.concatenate((factor1, factor2))