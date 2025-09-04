import numpy as np
import h5py
from utils.lensing_func import Sommerf
from pycbc.types import FrequencySeries
from pycbc.noise import noise_from_psd
from pycbc.filter import highpass, lowpass
from pycbc.waveform import get_fd_waveform
from tqdm import tqdm

def get_template_and_injection(
    y_array,
    tD_array,
    injection_parameters = dict(
        mass1=26, mass2=19, spin1z=0.4, spin2z=0.3,
        distance=2000.0, phase=1.3, ra=1.375, dec=-1.2108)
    ):

    # Simulation parameters
    print("Defining wave parameters...", end='\r')
    seed = 42
    yscale = "log" # linear or log scale for impact parameter sampling
    # ny, nd = 6, 6 # 30, 30 # Sampling of the lensing parameter space
    # N_template = 600

    # Lensing parameters sampling

    # Signal discretization parameters
    f_low = 20.0 # Lower frequency cutoff
    sample_rate = 4096 
    delta_t = 1.0 / sample_rate
    n_samples = int(10 / delta_t) # Introducing signal duration about 10 seconds
    signal_duration = n_samples * delta_t # Correcting signal duration due to discretization
    delta_f = 1.0 / signal_duration

    print("Generating fd waveform and PSD for injection...", end = "\r")
    approximant = "IMRPhenomXPHM"

    asd = np.loadtxt("../aLIGO_O4high.txt")
    f_max = asd[-1, 0]
    asd = np.interp(np.arange(0, f_max, delta_f), asd[:, 0], asd[:, 1])
    psd = FrequencySeries(asd**2, delta_f = delta_f)

    signal_f, _ = get_fd_waveform(
        approximant=approximant, delta_f=delta_f, f_lower=f_low, **injection_parameters
    )

    # Extension of higher frequency bins into test signal for matching of frequencies between injection and PSD.
    signal_f = FrequencySeries(
        np.append(signal_f.numpy(), np.zeros(len(psd) - len(signal_f))),
        delta_f=delta_f
    )

    injections = []

    print("Injecting signals...", end = "\r")
    for _y in y_array:
        for _td in tD_array:
            
            # Injecting the signal into noise without lensing
            lensed_signal_f = FrequencySeries(
                Sommerf(f = signal_f.sample_frequencies.data, tM = _td, y = _y) * signal_f.numpy(), 
                delta_f = delta_f
                )

            signal_t = lensed_signal_f.to_timeseries()

            ts = noise_from_psd(len(signal_t), signal_t.delta_t, psd, seed)

            signal_t = signal_t.cyclic_time_shift(signal_t.duration/2)
            signal_t.start_time = ts.start_time

            ts = ts.add_into(signal_t)
            ts = highpass(ts, 25)
            ts = lowpass(ts, 2048)
            ts = ts.crop(1, 1)

            signal_t = highpass(signal_t, 25)
            signal_t = lowpass(signal_t, 2048)
            signal_t = signal_t.crop(1, 1)

            filter = signal_t.to_frequencyseries()
            injection_f = ts.to_frequencyseries()

            injections.append((injection_f, filter))

    print("Recalculating signal parameters after injection...", end="\r")
    # PSD recalculation
    delta_f = injections[0][0].delta_f
    freqs = injections[0][0].sample_frequencies.data
    psd_matched_filtering = FrequencySeries(np.interp(freqs, psd.sample_frequencies.numpy(), psd.numpy()), delta_f = delta_f)

    og_signal_f = FrequencySeries(np.interp(freqs, psd.sample_frequencies.numpy(), signal_f.numpy()), delta_f = delta_f)

    return injections, psd_matched_filtering, og_signal_f

def get_template_bank(delta_f, freqs):

    f = h5py.File("./utils/template_bank.hdf")

    m1 = f['mass1'][:]
    m2 = f['mass2'][:]
    s1 = f['spin1z'][:]
    s2 = f['spin2z'][:]
    distance=2000.0 
    phase=1.3
    ra=1.375
    dec=-1.2108
    approximant = "IMRPhenomXPHM"
    
    template_bank = []

    for _m1, _m2, _s1, _s2 in tqdm(zip(m1, m2, s1, s2), desc = "Computing template bank", total = len(m1)):
        signal_f, _ = get_fd_waveform(
            approximant=approximant, delta_f=delta_f, f_lower=20, mass1 = _m1, mass2 = _m2,
            spin1z = _s1, spin2z = _s2, distance = distance, phase = phase, ra = ra, dec = dec
        )
        signal_f = np.interp(freqs, signal_f.sample_frequencies.data, signal_f.numpy())
        template_bank.append(FrequencySeries(signal_f, delta_f = delta_f))

    return template_bank
