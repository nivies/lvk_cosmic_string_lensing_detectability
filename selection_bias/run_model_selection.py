#!/usr/bin/env python
"""
Script for running the nessai sampler on injections lensed by a cosmic string to estimate 
the parameters using three models: an unlensed BBH model, a model that
considers cosmic strings and an isolated point mass lens (PML) model. It performs the
injection in noise simulated from the technical document TD200012 and runs the samplers,
saving each models in separated directories. The bayes factor is not computed explicitely,
but it can be extracted by loading the results.json file from each PE run and comparing the
evidences yielded.

The injections are tailored to have SNRs around ~7, ~12 and ~20 in order to study the SNR boundaries
where the distinguishability between models emerges.

In order to save computation time, at the moment, the PE is being performed on a very reduced
subset of the parameter space (masses, luminosity distance, orbital inclination and lensing parameters).
At some point we should assess/adress the bias that this limitation is introducing.
"""

def run_model_comparison(td_injection, y_injection, folder_name, experiment):
    import bilby
    from utils.lensing_func import Sommerf, hybrid_factor
    from bilby.gw.source import lal_binary_black_hole
    import matplotlib
    import os

    matplotlib.rcParams['text.usetex'] = False
    
    if experiment == 1:
        distance = 6205.03 # SNR ~ 6.99
        outdir = "outdir/bf1"
    elif experiment == 2:
        distance = 3608.04 # SNR ~ 12.02
        outdir = "outdir/bf2"
    elif experiment == 3:
        distance = 2156.78 # SNR ~ 20.12
        outdir = "outdir/bf3"

    # Set the duration and sampling frequency of the data segment that we're
    # going to inject the signal into
    duration = 4.0
    sampling_frequency = 2048.0
    minimum_frequency = 20
    sampler = "nessai"
    dlogz = 0.1
    npool = 4

    from pathlib import Path
    Path(os.path.join(outdir, folder_name)).mkdir(parents=True, exist_ok=True)

    # Specify the output directory and the name of the simulation.
    outdir_ul = os.path.join(outdir, folder_name, "unlensed")
    label = "unlensed"
    bilby.core.utils.setup_logger(outdir=outdir_ul, label=label)

    # Set up a random seed for result reproducibility.  This is optional!
    bilby.core.utils.random.seed(42)

    def generate_cs_lensed_waveform(
            frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
            phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, td, y,**kwargs):
        
        waveform_dict = lal_binary_black_hole(
            frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
            phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)
        
        transmission_factor = Sommerf(frequency_array, td, y)

        for key in waveform_dict.keys():
            waveform_dict[key] = transmission_factor*waveform_dict[key]

        return waveform_dict
    
    def generate_pml_lensed_waveform(
            frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
            phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, tM, y,**kwargs):
        
        waveform_dict = lal_binary_black_hole(
            frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
            phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs)
        
        transmission_factor = hybrid_factor(frequency_array, tM, y)

        for key in waveform_dict.keys():
            waveform_dict[key] = transmission_factor*waveform_dict[key]

        return waveform_dict
    
    injection_parameters = dict(
        mass_1=36,
        mass_2=29,
        a_1=0.4,
        a_2=0.3,
        tilt_1=0.0,
        tilt_2=0.0,
        phi_12=0.0,
        phi_jl=0.0,
        luminosity_distance=distance,
        theta_jn=0.0,
        phase=1.3,
        psi=0,
        ra=1.375,
        dec=-1.2108,
        geocent_time=1126259642.413,
        td=td_injection,
        y=y_injection
    )

    # Fixed arguments passed into the source model
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=50.0,
        minimum_frequency=minimum_frequency,
    )

    # Create the waveform_generator using a LAL BinaryBlackHole source function
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )
    # Create the waveform_generator using a LAL BinaryBlackHole source function
    # the generator will convert all the parameters
    lensed_waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        # frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        frequency_domain_source_model=generate_cs_lensed_waveform,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    pml_lensed_waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        # frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        frequency_domain_source_model=generate_pml_lensed_waveform,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    # Set up interferometers.  In this case we'll use two interferometers
    # (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
    # sensitivity
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - 2,
    )
    ifos.inject_signal(
        waveform_generator=lensed_waveform_generator, parameters=injection_parameters
    )

    ''' UNLENSED PE '''

    priors = bilby.gw.prior.BBHPriorDict()
    for key in [
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "psi",
        "ra",
        "dec",
        "geocent_time",
        "phase",
    ]:
        priors[key] = injection_parameters[key]

    # Perform a check that the prior does not extend to a parameter space longer than the data
    priors.validate_prior(duration, minimum_frequency)

    # Initialise the likelihood by passing in the interferometer data (ifos) and
    # the waveform generator
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator
    )


    # Run sampler.  In this case we're going to use the `dynesty` sampler
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler=sampler,
        npoints=1000,
        injection_parameters=injection_parameters,
        outdir=outdir_ul,
        label=label,
        dlogz=dlogz,
        npool=npool
    )

    # Make a corner plot.
    result.plot_corner()

    ''' CS LENSED PE '''

    # Specify the output directory and the name of the simulation.
    outdir_l = os.path.join(outdir, folder_name, "lensed")
    label = "lensed"
    bilby.core.utils.setup_logger(outdir=outdir_l, label=label)

    lensed_priors = bilby.gw.prior.BBHPriorDict()
    for key in [
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "psi",
        "ra",
        "dec",
        "geocent_time",
        "phase",
    ]:
        lensed_priors[key] = injection_parameters[key]

    lensed_priors['td'] = bilby.core.prior.Uniform(
        minimum=5e-4, maximum=1.3, latex_label="$t_\Delta$"
    )

    lensed_priors['y'] = bilby.core.prior.Uniform(
        minimum=5e-3, maximum=1.3, latex_label="$y$"
    )

    # Perform a check that the prior does not extend to a parameter space longer than the data
    lensed_priors.validate_prior(duration, minimum_frequency)

    # Initialise the likelihood by passing in the interferometer data (ifos) and
    # the waveform generator
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=lensed_waveform_generator
    )

    # Run sampler.  In this case we're going to use the `dynesty` sampler
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=lensed_priors,
        sampler=sampler,
        npoints=1000,
        injection_parameters=injection_parameters,
        outdir=outdir_l,
        label=label,
        dlogz=dlogz,
        npool=npool
    )

    # Make a corner plot.
    result.plot_corner()

    ''' PML LENSED PE '''

    # Specify the output directory and the name of the simulation.
    outdir_pml = os.path.join(outdir, folder_name, "pml_lensed")
    label = "pml_lensed"
    bilby.core.utils.setup_logger(outdir=outdir_pml, label=label)

    lensed_priors = bilby.gw.prior.BBHPriorDict()
    for key in [
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "psi",
        "ra",
        "dec",
        "geocent_time",
        "phase",
    ]:
        lensed_priors[key] = injection_parameters[key]

    lensed_priors['tM'] = bilby.core.prior.Uniform(
        minimum=5e-4, maximum=1.3, latex_label="$t_M$"
    )

    lensed_priors['y'] = bilby.core.prior.Uniform(
        minimum=1e-2, maximum=1.3, latex_label="$y$"
    )

    # Perform a check that the prior does not extend to a parameter space longer than the data
    lensed_priors.validate_prior(duration, minimum_frequency)

    # Initialise the likelihood by passing in the interferometer data (ifos) and
    # the waveform generator
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=pml_lensed_waveform_generator
    )

    # Run sampler.  In this case we're going to use the `dynesty` sampler
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=lensed_priors,
        sampler=sampler,
        npoints=1000,
        injection_parameters=injection_parameters,
        outdir=outdir_pml,
        label=label,
        dlogz=dlogz,
        npool=npool
    )

    # Make a corner plot.
    result.plot_corner()
