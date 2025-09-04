import torch  
import numpy as np  
import joblib  
import os  
import argparse  
from utils import torchcbc, data  
from tqdm import tqdm  

def main(args):
    """
    Main function to perform SNR (Signal-to-Noise Ratio) studies for lensed gravitational wave signals.
    The script generates injections, computes SNR metrics, and saves the results for further analysis.
    """

    # Select device: use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    # Define parameter grids for lensing: y (impact parameter) and tD (time delay)
    y_array = np.logspace(-2, 0, args.N)  # Logarithmic spacing between 0.01 and 1
    tD_array = np.logspace(-3, 0, args.N)  # Logarithmic spacing between 0.001 and 1

    # Injection parameters for the simulated gravitational wave event
    inj_dict = dict(
        mass1=66, mass2=29, spin1z=0.4, spin2z=0.3,
        distance=2000.0, phase=1.3, ra=1.375, dec=-1.2108
    )

    # Generate injections (lensed signals), PSD, and the original (unlensed) injection
    injections, psd, og_injection = data.get_template_and_injection(
        y_array = y_array,
        tD_array = tD_array,
        injection_parameters = inj_dict
    )

    # Frequency resolution and frequency array from injections to apply to the template bank
    delta_f = injections[0][0].delta_f
    freqs = injections[0][0].sample_frequencies.data

    # Build a template bank and add the original injection
    template_bank = data.get_template_bank(
        delta_f = delta_f, 
        freqs = freqs
    )
    template_bank.append(og_injection)
    # Convert template bank and PSD to torch tensors (complex128) and move to device
    template_bank = torch.tensor(np.array([data.numpy() for data in template_bank]), dtype = torch.complex128).to(device)
    psd = torch.tensor(psd.numpy(), dtype = torch.complex128).to(device)
    
    # Compute loudness (sigma^2) for the unlensed (original) injection
    unlensed_sigmasq = torchcbc.sigmasq(
        signal = torch.tensor(og_injection.numpy(), dtype = torch.complex128).to(device),
        psd = psd,
        delta_f = delta_f
    )

    # Lists to store best reweighted SNRs, best SNRs before chi^2 testing, and lensed loudness values
    best_nsnrs, best_snrs, lensed_sigmasqs = [[], [], []]

    # Loop over all generated injections and their corresponding optimal filters
    for injection, opt_filter in tqdm(injections, desc = "Filtering injections"):
        # Convert injection and optimal filter to torch tensors (complex128) and move to device
        injection = torch.tensor(injection.numpy(), dtype = torch.complex128).to(device)
        opt_filter = torch.tensor(opt_filter.numpy(), dtype = torch.complex128).to(device)

        # Compute normalization (sigma^2) for the lensed signal
        lens_sigmasq = torchcbc.sigmasq(
            signal = opt_filter,
            psd = psd,
            delta_f = delta_f
        ).item()

        # Compute best reweighted SNR (nsnr) and standard SNR (snr) for this injection
        nsnr, snr = torchcbc.get_metrics_template(
            template_bank = template_bank,
            injection = injection,
            psd = psd,
            delta_f = delta_f,
            batch_size = args.batch_size
        )

        # Store the computed metrics
        best_nsnrs.append(nsnr)
        best_snrs.append(snr)
        lensed_sigmasqs.append(lens_sigmasq)

    # Save results to files in the specified output directory
    joblib.dump(best_nsnrs, os.path.join(args.savedir, "best_nsnrs.bin"))
    joblib.dump(best_snrs, os.path.join(args.savedir, "best_snrs.bin"))
    joblib.dump(unlensed_sigmasq, os.path.join(args.savedir, "unlensed_sigmasq.bin"))
    joblib.dump(lensed_sigmasqs, os.path.join(args.savedir, "lensed_sigmasq.bin"))
    joblib.dump(y_array, os.path.join(args.savedir, "y_array.bin"))
    joblib.dump(tD_array, os.path.join(args.savedir, "td_array.bin"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-N", type=int, default=10,
        help="Number of lensing parameter samples (N^2). Default is %(default)s"
    )
    parser.add_argument(
        "-s", "--savedir", type=str, default="/lhome/ext/uv098/uv0987/workdir/ninovillanueva/lensing_inference/lensing-on-cosmic-strings/python-code/snr_study/torch_snr_study",
        help="Directory to save output files. Default is %(default)s"
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=32,
        help="Batch size for matched filtering. Default is %(default)s"
    )
    args = parser.parse_args()

    main(args)


