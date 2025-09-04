import torch


def sigmasq(signal, psd, delta_f, low_frequency_cutoff=None):
    """
    Compute the matched filter normalization (sigma^2) for batch of signals.

    TODO: Include frequency cutoffs.

    signal: [B, N] or [N] (if 1D, will be unsqueezed to [1, N])
    psd: [Nf] (rfft length)
    Returns: [B] (sigma^2 for each batch)
    """
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)  # [1, N]
    
    if psd.dim() == 1:
        psd = psd.unsqueeze(0)  # [1, N]

    assert signal.shape[1] == psd.shape[1], "Signal length and PSD length must match."
    
    integrand = (signal.conj() * signal) / psd  # [B, Nf]
    sigma_squared = 4.0 * delta_f * integrand.real.sum(dim=1)  # [B]

    return sigma_squared

def matched_filter(template, data, delta_f, psd=None):
    """
    Vectorized matched filter SNR time series for batches of templates and data.

    TODO: Include frequency cutoffs.

    template: [B, N] or [N]
    data: [B, N] or [N]
    psd: [Nf]
    Returns: snr_torch: [B, N] or [N] (matches batch shape)
    """
    if template.dim() == 1:
        template = template.unsqueeze(0)

    if data.dim() == 1:
        data = data.unsqueeze(0)

    if psd.dim() == 1:
        psd = psd.unsqueeze(0)


    B, N = template.shape

    if data.shape != (B, N):
        Warning("Repeating strain data for vectorization.")
        data = data.repeat(B, 1)

    if psd.shape != (B, N):
        Warning("Repeating PSD data for vectorization.")
        psd = psd.repeat(B, 1)

        
    if psd is not None:
        integrand = template.conj() * data / psd  # [B, Nf]
    else:
        integrand = template.conj() * data

    snr_torch = torch.fft.ifft(integrand, n=2*(N-1), norm = "forward", dim=1)  # [B, N]
    template_norm = sigmasq(template, psd, delta_f)  # [B]

    assert not torch.any(template_norm == 0), "Template sigma squared contains zero values."
    
    norm_prefactor = torch.tensor(4.0 * delta_f, dtype = torch.complex128).repeat(B).to(template.device)
    # Normalize SNR time series by template norm
    norm = norm_prefactor / template_norm.sqrt()
    norm = norm.unsqueeze(1)
    snr_torch = snr_torch * norm

    if snr_torch.shape[0] == 1:
        return snr_torch.squeeze(0)

    return snr_torch


def power_chisq(template, data, psd, delta_f, nbins):
    """
    Compute per-bin sigma^2 values for each batch element by dividing the template into
    `nbins` equal-power frequency bins. Vectorized over batch.

    NOTE: The values of the chi^2 are identical to those by PyCBC, but we still need to 
    include frequency cutoffs. 

    Args:
        template: [B, N] complex-valued frequency-domain templates
        psd: [B, N] or [N] real-valued PSD
        delta_f: float, frequency resolution
        nbins: int, number of frequency bins

    Returns:
        sigmasq_bins: [B, nbins] tensor of per-bin sigma^2 values
        bin_masks: [nbins, N] boolean tensor indicating which freqs belong to each bin
    """

    if template.dim() == 1:
        template = template.unsqueeze(0)

    if data.dim() == 1:
        data = data.unsqueeze(0)
    
    if psd.dim() == 1:
        psd = psd.unsqueeze(0)

    B, N = template.shape
    if data.shape != (B, N):
        Warning("Repeating strain data for vectorization.")
        data = data.repeat(B, 1)

    if psd.shape != (B, N):
        Warning("Repeating PSD data for vectorization.")
        psd = psd.repeat(B, 1)

    template_norm = sigmasq(
        signal = template,
        psd = psd,
        delta_f = delta_f
    )

    # Compute cumulative σ² over frequency
    integrand = (template.conj() * template) / psd  # [B, N]
    integrand = integrand.real
    norm_sigma = 4 * delta_f / integrand.sqrt().sum(axis = 1) # Included sigma squared normalization for bin calculation. For the moment, the results are worse.
    norm_sigma = norm_sigma.unsqueeze(1).repeat(1, N)

    sigma_cumsum = torch.cumsum(integrand * norm_sigma, dim=1)  # [B, N]
    sigma_total = sigma_cumsum[:, -1].unsqueeze(1)  # [B, 1]

    # Get target boundaries for each bin, shared across batch (normalized fraction)
    frac_edges = torch.linspace(0, 1, nbins + 1, dtype=torch.float64, device=template.device)  # [nbins+1]

    # Expand to shape [B, nbins+1] in absolute σ² units
    sigma_targets = sigma_total * frac_edges.unsqueeze(0)  # [B, nbins+1]

    # Find frequency bin edges using searchsorted (vectorized over batch)
    # Get the index for each bin edge where cumulative σ² crosses the target
    bin_indices = torch.searchsorted(sigma_cumsum, sigma_targets, right=True).clamp(max=N+1)  # [B, nbins+1]

    # Build bin masks: [nbins, N]
    bin_masks = []
    for i in range(nbins):
        start = bin_indices[:, i].unsqueeze(1)  # [B, 1]
        end = bin_indices[:, i+1].unsqueeze(1)  # [B, 1]
        freq_indices = torch.arange(N, device=template.device).unsqueeze(0)  # [1, N]
        mask = (freq_indices >= start) & (freq_indices < end)  # [B, N]
        bin_masks.append(mask)

    bin_masks = torch.stack(bin_masks, dim=1)  # [B, nbins, N]

    # Apply masks to integrand and compute per-bin SNR!
    integrand_snr = (template.conj() * data) / psd  # [B, N]
    integrand_snr_exp = integrand_snr.unsqueeze(1)  # [B, 1, N]
    masked_integrand_snr = integrand_snr_exp * bin_masks  # [B, nbins, N]
    norm = 4.0 * delta_f / template_norm.sqrt().view(B, 1, 1)  # [B]
    # Prepare normalization for [B, nbins, 2*(N-1)]
    # Batched IFFT: output is [B, nbins, 2*(N-1)]
    snr_bins = torch.fft.ifft(masked_integrand_snr, n=2*(N-1), norm="forward", dim=2)

    snr_bins = snr_bins * norm  # [B, nbins, 2*(N-1)]

    total_snr = matched_filter(
        template = template, 
        data = data, 
        delta_f = delta_f, 
        psd=psd
        ).view(B, 1, 2*(N-1))

    sumand = (snr_bins - total_snr / nbins).abs().square()

    chisq_ts = nbins * sumand.sum(axis=1)

    return chisq_ts

def newsnr(snr: torch.Tensor, chisq: torch.Tensor, nbins: int) -> torch.Tensor:
    """
    Compute reweighted SNR (newsnr) from standard matched-filter SNR and chi-squared statistic.

    Parameters:
        rho (Tensor): matched-filter SNR (real-valued), shape (batch,) or scalar
        chi2 (Tensor): chi-squared statistic (real-valued), shape (batch,) or scalar
        dof (int): degrees of freedom = 2*p - 2, where p is number of chi2 frequency bins

    Returns:
        Tensor: reweighted SNR (same shape as rho)
    """
    dof = nbins * 2 - 2
    chi2_r = chisq / dof  # Reduced chi-squared
    penalty = 0.5 * (1 + chi2_r**3)
    scale = torch.where(chi2_r > 1, penalty**(-1/6), torch.ones_like(penalty)).to(chisq.device)
    return snr * scale

def get_metrics(filter, injection, psd, delta_f):
    snr = matched_filter(
        template = filter, 
        data = injection, 
        delta_f = delta_f, 
        psd = psd
        )

    # Chi square test SNR reweighting:
    nbins = 26 # Ad hoc number of bins to compute the chi squared test.
    chisq = power_chisq(
        template = filter, 
        data = injection, 
        delta_f = delta_f,
        psd = psd,
        nbins = nbins
        )
    nsnr = newsnr(
        torch.abs(snr),
        chisq, 
        nbins
        ) # Obtain reweighted SNR
    
    return snr, chisq, nsnr

def get_metrics_template(template_bank, injection, psd, delta_f, batch_size):
    """
    Process the template bank in batches, computing SNR, chi^2, and newSNR for each batch, and keeping track of the best newSNR found.
    Returns: the overall best newSNR value across all batches.
    """
    nbins = 26
    N_templates = template_bank.shape[0]
    best_nsnr = None

    for start in range(0, N_templates, batch_size):
        end = min(start + batch_size, N_templates)
        batch_templates = template_bank[start:end]
        # Expand injection to batch shape if needed
        if injection.ndim == 1:
            batch_injection = injection.unsqueeze(0).expand(end-start, -1)
        elif injection.shape[0] == 1:
            batch_injection = injection.expand(end-start, -1)
        else:
            batch_injection = injection[start:end]
        snr = matched_filter(
            template = batch_templates,
            data = batch_injection,
            delta_f = delta_f,
            psd = psd
        )
        chisq = power_chisq(
            template = batch_templates,
            data = batch_injection,
            delta_f = delta_f,
            psd = psd,
            nbins = nbins
        )
        snr = torch.abs(snr)
        nsnr = newsnr(
            snr,
            chisq,
            nbins
        )  # [batch, ...]
        # Take max over time/axis if needed
        if nsnr.ndim > 1:
            nsnr = nsnr.max(dim=1).values
            snr = snr.max(dim=1).values
        batch_best = nsnr.max().item()

        if (best_nsnr is None) or (batch_best > best_nsnr):
            best_nsnr = batch_best
            best_snr = snr.max().item()
    return best_nsnr, best_snr # Best reweighted SNR and best found SNR.
