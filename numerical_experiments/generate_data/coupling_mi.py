"""Experiment A — pairwise mutual-information heatmaps between
coordinates in the posterior basin.

We draw **Laplace samples** at the posterior peak of each system, i.e.
samples from the multivariate Gaussian ``N(mean, H^{-1})`` where ``H``
is the Hessian we computed in experiment B (after zero-mode removal
and regularisation). This is the leading-order Gaussian basin shape,
and it is what a flow's coupling layers need to capture in the
neighbourhood of the peak. Using Laplace samples avoids the
near-singular Hessian issues that make direct MCMC at the GW peak
prohibitively expensive — the GW basin in unit-cube coords is so tight
that any reasonable isotropic random walk has accept rate ~10^-4.

Pairwise mutual information is estimated from 2D histograms with 30
bins per axis using the plug-in entropy formula
``I(X; Y) = H(X) + H(Y) - H(X, Y)`` with no bias correction.

What this measures: how strongly each pair of coordinates is *linearly*
correlated in the Gaussian approximation. For multivariate Gaussian
samples, the pairwise MI is monotonic in |corr_ij| via
``I = -0.5 log(1 - ρ^2)``, so the heatmap effectively visualises the
inverse-Hessian correlation structure. The Hessian heatmap from
experiment B shows the same physics in matrix form; this script
recasts it as a normalised MI to give a directly interpretable
"how much information one coordinate carries about another" number.

Caveat: non-Gaussian features of the basin (banana ridge in GW,
permutation orbit in LJ) are NOT captured by Laplace samples. They
would only appear from a genuinely posterior-distributed sample set.
For an honest non-Gaussian replay you'd need NS or a properly tuned
HMC chain.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import _lj_compat  # noqa: F401  # patches lennard_jones.energy self-mask
from nsflows.systems.lennard_jones import lennard_jones

os.environ.setdefault("BILBY_STYLE", "none")
warnings.filterwarnings("ignore")

import bilby  # noqa: E402

bilby.core.utils.logger.setLevel(logging.WARNING)


SAMPLED_PARAMS = [
    "chirp_mass", "mass_ratio",
    "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
    "luminosity_distance",
    "dec", "ra", "theta_jn", "psi", "phase",
    "geocent_time",
]

INJECTION_PARAMETERS = dict(
    mass_1=36.0, mass_2=29.0,
    a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3,
    luminosity_distance=2000.0,
    theta_jn=0.4, psi=2.659, phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375, dec=-1.2108,
)


def build_gw():
    duration = 4.0
    sampling_frequency = 2048.0
    np.random.seed(170817)
    wfg = bilby.gw.waveform_generator.WaveformGenerator(
        sampling_frequency=sampling_frequency, duration=duration,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=(
            bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
        ),
        waveform_arguments=dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=50.0, minimum_frequency=20.0,
        ),
    )
    injection = bilby.gw.conversion.generate_all_bbh_parameters(
        INJECTION_PARAMETERS.copy()
    )
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=injection["geocent_time"] - 3,
    )
    ifos.inject_signal(waveform_generator=wfg, parameters=injection)
    priors = bilby.gw.prior.BBHPriorDict()
    priors["geocent_time"] = bilby.core.prior.Uniform(
        minimum=injection["geocent_time"] - 0.1,
        maximum=injection["geocent_time"] + 0.1,
        name="geocent_time",
    )
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=wfg, priors=priors,
    )
    base = {name: float(injection[name]) for name in SAMPLED_PARAMS}
    base["chirp_mass"] = float(injection["chirp_mass"])
    base["mass_ratio"] = float(injection["mass_ratio"])
    u_truth = np.array(
        [priors[name].cdf(base[name]) for name in SAMPLED_PARAMS]
    )
    return likelihood, priors, base, u_truth


def gw_logl_unit(u, likelihood, priors, base):
    params = dict(base)
    for i, name in enumerate(SAMPLED_PARAMS):
        params[name] = float(priors[name].rescale(
            np.clip(u[i], 1e-9, 1 - 1e-9)
        ))
    try:
        p, _ = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(
            dict(params)
        )
        likelihood.parameters.update(p)
        return float(likelihood.log_likelihood_ratio())
    except Exception:
        return -np.inf


def laplace_samples_from_hessian(H: np.ndarray, n_samples: int, *,
                                 rng: np.random.Generator,
                                 zero_tol: float = 1e-6,
                                 cap_eigenvalue: float | None = None
                                 ) -> np.ndarray:
    """Draw N(0, H_eff^{-1}) where H_eff is built from H by dropping the
    near-zero eigenvalues entirely and (optionally) capping very large
    eigenvalues so the inversion does not produce vanishing
    displacements along ultra-stiff directions (which would suppress
    coupling detection in MI estimation).

    Returns shape (n_samples, dim) — samples in the original coordinate
    basis."""
    H = 0.5 * (H + H.T)
    eigs, vecs = np.linalg.eigh(H)
    keep = np.abs(eigs) > zero_tol * np.abs(eigs).max()
    eigs_kept = eigs[keep]
    vecs_kept = vecs[:, keep]
    # Make sure they're positive (negative eigenvalues are numerical artefacts
    # at a true minimum / maximum; flip sign).
    eigs_pos = np.abs(eigs_kept)
    if cap_eigenvalue is not None:
        eigs_pos = np.minimum(eigs_pos, cap_eigenvalue)
    # sample in eigenbasis with std = 1/sqrt(λ)
    z = rng.standard_normal(size=(n_samples, eigs_pos.size)) / np.sqrt(eigs_pos)
    # rotate to original basis
    return z @ vecs_kept.T


def sample_lj_basin_laplace(n_samples: int, H_lj_path: Path, *,
                            box_length: float, seed: int,
                            cap_eigenvalue: float | None = None):
    """LJ samples around IS#0 in displacement coordinates (PBC min-image
    safe by construction since we never reach the box edge with the
    Laplace covariance)."""
    d = np.load(H_lj_path, allow_pickle=True)
    H_lj = d["H_lj"]
    x_is = d["x_is"]
    e_is = float(d["e_is"])
    rng = np.random.default_rng(seed)
    samples = laplace_samples_from_hessian(
        H_lj, n_samples, rng=rng, cap_eigenvalue=cap_eigenvalue,
    )
    return samples, x_is, e_is, box_length


def sample_gw_basin_laplace(n_samples: int, H_gw_path: Path, *,
                            seed: int,
                            cap_eigenvalue: float | None = None):
    d = np.load(H_gw_path, allow_pickle=True)
    H_gw = d["H_gw"]
    u_truth = d["u_truth"]
    logl_truth = float(d["logl_truth"])
    rng = np.random.default_rng(seed)
    samples = laplace_samples_from_hessian(
        H_gw, n_samples, rng=rng, cap_eigenvalue=cap_eigenvalue,
    )
    return samples + u_truth, u_truth, logl_truth


def pairwise_mi_matrix(samples: np.ndarray, n_bins: int = 30) -> np.ndarray:
    """Mutual information between all pairs of columns of ``samples`` via
    2D histograms (plug-in estimator)."""
    n, d = samples.shape
    # marginal entropies
    H_marg = np.empty(d)
    edges_list = []
    for i in range(d):
        h, edges = np.histogram(samples[:, i], bins=n_bins)
        p = h / h.sum()
        # avoid log(0)
        nz = p > 0
        H_marg[i] = -np.sum(p[nz] * np.log(p[nz]))
        edges_list.append(edges)
    # joint entropies and MI
    M = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                M[i, j] = H_marg[i]
                continue
            if j < i:
                M[i, j] = M[j, i]
                continue
            h, _, _ = np.histogram2d(
                samples[:, i], samples[:, j], bins=n_bins,
            )
            p = h / h.sum()
            nz = p > 0
            H_joint = -np.sum(p[nz] * np.log(p[nz]))
            mi = H_marg[i] + H_marg[j] - H_joint
            M[i, j] = M[j, i] = max(mi, 0.0)
    return M, H_marg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100000)
    parser.add_argument("--box-length", type=float, default=2.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=30)
    parser.add_argument("--hessian-npz", type=str,
                        default=str(Path(__file__).parent / "output" / "hessian_spectra.npz"),
                        help="output of hessian_spectrum.py")
    parser.add_argument("--cap-eigenvalue", type=float, default=None,
                        help="if set, cap eigenvalues so that no direction "
                             "has σ < 1/sqrt(cap). Useful when extreme "
                             "stiffness collapses MI signal to zero in that "
                             "direction.")
    parser.add_argument("--output-npz", type=str,
                        default=str(Path(__file__).parent / "output" / "coupling_mi.npz"))
    args = parser.parse_args()

    H_path = Path(args.hessian_npz)
    if not H_path.exists():
        sys.exit(f"Expected {H_path} (run hessian_spectrum.py first)")

    print("Drawing LJ Laplace samples around IS#0...", flush=True)
    samples_lj, x_is, e_is, box = sample_lj_basin_laplace(
        args.n_samples, H_path, box_length=args.box_length,
        seed=args.seed, cap_eigenvalue=args.cap_eigenvalue,
    )
    print(f"  LJ sample shape: {samples_lj.shape}")
    print(f"  LJ displacement std per DOF: "
          f"[{samples_lj.std(axis=0).min():.3e}, "
          f"{samples_lj.std(axis=0).max():.3e}]")

    print("\nDrawing GW Laplace samples around truth...", flush=True)
    samples_gw, u_truth, logl_truth = sample_gw_basin_laplace(
        args.n_samples, H_path, seed=args.seed,
        cap_eigenvalue=args.cap_eigenvalue,
    )
    print(f"  GW sample shape: {samples_gw.shape}")
    print(f"  GW unit-cube displacement std per param: "
          f"[{(samples_gw - u_truth).std(axis=0).min():.3e}, "
          f"{(samples_gw - u_truth).std(axis=0).max():.3e}]")

    print("\nComputing pairwise MI matrices...", flush=True)
    MI_lj, H_marg_lj = pairwise_mi_matrix(samples_lj, n_bins=args.n_bins)
    MI_gw, H_marg_gw = pairwise_mi_matrix(samples_gw, n_bins=args.n_bins)

    # Off-diagonal MI statistics
    d_lj = MI_lj.shape[0]
    d_gw = MI_gw.shape[0]
    off_lj = MI_lj[np.triu_indices(d_lj, k=1)]
    off_gw = MI_gw[np.triu_indices(d_gw, k=1)]

    # Normalise by min(H_marg_i, H_marg_j) per pair to get fractional
    # coupling — this is the "normalised mutual information" (NMI), at
    # least in one common form. Bounded in [0, 1].
    def normalised_mi(M, H):
        d = M.shape[0]
        N = np.zeros_like(M)
        for i in range(d):
            for j in range(d):
                if i == j:
                    N[i, j] = 1.0
                    continue
                denom = min(H[i], H[j])
                N[i, j] = M[i, j] / denom if denom > 1e-9 else 0.0
        return N

    NMI_lj = normalised_mi(MI_lj, H_marg_lj)
    NMI_gw = normalised_mi(MI_gw, H_marg_gw)
    off_nmi_lj = NMI_lj[np.triu_indices(d_lj, k=1)]
    off_nmi_gw = NMI_gw[np.triu_indices(d_gw, k=1)]

    mean_nmi_lj = float(off_nmi_lj.mean())
    mean_nmi_gw = float(off_nmi_gw.mean())
    # Concentration: fraction of total off-diag MI accounted for by the
    # strongest 5% of pairs. Near-uniform coupling -> 0.05; concentrated
    # coupling in a few pairs -> close to 1.
    def concentration(x: np.ndarray, top_frac: float = 0.05) -> float:
        x = np.sort(x)[::-1]
        k = max(1, int(np.ceil(top_frac * x.size)))
        return float(x[:k].sum() / x.sum()) if x.sum() > 0 else 0.0

    conc_lj = concentration(off_nmi_lj)
    conc_gw = concentration(off_nmi_gw)
    # Per-coordinate fanout: how many other coords have meaningful MI?
    # Use a *relative* threshold so the count is comparable across systems
    # with different absolute MI scales.
    def fanout(NMI: np.ndarray, rel_threshold: float = 0.1) -> np.ndarray:
        N = NMI.copy()
        np.fill_diagonal(N, 0.0)
        thr = rel_threshold * N.max()
        return (N > thr).sum(axis=1)

    fan_lj = fanout(NMI_lj)
    fan_gw = fanout(NMI_gw)

    print(f"\nLJ:  mean off-diag NMI = {mean_nmi_lj:.4f}")
    print(f"     concentration (top 5% pairs): {conc_lj:.3f}")
    print(f"     per-coord fanout (>10% of max): "
          f"median={int(np.median(fan_lj))}, range=[{fan_lj.min()}, {fan_lj.max()}]")
    print(f"     total off-diag MI mass: {off_nmi_lj.sum():.3f}")
    print(f"GW:  mean off-diag NMI = {mean_nmi_gw:.4f}")
    print(f"     concentration (top 5% pairs): {conc_gw:.3f}")
    print(f"     per-coord fanout (>10% of max): "
          f"median={int(np.median(fan_gw))}, range=[{fan_gw.min()}, {fan_gw.max()}]")
    print(f"     total off-diag MI mass: {off_nmi_gw.sum():.3f}")

    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        MI_lj=MI_lj, NMI_lj=NMI_lj, H_marg_lj=H_marg_lj,
        MI_gw=MI_gw, NMI_gw=NMI_gw, H_marg_gw=H_marg_gw,
        samples_lj=samples_lj, samples_gw=samples_gw,
        mean_nmi_lj=mean_nmi_lj, mean_nmi_gw=mean_nmi_gw,
        conc_lj=conc_lj, conc_gw=conc_gw,
        fan_lj=fan_lj, fan_gw=fan_gw,
        param_names_gw=np.array(SAMPLED_PARAMS),
    )
    print(f"Wrote {args.output_npz}")

if __name__ == "__main__":
    sys.exit(main())
