"""Experiment B — compute and compare the local-geometry conditioning
of the LJ-8 energy basin and the GW-BBH log-likelihood at their
respective reference points.

We compute:
* LJ: the 16x16 Hessian of E(x) at the inherent-structure IS#0
  (lowest minimum found by basin hopping), via torch autograd.
* GW: the 15x15 *observed* Fisher information matrix at the injection
  truth, i.e. the Hessian of -log L(u) in the unit-cube prior
  parametrisation, via central finite differences.

The unit-cube parametrisation matters because it puts every GW
parameter on the same scale — without it, the Hessian is dominated by
whichever parameter has the largest natural range (e.g.
luminosity_distance in Mpc dwarfs the angular parameters).

For LJ we work in physical coordinates because the periodic box is
square; the natural scale (the box length) is the same for every DOF.
Removing the centre of mass would project out two zero modes; we keep
them so the spectrum is reported on the full 16D space.

Output:
- experiments/training_complexity/output/hessian_spectra.npz
- experiments/training_complexity/output/hessian_spectra.png
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


# ----- GW setup duplicated from experiments/symmetries/probe_gw.py ----

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


def build_gw_likelihood():
    duration = 4.0
    sampling_frequency = 2048.0
    np.random.seed(170817)
    wfg = bilby.gw.waveform_generator.WaveformGenerator(
        sampling_frequency=sampling_frequency,
        duration=duration,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=(
            bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
        ),
        waveform_arguments=dict(
            waveform_approximant="IMRPhenomPv2", reference_frequency=50.0,
            minimum_frequency=20.0,
        ),
    )
    injection = bilby.gw.conversion.generate_all_bbh_parameters(
        INJECTION_PARAMETERS.copy()
    )
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
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
    return likelihood, priors, injection


def build_base_truth(injection, priors):
    base = {
        name: float(injection[name]) for name in SAMPLED_PARAMS
    }
    base["chirp_mass"] = float(injection["chirp_mass"])
    base["mass_ratio"] = float(injection["mass_ratio"])
    u_truth = np.array(
        [priors[name].cdf(base[name]) for name in SAMPLED_PARAMS]
    )
    return base, u_truth


def gw_neg_logl_unit_factory(likelihood, priors, base):
    def fn(u):
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
            return -float(likelihood.log_likelihood_ratio())
        except Exception:
            return 1e6
    return fn


def numerical_hessian(fn, x, eps):
    n = x.size
    H = np.zeros((n, n))
    f0 = fn(x)
    # diagonal
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        H[i, i] = (fn(xp) - 2.0 * f0 + fn(xm)) / (eps * eps)
    # off-diagonal
    for i in range(n):
        for j in range(i):
            xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = H[j, i] = (
                fn(xpp) - fn(xpm) - fn(xmp) + fn(xmm)
            ) / (4 * eps * eps)
    return H


def lj_hessian(box: float, cutin: float, n_particles: int):
    device = torch.device("cpu")
    rho = n_particles / box ** 2
    system = lennard_jones(
        n_particles=n_particles, dimensions=2, rho=rho,
        device=device, cutin=cutin, lrc=True,
    )
    cat = np.load(
        Path(__file__).parent / "output" / "lj_minima.npz",
        allow_pickle=True,
    )
    x_is = cat["catalog_configs"][0].astype(np.float64)  # IS#0

    def energy_scalar(x):
        return system.energy(x.unsqueeze(0))[0]

    x_t = torch.tensor(x_is, dtype=torch.float64, requires_grad=True)
    H = torch.autograd.functional.hessian(energy_scalar, x_t)
    return H.detach().numpy(), x_is, float(system.energy(x_t.unsqueeze(0)).item())


def gw_hessian(eps: float):
    likelihood, priors, injection = build_gw_likelihood()
    base, u_truth = build_base_truth(injection, priors)
    fn = gw_neg_logl_unit_factory(likelihood, priors, base)
    print(f"  GW: building Hessian via finite differences (eps={eps}), "
          f"~{len(SAMPLED_PARAMS) * (len(SAMPLED_PARAMS) + 1) // 2 * 4} "
          f"likelihood evals...", flush=True)
    H = numerical_hessian(fn, u_truth, eps=eps)
    return H, u_truth, -fn(u_truth)


def spectrum_stats(eigs: np.ndarray, vecs: np.ndarray | None = None,
                   zero_tol: float = 1e-6):
    """Stats with proper zero-mode handling.

    Excludes |λ| < zero_tol * max|λ| from the condition-number calc
    (these are flat modes / numerical noise). Computes participation
    ratios from eigenvectors when available — high participation =
    delocalised collective mode, low = localised single-coordinate
    mode."""
    lam_max_abs = np.abs(eigs).max()
    keep = np.abs(eigs) > zero_tol * lam_max_abs
    eigs_kept = eigs[keep]
    pos = eigs_kept[eigs_kept > 0]
    if pos.size == 0:
        return dict(cond=np.inf, n_stiff=0, n_zero=int((~keep).sum()),
                    lambda_min=eigs.min(), lambda_max=eigs.max())
    lam_max = pos.max()
    lam_min = pos.min()
    median = np.median(pos)
    n_stiff = int(np.sum(pos > median))
    out = dict(
        cond=lam_max / lam_min, n_stiff=n_stiff,
        n_zero=int((~keep).sum()),
        lambda_min=lam_min, lambda_max=lam_max,
        median=median, n_positive=pos.size, n_total=eigs.size,
    )
    if vecs is not None:
        # Participation ratio for each mode (over the kept modes).
        v_kept = vecs[:, keep]
        # P(v) = 1 / sum(v_i^4)
        P = 1.0 / np.sum(v_kept ** 4, axis=0)
        out["participation_ratios"] = P
        out["mean_participation"] = float(P.mean())
        out["median_participation"] = float(np.median(P))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--box-length", type=float, default=2.9)
    parser.add_argument("--cutin", type=float, default=0.8)
    parser.add_argument("--n-particles", type=int, default=8)
    parser.add_argument("--gw-eps", type=float, default=2e-3)
    parser.add_argument("--output-npz", type=str,
                        default=str(Path(__file__).parent / "output" / "hessian_spectra.npz"))
    parser.add_argument("--output-png", type=str,
                        default=str(Path(__file__).parent / "output" / "hessian_spectra.png"))
    args = parser.parse_args()

    print("Computing LJ Hessian at IS#0 (torch autograd, exact)...", flush=True)
    H_lj, x_is, e_is = lj_hessian(args.box_length, args.cutin, args.n_particles)
    print(f"  LJ IS#0 energy: {e_is:.4f}")
    H_lj = 0.5 * (H_lj + H_lj.T)
    eigs_lj, vecs_lj = np.linalg.eigh(H_lj)
    stats_lj = spectrum_stats(eigs_lj, vecs_lj)
    print(f"  LJ Hessian eigenvalue range: "
          f"[{eigs_lj.min():.3e}, {eigs_lj.max():.3e}]")
    print(f"  LJ effective condition number (zero modes removed): "
          f"{stats_lj['cond']:.3e}")
    print(f"  LJ zero/null modes removed: {stats_lj['n_zero']}")
    print(f"  LJ stiff modes (above median of constrained): "
          f"{stats_lj['n_stiff']} / {stats_lj['n_positive']}")
    print(f"  LJ mean participation ratio: "
          f"{stats_lj['mean_participation']:.2f} "
          f"(out of {stats_lj['n_total']} DOF)")

    print("\nComputing GW Hessian at truth (finite differences)...",
          flush=True)
    H_gw, u_truth, logl_truth = gw_hessian(args.gw_eps)
    H_gw = 0.5 * (H_gw + H_gw.T)
    eigs_gw, vecs_gw = np.linalg.eigh(H_gw)
    stats_gw = spectrum_stats(eigs_gw, vecs_gw)
    print(f"  GW logL at truth: {logl_truth:.4f}")
    print(f"  GW Hessian eigenvalue range: "
          f"[{eigs_gw.min():.3e}, {eigs_gw.max():.3e}]")
    print(f"  GW effective condition number: {stats_gw['cond']:.3e}")
    print(f"  GW zero/null modes removed: {stats_gw['n_zero']}")
    print(f"  GW stiff modes: {stats_gw['n_stiff']} / "
          f"{stats_gw['n_positive']}")
    print(f"  GW mean participation ratio: "
          f"{stats_gw['mean_participation']:.2f} "
          f"(out of {stats_gw['n_total']} DOF)")

    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        H_lj=H_lj, eigs_lj=eigs_lj,
        H_gw=H_gw, eigs_gw=eigs_gw,
        x_is=x_is, e_is=e_is,
        u_truth=u_truth, logl_truth=logl_truth,
        stats_lj_cond=stats_lj["cond"],
        stats_gw_cond=stats_gw["cond"],
        param_names=np.array(SAMPLED_PARAMS),
    )
    print(f"Wrote {args.output_npz}")

    # ---- plot ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (0, 0): eigenvalue spectra
    ax = axes[0, 0]
    ax.semilogy(np.arange(eigs_lj.size),
                np.sort(np.abs(eigs_lj))[::-1], "o-",
                label=f"LJ-8 (cond$_\\mathrm{{eff}}$ {stats_lj['cond']:.1e})",
                color="C0")
    ax.semilogy(np.arange(eigs_gw.size),
                np.sort(np.abs(eigs_gw))[::-1], "s-",
                label=f"GW BBH (cond$_\\mathrm{{eff}}$ {stats_gw['cond']:.1e})",
                color="C3")
    ax.set_xlabel("eigenvalue index (sorted, largest first)")
    ax.set_ylabel(r"$|\lambda|$  (log scale)")
    ax.set_title("A. Eigenvalue spectra of local Hessian")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # (0, 1): participation ratios per mode
    ax = axes[0, 1]
    n_lj_kept = stats_lj["participation_ratios"].size
    n_gw_kept = stats_gw["participation_ratios"].size
    # sort modes by eigenvalue (largest first) for x-axis alignment
    abs_eigs_lj = np.abs(eigs_lj)
    abs_eigs_gw = np.abs(eigs_gw)
    keep_lj = abs_eigs_lj > 1e-6 * abs_eigs_lj.max()
    keep_gw = abs_eigs_gw > 1e-6 * abs_eigs_gw.max()
    order_lj = np.argsort(abs_eigs_lj[keep_lj])[::-1]
    order_gw = np.argsort(abs_eigs_gw[keep_gw])[::-1]
    P_lj = stats_lj["participation_ratios"][order_lj]
    P_gw = stats_gw["participation_ratios"][order_gw]
    ax.plot(np.arange(P_lj.size), P_lj, "o-",
            label=f"LJ-8 (mean P = {P_lj.mean():.1f} / {eigs_lj.size} DOF)",
            color="C0")
    ax.plot(np.arange(P_gw.size), P_gw, "s-",
            label=f"GW BBH (mean P = {P_gw.mean():.1f} / {eigs_gw.size} DOF)",
            color="C3")
    ax.axhline(1.0, color="k", lw=0.5, ls=":", label="P=1: localised (axis-aligned)")
    ax.set_xlabel("mode index (largest |λ| first)")
    ax.set_ylabel("participation ratio P")
    ax.set_title("B. How collective is each mode?  "
                 "P=N is fully delocalised, P=1 is axis-aligned")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1, 0): LJ Hessian heatmap (normalised by max |λ|)
    ax = axes[1, 0]
    H_lj_norm = H_lj / abs_eigs_lj.max()
    im = ax.imshow(H_lj_norm, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_title(f"C. LJ Hessian (normalised by max |λ| = "
                 f"{abs_eigs_lj.max():.2e})")
    ax.set_xlabel("DOF index"); ax.set_ylabel("DOF index")
    fig.colorbar(im, ax=ax, label=r"$H_{ij}/\max|\lambda|$")

    # (1, 1): GW Hessian heatmap (normalised by max |λ|)
    ax = axes[1, 1]
    H_gw_norm = H_gw / abs_eigs_gw.max()
    im = ax.imshow(H_gw_norm, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_title(f"D. GW Hessian (normalised by max |λ| = "
                 f"{abs_eigs_gw.max():.2e})")
    ax.set_xticks(range(len(SAMPLED_PARAMS)))
    ax.set_xticklabels(SAMPLED_PARAMS, rotation=70, fontsize=7)
    ax.set_yticks(range(len(SAMPLED_PARAMS)))
    ax.set_yticklabels(SAMPLED_PARAMS, fontsize=7)
    fig.colorbar(im, ax=ax, label=r"$H_{ij}/\max|\lambda|$")

    fig.suptitle(
        "Local geometry at the posterior peak — LJ-8 vs GW BBH",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=140)
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    sys.exit(main())
