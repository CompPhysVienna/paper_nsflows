"""Probe the BBH log-likelihood along KNOWN degeneracy directions in
parameter space.

Random straight-line probes (experiments/posterior) showed only one
sharp basin around the injection. The interesting multimodality of GW
parameter estimation lives on low-dimensional manifolds: the chirp-mass
/ mass-ratio "banana", the sky-position antipode, and the periodic
phase. Here we evaluate ``log L`` along those directions explicitly.

Probes:
* **Phase wrap 1D** — sweep ``phase`` over [0, 2π], all other parameters
  fixed at the injection truth. Since we do NOT phase-marginalise, this
  reveals the phase structure of the un-marginalised likelihood.
* **Chirp-mass / mass-ratio 2D** — grid over (chirp_mass, mass_ratio)
  across their prior ranges, all other parameters fixed. Expected: a
  ridge that is much wider in ``q`` than in ``M_c``.
* **Sky position 2D (ra, dec)** — grid over the full sky, all other
  parameters fixed. Expected: a primary mode at the truth and at least
  one secondary mode at the antipode for two-detector configurations
  (somewhat broken for the H1/L1/V1 network used here).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("BILBY_STYLE", "none")
warnings.filterwarnings("ignore")

import bilby  # noqa: E402

bilby.core.utils.logger.setLevel(logging.WARNING)


# Reuse the same injection / waveform setup as the random-direction probe
INJECTION_PARAMETERS = dict(
    mass_1=36.0, mass_2=29.0,
    a_1=0.4, a_2=0.3,
    tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3,
    luminosity_distance=2000.0,
    theta_jn=0.4, psi=2.659, phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375, dec=-1.2108,
)


def build_likelihood_and_priors():
    duration = 4.0
    sampling_frequency = 2048.0
    np.random.seed(170817)

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomPv2", reference_frequency=50.0,
        minimum_frequency=20.0,
    )
    wfg = bilby.gw.waveform_generator.WaveformGenerator(
        sampling_frequency=sampling_frequency,
        duration=duration,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=(
            bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
        ),
        waveform_arguments=waveform_arguments,
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
    ifos.inject_signal(
        waveform_generator=wfg, parameters=injection
    )
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


def safe_logl(likelihood, params: dict) -> float:
    try:
        # masses may be invalid if chirp/q corner of grid; convert and check
        p, _ = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(
            dict(params)
        )
        likelihood.parameters.update(p)
        return float(likelihood.log_likelihood_ratio())
    except Exception:
        return -np.inf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-points-1d", type=int, default=201)
    parser.add_argument("--n-points-2d", type=int, default=51)
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent / "output" / "gw_degeneracies.npz"))
    args = parser.parse_args()

    print("Building likelihood (a few seconds)...", flush=True)
    likelihood, priors, injection = build_likelihood_and_priors()

    # generate_all_bbh_parameters populates a lot of derived parameters
    # (total_mass, symmetric_mass_ratio, ...) and the bilby converter
    # then derives mass_1, mass_2 from total_mass etc., IGNORING any
    # updates we make to chirp_mass / mass_ratio. Build a minimal base
    # dict that contains only the canonical sampled parameters so that
    # the converter recomputes masses from chirp_mass / mass_ratio.
    truth_full = bilby.gw.conversion.generate_all_bbh_parameters(
        INJECTION_PARAMETERS.copy()
    )
    base = {
        "chirp_mass": float(truth_full["chirp_mass"]),
        "mass_ratio": float(truth_full["mass_ratio"]),
        "a_1": float(truth_full["a_1"]),
        "a_2": float(truth_full["a_2"]),
        "tilt_1": float(truth_full["tilt_1"]),
        "tilt_2": float(truth_full["tilt_2"]),
        "phi_12": float(truth_full["phi_12"]),
        "phi_jl": float(truth_full["phi_jl"]),
        "luminosity_distance": float(truth_full["luminosity_distance"]),
        "theta_jn": float(truth_full["theta_jn"]),
        "psi": float(truth_full["psi"]),
        "phase": float(truth_full["phase"]),
        "geocent_time": float(truth_full["geocent_time"]),
        "ra": float(truth_full["ra"]),
        "dec": float(truth_full["dec"]),
    }
    e0 = safe_logl(likelihood, base)
    print(f"log L at truth: {e0:.4f}")

    # ------------------------------------------------------------------
    # 1) Phase wrap (1D)
    # ------------------------------------------------------------------
    phase_grid = np.linspace(0.0, 2.0 * np.pi, args.n_points_1d)
    logl_phase = np.empty(args.n_points_1d)
    for k, ph in enumerate(phase_grid):
        params = dict(base); params["phase"] = float(ph)
        logl_phase[k] = safe_logl(likelihood, params)
        if (k + 1) % 50 == 0:
            print(f"  phase progress: {k+1}/{args.n_points_1d}", flush=True)

    # ------------------------------------------------------------------
    # 2) Chirp-mass / mass-ratio 2D
    # ------------------------------------------------------------------
    mc_min, mc_max = priors["chirp_mass"].minimum, priors["chirp_mass"].maximum
    q_min, q_max = priors["mass_ratio"].minimum, priors["mass_ratio"].maximum
    mc_grid = np.linspace(mc_min, mc_max, args.n_points_2d)
    q_grid = np.linspace(q_min, q_max, args.n_points_2d)
    logl_mq = np.full((args.n_points_2d, args.n_points_2d), -np.inf)
    done = 0
    for ii, mc in enumerate(mc_grid):
        for jj, q in enumerate(q_grid):
            params = dict(base)
            params["chirp_mass"] = float(mc)
            params["mass_ratio"] = float(q)
            logl_mq[ii, jj] = safe_logl(likelihood, params)
            done += 1
            if done % 500 == 0:
                print(f"  Mc/q progress: {done}/{args.n_points_2d**2}", flush=True)

    # ------------------------------------------------------------------
    # 3) Sky position (ra, dec) 2D
    # ------------------------------------------------------------------
    ra_grid = np.linspace(0.0, 2.0 * np.pi, args.n_points_2d)
    dec_grid = np.linspace(-np.pi / 2, np.pi / 2, args.n_points_2d)
    logl_sky = np.full((args.n_points_2d, args.n_points_2d), -np.inf)
    done = 0
    for ii, ra in enumerate(ra_grid):
        for jj, dec in enumerate(dec_grid):
            params = dict(base); params["ra"] = float(ra); params["dec"] = float(dec)
            logl_sky[ii, jj] = safe_logl(likelihood, params)
            done += 1
            if done % 500 == 0:
                print(f"  sky progress: {done}/{args.n_points_2d**2}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        phase_grid=phase_grid, logl_phase=logl_phase,
        mc_grid=mc_grid, q_grid=q_grid, logl_mq=logl_mq,
        ra_grid=ra_grid, dec_grid=dec_grid, logl_sky=logl_sky,
        e0=e0,
        truth_phase=base["phase"],
        truth_chirp_mass=base["chirp_mass"],
        truth_mass_ratio=base["mass_ratio"],
        truth_ra=base["ra"], truth_dec=base["dec"],
    )
    print(f"Wrote {args.output}")
    def _summary(name, a):
        finite = np.isfinite(a)
        if finite.any():
            print(f"  {name}: range=[{a[finite].min():.3g}, {a[finite].max():.3g}] "
                  f"({finite.sum()}/{a.size} finite)")
        else:
            print(f"  {name}: all non-finite")
    _summary("phase 1D", logl_phase)
    _summary("Mc/q 2D", logl_mq)
    _summary("sky 2D", logl_sky)


if __name__ == "__main__":
    sys.exit(main())
