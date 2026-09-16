"""
Evaluation for the spectral-masking U-Net denoiser — REPRODUCIBLE.

At startup you enter two integers:

    1. Speech index  : which Label 0 file to use for the exported WAVs.
                       Valid range printed on screen (0 to N-1).

    2. Noise seed    : fixes the noise file selection, noise gains, and SNR
                       for that specific mixture. Same seed → same WAVs.

Run with the same two numbers → byte-identical test_*.wav files.
Change either number → different mixture, still reproducible.

Mean SI-SNR metrics are still reported over all Label 0 files in test.csv.

Exports:
    test_clean_target.wav    -- Label 0 clean speech
    test_noisy_input.wav     -- speech + battlefield noise
    test_cleaned_output.wav  -- model output
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import soundfile as sf

from train import (
    SAMPLE_RATE, SEGMENT_LEN, SNR_DB_RANGE, NOISE_GAIN_RANGE,
    CHECKPOINT_PATH, _HERE,
    SpectralUNet, AudioIO, si_snr, mix_at_snr,
    stft_mag_phase, istft_from_mag_phase,
)


# ============================================================================
# Noise mixer
# ============================================================================
def _load_noise(io: AudioIO, noise_paths) -> torch.Tensor:
    """Sum 1-3 random noise files with random gains, peak-normalize."""
    combined = torch.zeros(1, SEGMENT_LEN)
    for _ in range(random.randint(1, 3)):
        n = io.load(random.choice(noise_paths), SEGMENT_LEN, random_crop=True)
        combined = combined + n * random.uniform(*NOISE_GAIN_RANGE)
    peak = torch.max(torch.abs(combined))
    if peak > 1e-8:
        combined = combined / peak
    return combined


def _infer(model, noisy: torch.Tensor, device) -> torch.Tensor:
    """noisy: [1, T] waveform. Returns enhanced waveform [1, T]."""
    mag, phase = stft_mag_phase(noisy)
    with torch.no_grad():
        mask = model(mag.unsqueeze(1).to(device))
    est_mag = mask.squeeze(1).cpu() * mag
    est = istft_from_mag_phase(est_mag, phase, length=SEGMENT_LEN)
    return torch.clamp(est, -1.0, 1.0)


# ============================================================================
# Interactive prompts
# ============================================================================
def prompt_for_index(speech_paths, root) -> int:
    n = len(speech_paths)
    print(f"\nAvailable Label 0 files: index 0 to {n - 1}")
    preview = min(20, n)
    print(f"Preview (first {preview}):")
    for i in range(preview):
        print(f"   [{i:4d}] {os.path.relpath(speech_paths[i], root)}")
    if n > preview:
        print(f"   ... and {n - preview} more")

    while True:
        raw = input(f"\nEnter speech index (0-{n - 1}, blank = random): ").strip()
        if raw == "":
            idx = random.randrange(n)
            print(f"  → random: {idx}")
            return idx
        try:
            idx = int(raw)
        except ValueError:
            print("  ✗ not an integer, try again")
            continue
        if 0 <= idx < n:
            return idx
        print(f"  ✗ out of range, must be 0-{n - 1}")


def prompt_for_noise_seed() -> int:
    """Prompt for a noise seed. Blank = random (seed will be printed)."""
    while True:
        raw = input("Enter noise seed (integer, blank = random): ").strip()
        if raw == "":
            seed = random.randrange(1_000_000)
            print(f"  → random seed: {seed}")
            return seed
        try:
            seed = int(raw)
            print(f"  → noise seed: {seed}")
            return seed
        except ValueError:
            print("  ✗ not an integer, try again")


# ============================================================================
# Main
# ============================================================================
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint '{CHECKPOINT_PATH}' not found. Run `python train.py` first."
        )

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model = SpectralUNet(base=ckpt.get("base", 16)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    io = AudioIO(sample_rate=SAMPLE_RATE)

    # ---------------------------------------------------------------
    # Collect Label 0 (speech) and Label 1-6 (noise) files from test.csv
    # ---------------------------------------------------------------
    test_df = pd.read_csv(os.path.join(_HERE, "dataset", "test.csv"))
    speech_paths, noise_paths = [], []
    for _, row in test_df.iterrows():
        rel = str(row.get("path", "")).strip()
        if not rel:
            continue
        full_p = os.path.join(_HERE, "dataset", rel)
        if not os.path.exists(full_p):
            continue
        try:
            lbl = int(row["label"])
        except (ValueError, TypeError, KeyError):
            continue
        if lbl == 0:
            try:
                d, _ = sf.read(full_p, dtype="float32", always_2d=True)
                if float(np.max(np.abs(d))) >= 1e-3:
                    speech_paths.append(full_p)
            except Exception:
                continue
        elif lbl in (1, 2, 3, 4, 5, 6):
            noise_paths.append(full_p)

    if not speech_paths:
        raise ValueError("No usable Label 0 files in test.csv.")
    if not noise_paths:
        raise ValueError("No Label 1-6 noise files in test.csv.")

    print(f"\nEvaluating over {len(speech_paths)} Label 0 file(s) "
          f"with {len(noise_paths)} noise source(s).")

    # ---------------------------------------------------------------
    # Ask user for the two numbers that pin the exported WAVs
    # ---------------------------------------------------------------
    export_idx = prompt_for_index(speech_paths, _HERE)
    chosen = speech_paths[export_idx]
    print(f"\nExport speech file: speech_paths[{export_idx}]")
    print(f"                  = {os.path.relpath(chosen, _HERE)}")

    noise_seed = prompt_for_noise_seed()

    print(f"\n----------------------------------------------------")
    print(f"  Reproducible combo: speech index = {export_idx}")
    print(f"                      noise seed   = {noise_seed}")
    print(f"  Re-run with these two numbers for identical WAVs.")
    print(f"----------------------------------------------------\n")

    # ---------------------------------------------------------------
    # Aggregate pass over all speech files (fresh noise each file)
    # ---------------------------------------------------------------
    noisy_snrs, out_snrs = [], []
    export_bundle = None

    for i, path in enumerate(speech_paths):
        clean = io.load(path, SEGMENT_LEN, random_crop=True)
        if torch.max(torch.abs(clean)) < 1e-4:
            continue

        # For the export file ONLY, lock the RNG so this mixture is
        # deterministic given the chosen seed.
        if i == export_idx:
            random.seed(noise_seed)
            np.random.seed(noise_seed)
            torch.manual_seed(noise_seed)

        noise = _load_noise(io, noise_paths)
        snr_db = random.uniform(*SNR_DB_RANGE)
        noisy, clean_scaled = mix_at_snr(clean, noise, snr_db)

        est = _infer(model, noisy, device)

        n_snr = si_snr(noisy.unsqueeze(0),
                       clean_scaled.unsqueeze(0)).item()
        o_snr = si_snr(est,
                       clean_scaled.unsqueeze(0)).item()
        noisy_snrs.append(n_snr)
        out_snrs.append(o_snr)

        if i == export_idx:
            export_bundle = (clean_scaled, noisy, est.squeeze(0),
                             path, snr_db, n_snr, o_snr)

    if not noisy_snrs:
        raise RuntimeError("No valid test segments after silence filter.")

    mean_noisy = float(np.mean(noisy_snrs))
    mean_out   = float(np.mean(out_snrs))
    print(f"Mean SI-SNR (noisy input): {mean_noisy:+.2f} dB")
    print(f"Mean SI-SNR (denoised)   : {mean_out:+.2f} dB")
    print(f"Mean improvement         : {mean_out - mean_noisy:+.2f} dB")
    print(f"Files evaluated          : {len(noisy_snrs)}")

    # ---------------------------------------------------------------
    # Export comparison WAVs for the chosen file
    # ---------------------------------------------------------------
    if export_bundle is None:
        print("\n⚠ Chosen index had no valid audio. Skipping WAV export.")
        return

    clean_s, noisy_s, est_s, path, snr_db, n_snr, o_snr = export_bundle
    print(f"\nExporting comparison WAVs for: "
          f"{os.path.relpath(path, _HERE)}")
    print(f"  sampled SNR (mixture): {snr_db:+.2f} dB")
    print(f"  SI-SNR noisy/denoised: {n_snr:+.2f} / {o_snr:+.2f} dB")

    sf.write("test_clean_target.wav",   clean_s.squeeze(0).numpy(), SAMPLE_RATE)
    sf.write("test_noisy_input.wav",    noisy_s.squeeze(0).numpy(), SAMPLE_RATE)
    sf.write("test_cleaned_output.wav", est_s.squeeze(0).numpy(), SAMPLE_RATE)

    print("\nEvaluation completed successfully.")
    print(" - 'test_clean_target.wav'   (Label 0 clean speech)")
    print(" - 'test_noisy_input.wav'    (speech + battlefield noise)")
    print(" - 'test_cleaned_output.wav' (AI denoised output)")
    print(f"\nReproduce these WAVs with: speech index {export_idx}, "
          f"noise seed {noise_seed}")


if __name__ == "__main__":
    evaluate()