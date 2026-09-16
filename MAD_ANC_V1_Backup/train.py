"""
Military Audio Denoiser — spectral-masking U-Net training pipeline.

Model: small U-Net over the STFT magnitude spectrogram of the noisy input.
       Predicts a mask in [0, 1]. Enhanced magnitude = mask * noisy magnitude.
       Phase is reused from the noisy input (standard practice).

Loss:  L1 on magnitudes (main) + small SI-SNR term on the reconstructed
       waveform (monitoring only, not backpropped, keeps training fast).

The SNR-controlled mixer is unchanged from the previous version.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import soundfile as sf


# ============================================================================
# Constants
# ============================================================================
SAMPLE_RATE      = 16000
SEGMENT_DURATION = 2.0
SEGMENT_LEN      = int(SAMPLE_RATE * SEGMENT_DURATION)   # 32000
SEED             = 42
SNR_DB_RANGE     = (-5.0, 15.0)
MIN_PEAK         = 1e-3
NOISE_GAIN_RANGE = (0.3, 0.7)

_HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(_HERE, "mad_denoiser_model.pth")

# STFT parameters.  hop = n_fft/2 gives 50% overlap (fast); higher overlap
# gives better quality but 2x compute.
N_FFT    = 512
HOP_STFT = 256
WIN_LEN  = N_FFT
F_BINS   = N_FFT // 2 + 1                 # 257
T_FRAMES = SEGMENT_LEN // HOP_STFT + 1    # 126
F_PAD    = (8 - F_BINS   % 8) % 8         # 7   -> padded F = 264
T_PAD    = (8 - T_FRAMES % 8) % 8         # 2   -> padded T = 128


# ============================================================================
# Reproducibility
# ============================================================================
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# STFT / iSTFT helpers
# ============================================================================
def stft_mag_phase(waveform: torch.Tensor):
    """waveform: [B, T] or [B, 1, T]. Returns mag [B, F, T'], phase [B, F, T']."""
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)
    window = torch.hann_window(N_FFT, device=waveform.device)
    spec = torch.stft(
        waveform, n_fft=N_FFT, hop_length=HOP_STFT, win_length=WIN_LEN,
        window=window, center=True, return_complex=True,
    )
    return spec.abs(), spec.angle()


def istft_from_mag_phase(mag: torch.Tensor, phase: torch.Tensor,
                         length: int = SEGMENT_LEN) -> torch.Tensor:
    """mag, phase: [B, F, T']. Returns waveform [B, 1, length]."""
    spec = torch.polar(mag, phase)
    window = torch.hann_window(N_FFT, device=mag.device)
    wav = torch.istft(
        spec, n_fft=N_FFT, hop_length=HOP_STFT, win_length=WIN_LEN,
        window=window, center=True, length=length,
    )
    return wav.unsqueeze(1)


# ============================================================================
# Audio I/O — cached resamplers, mono collapse, fixed-length output
# ============================================================================
class AudioIO:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def _resampler(self, orig_sr: int):
        if orig_sr == self.sample_rate:
            return None
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_sr, self.sample_rate
            )
        return self._resamplers[orig_sr]

    def load(self, path: str, segment_len: int,
             random_crop: bool = False) -> torch.Tensor:
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            raise RuntimeError(f"Failed to read '{path}': {e}") from e

        wav = torch.from_numpy(data).T
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        rs = self._resampler(sr)
        if rs is not None:
            wav = rs(wav)

        T = wav.shape[1]
        if T < segment_len:
            wav = F.pad(wav, (0, segment_len - T))
        elif random_crop and T > segment_len:
            start = random.randint(0, T - segment_len)
            wav = wav[:, start:start + segment_len]
        else:
            wav = wav[:, :segment_len]

        return wav.contiguous()


# ============================================================================
# SNR-controlled mixer (unchanged)
# ============================================================================
def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float,
               eps: float = 1e-8):
    target_snr_lin = 10.0 ** (snr_db / 10.0)
    clean_rms = torch.sqrt(torch.mean(clean ** 2) + eps)
    noise_rms = torch.sqrt(torch.mean(noise ** 2) + eps)
    desired_noise_rms = clean_rms / (target_snr_lin ** 0.5)
    gain = desired_noise_rms / noise_rms
    noise = noise * gain
    noisy = clean + noise
    peak = torch.max(torch.abs(noisy))
    if peak > 1.0:
        noisy = noisy / peak
        clean_scaled = clean / peak
    else:
        clean_scaled = clean
    return noisy, clean_scaled


# ============================================================================
# Dataset (unchanged interface)
# ============================================================================
class MADDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str = "dataset",
                 sample_rate: int = SAMPLE_RATE,
                 duration: float = SEGMENT_DURATION,
                 snr_db_range=SNR_DB_RANGE,
                 min_peak: float = MIN_PEAK):
        self.sample_rate = sample_rate
        self.segment_len = int(sample_rate * duration)
        self.snr_db_range = snr_db_range
        self.min_peak = min_peak
        self.io = AudioIO(sample_rate=sample_rate)

        df = pd.read_csv(csv_file)
        speech_files, noise_files = [], []
        for _, row in df.iterrows():
            rel = str(row.get("path", "")).strip()
            if not rel:
                continue
            full_path = os.path.join(root_dir, rel)
            if not os.path.exists(full_path):
                continue
            try:
                label = int(row["label"])
            except (ValueError, TypeError, KeyError):
                continue

            if label == 0:
                try:
                    data, _ = sf.read(full_path, dtype="float32", always_2d=True)
                    if float(np.max(np.abs(data))) >= min_peak:
                        speech_files.append(full_path)
                except Exception:
                    continue
            elif label in (1, 2, 3, 4, 5, 6):
                noise_files.append(full_path)

        if not speech_files or not noise_files:
            raise ValueError(
                "Dataset parsing error: need Label 0 speech and Label 1-6 noise."
            )

        self.speech_files = speech_files
        self.noise_files = noise_files
        print("Dataset Loaded Successfully:")
        print(f" -> Clean Speech (Label 0, non-silent): {len(speech_files)} files")
        print(f" -> Defense Noise (Labels 1-6):          {len(noise_files)} files")

    def _load_clean(self, path: str) -> torch.Tensor:
        return self.io.load(path, self.segment_len, random_crop=True)

    def _make_noise(self) -> torch.Tensor:
        combined = torch.zeros(1, self.segment_len)
        for _ in range(random.randint(1, 3)):
            n = self.io.load(random.choice(self.noise_files),
                             self.segment_len, random_crop=True)
            combined = combined + n * random.uniform(*NOISE_GAIN_RANGE)
        peak = torch.max(torch.abs(combined))
        if peak > 1e-8:
            combined = combined / peak
        return combined

    def __len__(self) -> int:
        return len(self.speech_files)

    def __getitem__(self, idx: int):
        clean = self._load_clean(self.speech_files[idx])
        if torch.max(torch.abs(clean)) < 1e-4:
            for _ in range(5):
                clean = self._load_clean(random.choice(self.speech_files))
                if torch.max(torch.abs(clean)) >= 1e-4:
                    break
            else:
                clean = clean + 1e-6
        noise = self._make_noise()
        snr_db = random.uniform(*self.snr_db_range)
        noisy, clean_scaled = mix_at_snr(clean, noise, snr_db)
        return noisy, clean_scaled


# ============================================================================
# Model — spectral-masking U-Net
# ============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SpectralUNet(nn.Module):
    """Small U-Net over a magnitude spectrogram. Returns a sigmoid mask."""
    def __init__(self, base: int = 16):
        super().__init__()
        self.base = base
        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bottleneck = ConvBlock(base * 4, base * 4)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(base * 4 + base * 4, base * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(base * 2 + base * 2, base)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(base + base, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, mag: torch.Tensor) -> torch.Tensor:
        # mag: [B, 1, F, T] with F=257, T=126 in this project.
        F_orig, T_orig = mag.shape[-2], mag.shape[-1]
        x = F.pad(mag, (0, T_PAD, 0, F_PAD))  # -> [B, 1, 264, 128]

        e1 = self.enc1(x)                              # [B, base,   264, 128]
        e2 = self.enc2(self.pool(e1))                  # [B, 2base,  132,  64]
        e3 = self.enc3(self.pool(e2))                  # [B, 4base,   66,  32]
        b  = self.bottleneck(self.pool(e3))            # [B, 4base,   33,  16]

        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))  # [B, 2base,  66,  32]
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # [B, base,  132,  64]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # [B, base,  264, 128]
        mask = torch.sigmoid(self.out(d1))                    # [B, 1,     264, 128]

        return mask[..., :F_orig, :T_orig]                    # [B, 1, 257, 126]


# ============================================================================
# SI-SNR (monitoring only)
# ============================================================================
def si_snr(estimate: torch.Tensor, target: torch.Tensor,
           eps: float = 1e-8) -> torch.Tensor:
    e = estimate - estimate.mean(dim=-1, keepdim=True)
    t = target   - target.mean(dim=-1, keepdim=True)
    dot = (e * t).sum(dim=-1, keepdim=True)
    t_energy = (t * t).sum(dim=-1, keepdim=True) + eps
    proj = dot / t_energy * t
    noise = e - proj
    ratio = (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


# ============================================================================
# Training loop
# ============================================================================
def train():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    full_ds = MADDataset(csv_file=os.path.join(_HERE, "dataset", "training.csv"))

    n_val = max(1, int(0.1 * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    num_workers = min(4, max(1, (os.cpu_count() or 2) - 1))
    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True,
        num_workers=num_workers, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False,
        num_workers=min(2, num_workers),
        persistent_workers=(num_workers > 0),
    )

    model = SpectralUNet(base=16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    epochs, best_val = 20, float("inf")

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        running = 0.0
        for noisy, clean in train_loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            n_mag, _ = stft_mag_phase(noisy)             # [B, F, T]
            c_mag, _ = stft_mag_phase(clean)

            optimizer.zero_grad(set_to_none=True)
            mask = model(n_mag.unsqueeze(1))             # [B, 1, F, T]
            est_mag = mask.squeeze(1) * n_mag            # [B, F, T]
            loss = F.l1_loss(est_mag, c_mag)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += loss.item()
        train_loss = running / max(1, len(train_loader))

        # ---- Validate ----
        model.eval()
        val_loss, val_snr, n_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)

                n_mag, n_phase = stft_mag_phase(noisy)
                c_mag, _       = stft_mag_phase(clean)

                mask = model(n_mag.unsqueeze(1))
                est_mag = mask.squeeze(1) * n_mag
                val_loss += F.l1_loss(est_mag, c_mag).item()

                est_wav = istft_from_mag_phase(est_mag, n_phase)
                snr = si_snr(est_wav, clean.unsqueeze(1)).mean().item()
                val_snr += snr
                n_batches += 1
        val_loss /= max(1, n_batches)
        val_snr  /= max(1, n_batches)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_SI-SNR={val_snr:+.2f} dB | lr={lr:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model_state": model.state_dict(),
                 "segment_len": SEGMENT_LEN,
                 "arch": "SpectralUNet",
                 "base": 16},
                CHECKPOINT_PATH,
            )
            print("  ↳ saved new best checkpoint.")

    print(f"Training complete. Best val_loss={best_val:.4f}. "
          f"Weights saved to '{CHECKPOINT_PATH}'.")


if __name__ == "__main__":
    train()