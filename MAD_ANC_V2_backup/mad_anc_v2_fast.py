# MAD-ANC V2 FAST — GPU-ready improved training
# Complex spectral masking + larger U-Net + waveform + SI-SNR losses
# Designed for Google Colab Tesla T4.
#
# Improvements over baseline:
# 1) Complex mask (real + imaginary) instead of magnitude-only mask
# 2) Larger U-Net (base=24)
# 3) Complex + magnitude spectral losses
# 4) Waveform L1 loss
# 5) Differentiable SI-SNR loss
# 6) Random clean gain + mixed military-noise augmentation
# 7) Mixed precision (AMP) on CUDA
#
# This version intentionally uses ONE STFT resolution during training
# to keep training practical on a T4.

import os
import glob
import random
import time
import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split

# =========================================================
# 1. CONFIG
# =========================================================

SEED = 42

SAMPLE_RATE = 16000
SEGMENT_DURATION = 2.0
SEGMENT_LEN = int(SAMPLE_RATE * SEGMENT_DURATION)

BATCH_SIZE = 8
EPOCHS = 20

LR = 2e-4
WEIGHT_DECAY = 1e-4

N_FFT = 512
HOP = 256
WIN = 512

SNR_DB_RANGE = (-5.0, 15.0)
NOISE_GAIN_RANGE = (0.3, 0.7)

BASE = 24

# 0 is intentionally chosen for Colab reliability.
# Increase to 2 later only if training is stable.
NUM_WORKERS = 0

CHECKPOINT_DIR = "/content/checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR, "mad_anc_v2_fast_best.pth"
)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

PRINT_EVERY = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if DEVICE.type != "cuda":
    raise RuntimeError(
        "CUDA GPU is not available. "
        "In Colab select Runtime -> Change runtime type -> T4 GPU."
    )

torch.backends.cudnn.benchmark = True

print("=" * 60)
print("MAD-ANC V2 FAST")
print("=" * 60)
print("Device:", DEVICE)
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("=" * 60)


# =========================================================
# 2. FIND DATASET
# =========================================================

CANDIDATES = [
    "/root/.cache/kagglehub/datasets/"
    "junewookim/mad-dataset-military-audio-dataset/"
    "versions/1/MAD_dataset",

    "/content/MAD_dataset",
]

DATASET_DIR = None

for candidate in CANDIDATES:
    if os.path.exists(os.path.join(candidate, "training.csv")):
        DATASET_DIR = candidate
        break

if DATASET_DIR is None:
    for base in [
        "/root/.cache/kagglehub",
        "/content",
    ]:
        if os.path.exists(base):
            for root, dirs, files in os.walk(base):
                if (
                    "training.csv" in files
                    and os.path.basename(root) == "MAD_dataset"
                ):
                    DATASET_DIR = root
                    break
        if DATASET_DIR:
            break

if DATASET_DIR is None:
    raise FileNotFoundError(
        "MAD_dataset/training.csv not found. "
        "Run the KaggleHub download cell first."
    )

print("Dataset:", DATASET_DIR)


# =========================================================
# 3. AUDIO
# =========================================================

def load_audio(path):
    audio, sr = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )

    audio = np.mean(audio, axis=1)
    x = torch.from_numpy(audio)

    if sr != SAMPLE_RATE:
        x = torchaudio.functional.resample(
            x,
            sr,
            SAMPLE_RATE,
        )

    return x


def fix_length(x):
    if x.numel() < SEGMENT_LEN:
        x = F.pad(
            x,
            (0, SEGMENT_LEN - x.numel())
        )

    elif x.numel() > SEGMENT_LEN:
        start = random.randint(
            0,
            x.numel() - SEGMENT_LEN
        )
        x = x[
            start:start + SEGMENT_LEN
        ]

    return x


def peak_normalize(x):
    peak = x.abs().max()

    if peak > 1e-6:
        x = x / peak

    return x


def mix_at_snr(clean, noise, snr_db):
    clean_rms = torch.sqrt(
        torch.mean(clean ** 2) + 1e-8
    )

    noise_rms = torch.sqrt(
        torch.mean(noise ** 2) + 1e-8
    )

    target_noise_rms = (
        clean_rms
        / (10.0 ** (snr_db / 20.0))
    )

    noise = noise * (
        target_noise_rms
        / (noise_rms + 1e-8)
    )

    noisy = clean + noise

    peak = noisy.abs().max()

    if peak > 0.99:
        scale = 0.99 / peak
        noisy = noisy * scale
        clean = clean * scale

    return noisy, clean


# =========================================================
# 4. DATASET
# =========================================================

def resolve_path(root_dir, p):
    p = str(p).strip()

    if os.path.isabs(p) and os.path.exists(p):
        return p

    candidate = os.path.join(root_dir, p)

    if os.path.exists(candidate):
        return candidate

    for folder in ["training", "test"]:
        candidate = os.path.join(
            root_dir,
            folder,
            p
        )

        if os.path.exists(candidate):
            return candidate

    matches = glob.glob(
        os.path.join(
            root_dir,
            "**",
            os.path.basename(p)
        ),
        recursive=True,
    )

    if matches:
        return matches[0]

    return candidate


class MADDataset(Dataset):

    def __init__(
        self,
        csv_file,
        root_dir,
    ):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)

        label_col = None
        path_col = None

        for c in self.df.columns:
            if c.lower() in [
                "label",
                "class",
                "target",
            ]:
                label_col = c

            if c.lower() in [
                "path",
                "file",
                "filename",
                "filepath",
                "audio",
            ]:
                path_col = c

        if path_col is None:
            object_cols = self.df.select_dtypes(
                include=["object"]
            ).columns

            if len(object_cols):
                path_col = object_cols[0]

        if label_col is None or path_col is None:
            raise ValueError(
                "Could not identify CSV columns. "
                f"Found: {list(self.df.columns)}"
            )

        self.speech = []
        self.noise = []

        for _, row in self.df.iterrows():

            p = resolve_path(
                root_dir,
                row[path_col]
            )

            label = int(row[label_col])

            if label == 0:
                self.speech.append(p)

            elif 1 <= label <= 6:
                self.noise.append(p)

        if not self.speech:
            raise RuntimeError(
                "No speech files found."
            )

        if not self.noise:
            raise RuntimeError(
                "No noise files found."
            )

        # Verify actual files before expensive training.
        missing_speech = [
            p for p in self.speech[:10]
            if not os.path.isfile(p)
        ]

        missing_noise = [
            p for p in self.noise[:10]
            if not os.path.isfile(p)
        ]

        if missing_speech or missing_noise:
            raise FileNotFoundError(
                "Dataset paths could not be resolved correctly.\n"
                f"Example missing speech: {missing_speech[:2]}\n"
                f"Example missing noise: {missing_noise[:2]}"
            )

        print("\nDataset loaded:")
        print("  Speech files:", len(self.speech))
        print("  Noise files :", len(self.noise))

    def __len__(self):
        return len(self.speech)

    def __getitem__(self, idx):

        clean = fix_length(
            load_audio(self.speech[idx])
        )

        # Mix 1-3 random military noise recordings.
        num_noises = random.randint(1, 3)

        noise = torch.zeros(
            SEGMENT_LEN,
            dtype=torch.float32
        )

        for _ in range(num_noises):

            p = random.choice(self.noise)

            x = fix_length(
                load_audio(p)
            )

            gain = random.uniform(
                *NOISE_GAIN_RANGE
            )

            noise += gain * x

        noise = peak_normalize(noise)

        # Random SNR augmentation.
        snr_db = random.uniform(
            *SNR_DB_RANGE
        )

        # Random speech level augmentation.
        clean_gain = random.uniform(
            0.7,
            1.0
        )

        clean = clean * clean_gain

        noisy, clean = mix_at_snr(
            clean,
            noise,
            snr_db
        )

        return noisy, clean


# =========================================================
# 5. COMPLEX U-NET
# =========================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_ch,
                out_ch,
                3,
                padding=1,
            ),

            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_ch,
                out_ch,
                3,
                padding=1,
            ),

            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ComplexUNet(nn.Module):

    def __init__(self, base=24):
        super().__init__()

        self.e1 = ConvBlock(
            2,
            base
        )

        self.e2 = ConvBlock(
            base,
            base * 2
        )

        self.e3 = ConvBlock(
            base * 2,
            base * 4
        )

        self.b = ConvBlock(
            base * 4,
            base * 4
        )

        self.d3 = ConvBlock(
            base * 8,
            base * 2
        )

        self.d2 = ConvBlock(
            base * 4,
            base
        )

        self.d1 = ConvBlock(
            base * 2,
            base
        )

        # Output:
        # channel 0 = real mask
        # channel 1 = imaginary mask
        self.out = nn.Conv2d(
            base,
            2,
            1
        )

    def forward(self, x):

        e1 = self.e1(x)

        e2 = self.e2(
            F.max_pool2d(e1, 2)
        )

        e3 = self.e3(
            F.max_pool2d(e2, 2)
        )

        b = self.b(
            F.max_pool2d(e3, 2)
        )

        u3 = F.interpolate(
            b,
            size=e3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        d3 = self.d3(
            torch.cat(
                [u3, e3],
                dim=1
            )
        )

        u2 = F.interpolate(
            d3,
            size=e2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        d2 = self.d2(
            torch.cat(
                [u2, e2],
                dim=1
            )
        )

        u1 = F.interpolate(
            d2,
            size=e1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        d1 = self.d1(
            torch.cat(
                [u1, e1],
                dim=1
            )
        )

        return self.out(d1)


# =========================================================
# 6. STFT
# =========================================================

WINDOW = torch.hann_window(WIN)


def stft(x):

    return torch.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=WINDOW.to(x.device),
        return_complex=True,
    )


def istft(X, length):

    return torch.istft(
        X,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=WINDOW.to(X.device),
        length=length,
    )


# =========================================================
# 7. SI-SNR
# =========================================================

def si_snr(est, target):

    est = (
        est
        - est.mean(
            dim=-1,
            keepdim=True
        )
    )

    target = (
        target
        - target.mean(
            dim=-1,
            keepdim=True
        )
    )

    target_energy = (
        torch.sum(
            target ** 2,
            dim=-1,
            keepdim=True
        )
        + 1e-8
    )

    projection = (
        torch.sum(
            est * target,
            dim=-1,
            keepdim=True
        )
        / target_energy
    ) * target

    error = est - projection

    ratio = (
        torch.sum(
            projection ** 2,
            dim=-1
        )
        /
        (
            torch.sum(
                error ** 2,
                dim=-1
            )
            + 1e-8
        )
    )

    return 10.0 * torch.log10(
        ratio + 1e-8
    )


# =========================================================
# 8. ENHANCEMENT
# =========================================================

def enhance(model, noisy):

    X = stft(noisy)

    real = X.real
    imag = X.imag

    scale = torch.amax(
        torch.abs(X),
        dim=(-2, -1),
        keepdim=True,
    ).clamp_min(1e-5)

    inp = torch.stack(
        [
            real / scale,
            imag / scale,
        ],
        dim=1,
    )

    mask = model(inp)

    # Keep complex mask bounded.
    mask = 1.5 * torch.tanh(mask)

    Mr = mask[:, 0]
    Mi = mask[:, 1]

    Yr = (
        Mr * real
        - Mi * imag
    )

    Yi = (
        Mr * imag
        + Mi * real
    )

    Y = torch.complex(
        Yr,
        Yi
    )

    enhanced = istft(
        Y,
        noisy.shape[-1]
    )

    return enhanced


# =========================================================
# 9. LOSS
# =========================================================

def loss_function(
    enhanced,
    clean,
):

    # -----------------------------
    # Waveform L1
    # -----------------------------
    loss_wave = F.l1_loss(
        enhanced,
        clean
    )

    # -----------------------------
    # Single-resolution spectral loss
    # -----------------------------
    E = stft(enhanced)
    T = stft(clean)

    E_mag = torch.abs(E)
    T_mag = torch.abs(T)

    # Magnitude reconstruction.
    loss_mag = F.l1_loss(
        E_mag,
        T_mag
    )

    # Complex reconstruction:
    # compares both real and imaginary components,
    # so the network is explicitly trained on phase information.
    loss_complex = (
        F.l1_loss(
            E.real,
            T.real
        )
        +
        F.l1_loss(
            E.imag,
            T.imag
        )
    )

    # Log magnitude helps weaker spectral components.
    loss_logmag = F.l1_loss(
        torch.log1p(E_mag),
        torch.log1p(T_mag)
    )

    # -----------------------------
    # SI-SNR
    # -----------------------------
    loss_sisnr = -si_snr(
        enhanced,
        clean
    ).mean()

    # Weighted combined objective.
    loss = (
        0.35 * loss_complex
        + 0.20 * loss_mag
        + 0.10 * loss_logmag
        + 0.15 * loss_wave
        + 0.20 * loss_sisnr
    )

    return (
        loss,
        loss_complex.detach(),
        loss_mag.detach(),
        loss_wave.detach(),
        loss_sisnr.detach(),
    )


# =========================================================
# 10. QUICK DATA CHECK
# =========================================================

def check_dataset(ds):

    print("\nRunning one-sample data check...")

    noisy, clean = ds[0]

    print(
        "  Noisy shape:",
        tuple(noisy.shape)
    )

    print(
        "  Clean shape:",
        tuple(clean.shape)
    )

    print(
        "  Noisy finite:",
        bool(torch.isfinite(noisy).all())
    )

    print(
        "  Clean finite:",
        bool(torch.isfinite(clean).all())
    )

    if noisy.shape[0] != SEGMENT_LEN:
        raise RuntimeError(
            "Unexpected noisy segment length."
        )

    if clean.shape[0] != SEGMENT_LEN:
        raise RuntimeError(
            "Unexpected clean segment length."
        )

    print("Data check passed.")


# =========================================================
# 11. TRAINING
# =========================================================

def train():

    csv_path = os.path.join(
        DATASET_DIR,
        "training.csv"
    )

    full_ds = MADDataset(
        csv_file=csv_path,
        root_dir=DATASET_DIR,
    )

    check_dataset(full_ds)

    val_size = max(
        1,
        int(0.10 * len(full_ds))
    )

    train_size = (
        len(full_ds) - val_size
    )

    generator = torch.Generator().manual_seed(
        SEED
    )

    train_ds, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=generator,
    )

    print("\nSplit:")
    print("  Train:", len(train_ds))
    print("  Val  :", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print("\nBuilding model...")

    model = ComplexUNet(
        base=BASE
    ).to(DEVICE)

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{params:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # Modern AMP API.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True
    )

    best_val = float("inf")

    print("\nStarting training...")
    print(
        f"Epochs={EPOCHS}, "
        f"Batch={BATCH_SIZE}, "
        f"LR={LR}"
    )
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):

        model.train()

        train_loss_sum = 0.0
        num_batches = len(train_loader)

        start_time = time.time()

        for batch_idx, (
            noisy,
            clean
        ) in enumerate(
            train_loader,
            start=1
        ):

            noisy = noisy.to(
                DEVICE,
                non_blocking=True
            )

            clean = clean.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                enhanced = enhance(
                    model,
                    noisy
                )

                (
                    loss,
                    loss_complex,
                    loss_mag,
                    loss_wave,
                    loss_sisnr,
                ) = loss_function(
                    enhanced,
                    clean
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()

            if (
                batch_idx == 1
                or batch_idx % PRINT_EVERY == 0
                or batch_idx == num_batches
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                print(
                    f"Epoch {epoch:02d} | "
                    f"Batch {batch_idx:03d}/{num_batches:03d} | "
                    f"Loss {loss.item():.4f} | "
                    f"Complex {loss_complex.item():.4f} | "
                    f"Mag {loss_mag.item():.4f} | "
                    f"SI-SNR loss {loss_sisnr.item():.4f} | "
                    f"{elapsed:.1f}s"
                )

        avg_train = (
            train_loss_sum
            / num_batches
        )

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()

        val_loss_sum = 0.0
        val_sisnr_sum = 0.0
        noisy_sisnr_sum = 0.0
        val_batches = 0

        with torch.no_grad():

            for noisy, clean in val_loader:

                noisy = noisy.to(
                    DEVICE,
                    non_blocking=True
                )

                clean = clean.to(
                    DEVICE,
                    non_blocking=True
                )

                # No gradient, but still use AMP.
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    enhanced = enhance(
                        model,
                        noisy
                    )

                    (
                        vloss,
                        _,
                        _,
                        _,
                        _
                    ) = loss_function(
                        enhanced,
                        clean
                    )

                val_loss_sum += vloss.item()

                val_sisnr_sum += (
                    si_snr(
                        enhanced.float(),
                        clean.float()
                    ).mean().item()
                )

                noisy_sisnr_sum += (
                    si_snr(
                        noisy.float(),
                        clean.float()
                    ).mean().item()
                )

                val_batches += 1

        avg_val = (
            val_loss_sum
            / val_batches
        )

        avg_val_sisnr = (
            val_sisnr_sum
            / val_batches
        )

        avg_noisy_sisnr = (
            noisy_sisnr_sum
            / val_batches
        )

        scheduler.step(avg_val)

        current_lr = optimizer.param_groups[0]["lr"]

        print("-" * 60)
        print(
            f"Epoch {epoch:02d} COMPLETE | "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | "
            f"Noisy SI-SNR: {avg_noisy_sisnr:.2f} dB | "
            f"Enhanced SI-SNR: {avg_val_sisnr:.2f} dB | "
            f"Delta: "
            f"{avg_val_sisnr - avg_noisy_sisnr:+.2f} dB | "
            f"LR: {current_lr:.2e}"
        )

        # Save best model.
        if avg_val < best_val:

            best_val = avg_val

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "epoch":
                        epoch,

                    "val_loss":
                        avg_val,

                    "val_sisnr":
                        avg_val_sisnr,

                    "noisy_sisnr":
                        avg_noisy_sisnr,

                    "config": {
                        "sample_rate":
                            SAMPLE_RATE,

                        "segment_len":
                            SEGMENT_LEN,

                        "n_fft":
                            N_FFT,

                        "hop":
                            HOP,

                        "win":
                            WIN,

                        "base":
                            BASE,
                    },
                },
                CHECKPOINT_PATH,
            )

            print(
                "Saved BEST checkpoint:",
                CHECKPOINT_PATH
            )

        print("-" * 60)

    print("\nTraining finished.")
    print("Best validation loss:", best_val)
    print("Checkpoint:", CHECKPOINT_PATH)


if __name__ == "__main__":
    train()
