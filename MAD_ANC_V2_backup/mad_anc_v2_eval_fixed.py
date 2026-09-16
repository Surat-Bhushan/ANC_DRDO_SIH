# MAD-ANC V2 FAST — TEST EVALUATION
# Evaluates the trained complex-mask model on MAD test.csv.
# Run in the SAME Colab runtime where the checkpoint exists.

import os, glob, random
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# ---------------- CONFIG ----------------
SEED = 42
SAMPLE_RATE = 16000
SEGMENT_LEN = 32000
N_FFT = 512
HOP = 256
WIN = 512
BASE = 24
CHECKPOINT = "/content/checkpoints/mad_anc_v2_fast_best.pth"
NUM_EXAMPLES = 5

# Find dataset
DATASET_DIR = None
candidates = [
    "/content/MAD_dataset",
    "/content/dataset/MAD_dataset",
]
for c in candidates:
    if os.path.isfile(os.path.join(c, "test.csv")):
        DATASET_DIR = c
        break

# Also search the KaggleHub cache
if DATASET_DIR is None:
    roots = [
        "/root/.cache/kagglehub/datasets/junewookim/mad-dataset-military-audio-dataset/versions/1",
        "/root/.cache/kagglehub",
    ]
    for root in roots:
        matches = glob.glob(os.path.join(root, "**", "MAD_dataset", "test.csv"), recursive=True)
        if matches:
            DATASET_DIR = os.path.dirname(matches[0])
            break

if DATASET_DIR is None:
    raise FileNotFoundError(
        "Could not find MAD_dataset/test.csv. Run the KaggleHub download cell first."
    )

if not os.path.isfile(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}\n"
        "Make sure training finished in this same Colab runtime."
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("CUDA GPU is not available. Select T4 GPU in Colab.")

print("=" * 60)
print("MAD-ANC V2 FAST — TEST EVALUATION")
print("=" * 60)
print("Device:", DEVICE)
print("GPU:", torch.cuda.get_device_name(0))
print("Dataset:", DATASET_DIR)
print("Checkpoint:", CHECKPOINT)
print("=" * 60)

# ---------------- AUDIO ----------------
def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = np.mean(audio, axis=1)
    x = torch.from_numpy(audio)
    if sr != SAMPLE_RATE:
        x = torchaudio.functional.resample(x, sr, SAMPLE_RATE)
    return x

def fix_length(x):
    if x.numel() < SEGMENT_LEN:
        return F.pad(x, (0, SEGMENT_LEN - x.numel()))
    if x.numel() > SEGMENT_LEN:
        # deterministic evaluation: take a fixed crop from each file
        return x[:SEGMENT_LEN]
    return x

def peak_normalize(x):
    peak = x.abs().max()
    if peak > 1e-6:
        x = x / peak
    return x

def resolve_path(root_dir, p):
    p = str(p).strip()
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    for candidate in [
        os.path.join(root_dir, p),
        os.path.join(root_dir, "training", p),
        os.path.join(root_dir, "test", p),
    ]:
        if os.path.isfile(candidate):
            return candidate
    matches = glob.glob(
        os.path.join(root_dir, "**", os.path.basename(p)),
        recursive=True,
    )
    if matches:
        return matches[0]
    return os.path.join(root_dir, p)

# ---------------- TEST FILE LISTS ----------------
df = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))

label_col = next((c for c in df.columns if c.lower() in ["label", "class", "target"]), None)
path_col = next((c for c in df.columns if c.lower() in ["path", "file", "filename", "filepath", "audio"]), None)

if path_col is None:
    object_cols = df.select_dtypes(include=["object"]).columns
    if len(object_cols):
        path_col = object_cols[0]

if label_col is None or path_col is None:
    raise ValueError(f"Could not identify CSV columns. Found: {list(df.columns)}")

speech = []
noise_files_list = []

for _, row in df.iterrows():
    p = resolve_path(DATASET_DIR, row[path_col])
    label = int(row[label_col])
    if label == 0:
        speech.append(p)
    elif 1 <= label <= 6:
        noise_files_list.append(p)

print("Test speech files:", len(speech))
print("Test noise files :", len(noise_files_list))

# ---------------- MODEL ----------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class ComplexUNet(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.e1 = ConvBlock(2, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.b = ConvBlock(base * 4, base * 4)
        self.d3 = ConvBlock(base * 8, base * 2)
        self.d2 = ConvBlock(base * 4, base)
        self.d1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 2, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.b(F.max_pool2d(e3, 2))

        u3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([u3, e3], dim=1))

        u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], dim=1))

        return self.out(d1)

WINDOW = torch.hann_window(WIN, device=DEVICE)

def stft(x):
    return torch.stft(
        x, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        window=WINDOW, return_complex=True
    )

def istft(X, length):
    return torch.istft(
        X, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        window=WINDOW, length=length
    )

def enhance(model, noisy):
    X = stft(noisy)
    real, imag = X.real, X.imag

    scale = torch.amax(torch.abs(X), dim=(-2, -1), keepdim=True).clamp_min(1e-5)
    inp = torch.stack([real / scale, imag / scale], dim=1)

    mask = 1.5 * torch.tanh(model(inp))
    Mr, Mi = mask[:, 0], mask[:, 1]

    Yr = Mr * real - Mi * imag
    Yi = Mr * imag + Mi * real
    Y = torch.complex(Yr, Yi)

    return istft(Y, noisy.shape[-1])

def si_snr(est, target):
    est = est - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8
    projection = (torch.sum(est * target, dim=-1, keepdim=True) / target_energy) * target
    error = est - projection

    ratio = torch.sum(projection ** 2, dim=-1) / (
        torch.sum(error ** 2, dim=-1) + 1e-8
    )
    return 10.0 * torch.log10(ratio + 1e-8)

# ---------------- LOAD CHECKPOINT ----------------
model = ComplexUNet(BASE).to(DEVICE)

ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
if isinstance(ckpt, dict):
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
else:
    state = ckpt

model.load_state_dict(state)
model.eval()

print("Model loaded successfully.")
print("Evaluating...")

# ---------------- EVALUATION ----------------
SNR_RANGE = (-5.0, 15.0)
NOISE_GAIN_RANGE = (0.3, 0.7)

noisy_scores = []
enhanced_scores = []

example_dir = "/content/v2_test_examples"
os.makedirs(example_dir, exist_ok=True)

with torch.inference_mode():
    for idx, speech_path in enumerate(speech):
        # Make every test example reproducible.
        rng = random.Random(SEED + idx)

        clean = fix_length(load_audio(speech_path))

        num_noises = rng.randint(1, 3)
        noise = torch.zeros(SEGMENT_LEN)

        # Build a reproducible mixture of 1–3 test noise recordings.
        noise = torch.zeros(SEGMENT_LEN)
        for _ in range(num_noises):
            noise_path = rng.choice(noise_files_list)
            n = fix_length(load_audio(noise_path))
            gain = rng.uniform(*NOISE_GAIN_RANGE)
            noise += gain * n

        noise = peak_normalize(noise)

        snr_db = rng.uniform(*SNR_RANGE)

        clean_gain = rng.uniform(0.7, 1.0)
        clean = clean * clean_gain

        clean_rms = torch.sqrt(torch.mean(clean ** 2) + 1e-8)
        noise_rms = torch.sqrt(torch.mean(noise ** 2) + 1e-8)
        target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
        noise = noise * target_noise_rms / (noise_rms + 1e-8)

        noisy = clean + noise

        peak = noisy.abs().max()
        if peak > 0.99:
            scale = 0.99 / peak
            noisy = noisy * scale
            clean = clean * scale

        noisy_b = noisy.unsqueeze(0).to(DEVICE)
        clean_b = clean.unsqueeze(0).to(DEVICE)

        enhanced = enhance(model, noisy_b).squeeze(0)
        noisy_score = si_snr(noisy_b, clean_b).item()
        enhanced_score = si_snr(enhanced.unsqueeze(0), clean_b).item()

        noisy_scores.append(noisy_score)
        enhanced_scores.append(enhanced_score)

        if idx < NUM_EXAMPLES:
            sf.write(
                os.path.join(example_dir, f"{idx:03d}_clean.wav"),
                clean.cpu().numpy(),
                SAMPLE_RATE,
            )
            sf.write(
                os.path.join(example_dir, f"{idx:03d}_noisy.wav"),
                noisy.cpu().numpy(),
                SAMPLE_RATE,
            )
            sf.write(
                os.path.join(example_dir, f"{idx:03d}_enhanced.wav"),
                enhanced.cpu().numpy(),
                SAMPLE_RATE,
            )

        if (idx + 1) % 25 == 0 or idx == 0:
            print(
                f"Processed {idx + 1}/{len(speech)} | "
                f"Running noisy: {np.mean(noisy_scores):.2f} dB | "
                f"enhanced: {np.mean(enhanced_scores):.2f} dB | "
                f"delta: {np.mean(np.array(enhanced_scores) - np.array(noisy_scores)):+.2f} dB"
            )

noisy_mean = float(np.mean(noisy_scores))
enhanced_mean = float(np.mean(enhanced_scores))
delta_mean = enhanced_mean - noisy_mean

print("\n" + "=" * 60)
print("FINAL V2 TEST RESULTS")
print("=" * 60)
print(f"Test files:          {len(speech)}")
print(f"Noisy SI-SNR:        {noisy_mean:.2f} dB")
print(f"Enhanced SI-SNR:     {enhanced_mean:.2f} dB")
print(f"SI-SNR improvement:  {delta_mean:+.2f} dB")
print("=" * 60)
print("Examples saved to:", example_dir)
