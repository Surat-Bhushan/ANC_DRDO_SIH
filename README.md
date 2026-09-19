# Military Audio Denoiser (MAD-ANC)

Real-time AI noise cancellation system to isolate human speech commands from heavy non-stationary battlefield noise (gunfire, artillery, vehicles) for tactical military communications.

---

## Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution](#-solution)
- [Dataset](#-dataset)
- [Directory Structure](#-directory-structure)
- [Requirements](#-requirements)
- [Model Details](#-model-details)
- [Working Pipeline](#-working-pipeline)
- [Results](#-results)
- [Live Streaming Demo](#-live-streaming-demo)



---

## 📖 Problem Statement

In battlefield environments, communication is critical for coordination and survivability. However, soldiers are constantly exposed to high-intensity, non-stationary noises such as gunfire, shelling, and vehicle engines. These noises severely degrade the intelligibility of voice commands transmitted over communication systems, leading to miscommunication and potential mission failure.

The challenge is to build a system that can isolate human speech from this complex acoustic background in **real-time**, with **low latency**, and on **resource-constrained edge devices** (laptops, Raspberry Pi 5, Jetson Nano) without relying on cloud connectivity. The software part is documented here. The edge implementation is under work.

---

## 💡 Solution

### Pipeline Overview

```
noisy waveform
   │
   ▼
STFT (n_fft=512, hop=256)  ──►  magnitude [257, 126]  +  phase [257, 126]
                                      │
                                      ▼
                              U-Net (2D conv over F×T)
                                      │
                                      ▼
                               mask [257, 126] ∈ [0,1]
                                      │
                                      ▼
                      enhanced_magnitude = mask ⊙ noisy_magnitude
                                      │
                                      ▼
                      iSTFT with ORIGINAL noisy phase
                                      │
                                      ▼
                               enhanced waveform
```

### Why spectral masking?

- The network sees **frequency structure** — speech formants and battlefield noise spectra are separable in this domain.
- The model only predicts a **bounded mask** `[0, 1]` instead of a raw waveform — a much easier learning problem.
- Phase is reused from the noisy input (humans are insensitive to small phase errors).
- Same architecture family as modern speech enhancers (Demucs, Conv-TasNet, DeepFilterNet, FRCRN).

---

## 📂 Dataset

This project uses the **MAD (Military Audio Dataset)**.

- **Source:** [Kaggle — MAD Dataset: Military Audio Dataset](https://www.kaggle.com/datasets/junewookim/mad-dataset-military-audio-dataset)
- **Overview:** ~8,075 sound samples across 7 classes, ~12 hours of audio.
- **Citation:** Kim, J.-W., Yoon, C., & Jung, H.-Y. (2024). *A Military Audio Dataset for Situational Awareness and Surveillance*. Scientific Data, 11(1), 668.

### Label Mapping

| Label | Meaning                              |
|-------|--------------------------------------|
| 0     | Communication / Human Speech (target)|
| 1     | Gunshot                              |
| 2     | Footsteps                            |
| 3     | Shelling                             |
| 4     | Vehicle                              |
| 5     | Helicopter                           |
| 6     | Fighter Jet                          |

### Dataset Setup

The full dataset (~1 GB) is **not included in this repository**. To configure the project with your own copy:

1. Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/junewookim/mad-dataset-military-audio-dataset).
2. Extract the archive.
3. Place the files under a `dataset/` directory at the project root.

The CSV files must contain at least the columns:

- `path` — relative path to the `.wav` file (from the `dataset/` folder)
- `label` — integer 0–6

Other columns (`youtube title`, `youtube url`) are ignored.

---

## 📁 Directory Structure

```
MAD_ANC_Project/
├── dataset/
│   ├── training/
│   ├── test/
│   ├── training.csv
│   └── test.csv
├── train.py                  # Model training pipeline
├── eval_test.py              # Offline evaluation + WAV export
├── stream_denoise.py         # Real-time live streaming demo
├── mad_denoiser_model.pth    # Saved checkpoint (after training)
├── requirements.txt
└── README.md
```

---

## 🛠️ Requirements

Python 3.10+ (3.11 recommended). The project deliberately avoids `torchcodec` and `FFmpeg` by using `soundfile` for audio I/O.

### requirements.txt

```
torch
torchaudio
numpy
pandas
soundfile
sounddevice
psutil
```

### Installation

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> **macOS:** grant microphone permission to your terminal at
> System Settings → Privacy & Security → Microphone, then restart the terminal.
>
> **Linux:** install PortAudio with `sudo apt install libportaudio2` (Debian/Ubuntu)
> or `sudo dnf install portaudio` (Fedora).

---

## 🧠 Model Details

| Property          | Value                                                |
|-------------------|------------------------------------------------------|
| Architecture      | Spectral-Masking U-Net                               |
| Parameters        | ~150,000                                             |
| Input             | Magnitude spectrogram, 2 s audio, `[1, 257, 126]`    |
| Output            | Soft mask, same shape as input, values in `[0, 1]`   |
| STFT              | `n_fft=512`, `hop_length=256`, Hann window           |
| Loss              | L1 between enhanced and clean magnitudes             |
| Optimizer         | Adam, `lr=1e-3`, `weight_decay=1e-5`                 |
| LR scheduler      | `ReduceLROnPlateau`, factor 0.5, patience 2          |
| Gradient clipping | `max_norm=5.0`                                       |
| Checkpoint rule   | Best validation loss across epochs                   |

The model is intentionally small enough to run in ~13 ms per inference on an Apple Silicon (MPS) GPU, enabling real-time edge deployment.

---

## 🚀 Working Pipeline

### 1. Training — `train.py`

Parses `dataset/training.csv`, builds `MADDataset`, and trains the U-Net.

**Per-sample augmentation:**
- Load a Label 0 file (clean target).
- Load 1–3 random noise files from Labels 1–6.
- Sum noise files with per-layer gain `U(0.3, 0.7)`, peak-normalize.
- Mix with clean speech at a random SNR drawn from `[-5, +15] dB`.
- Joint peak-normalize if the mixture would clip.

**To train:**
```bash
python train.py
```

Expected output:

```
Dataset Loaded Successfully:
 -> Clean Speech (Label 0, non-silent): NNN files
 -> Defense Noise (Labels 1-6):          MMM files
Training on device: cpu (or mps)
Model parameters: 150,000
Epoch 01/20 | train_loss=... | val_loss=... | val_SI-SNR=... dB | lr=1.00e-03
  ↳ saved new best checkpoint.
...
Training complete. Best val_loss=... Weights saved to 'mad_denoiser_model.pth'.
```

### 2. Offline Evaluation — `eval_test.py`

Evaluates the saved checkpoint on the held-out test set.

**Workflow:**
1. Loads `mad_denoiser_model.pth`.
2. Reads `dataset/test.csv` and separates Label 0 (speech) from Labels 1–6 (noise).
3. Evaluates **all 207 Label 0 files** — mixes each with random noise at a random SNR, runs the model, computes SI-SNR before and after.
4. Reports the **mean SI-SNR improvement** across all 207 files.
5. Prompts you for:
   - **Speech index** — which Label 0 file to export comparison WAVs for, random if space entered. 
   - **Noise seed** — fixes the noise selection for that file (reproducible), random if space entered. 
6. Exports three WAVs for the chosen file:
   - `test_clean_target.wav` — Label 0 clean speech (ground truth)
   - `test_noisy_input.wav` — speech + battlefield noise
   - `test_cleaned_output.wav` — AI denoised output

> **Note:** The mean metrics are averages across all 207 files. The exported WAVs are for **one** file (your chosen index). Re-running with the same speech index and noise seed produces byte-identical WAVs.

**To run:**
```bash
python eval_test.py
```

Example session:

```
Available Label 0 files: index 0 to 206
Preview (first 20):
   [   0] dataset/test/147/00.wav
   [   1] dataset/test/147/01.wav
   ...
Enter speech index (0-206, blank = random): 42
  → 42

Export speech file: speech_paths[42]
                  = dataset/test/147/42.wav
Enter noise seed (integer, blank = random): 7
  → noise seed: 7
...
Mean SI-SNR (noisy input): +4.94 dB
Mean SI-SNR (denoised)   : +5.83 dB
Mean improvement         : +0.89 dB
Files evaluated          : 207

Exporting comparison WAVs for: dataset/test/147/42.wav
  sampled SNR (mixture): -4.33 dB
  SI-SNR noisy/denoised: -4.25 / +0.89 dB

Evaluation completed successfully.
 - 'test_clean_target.wav'   (Label 0 clean speech)
 - 'test_noisy_input.wav'    (speech + battlefield noise)
 - 'test_cleaned_output.wav' (AI denoised output)

Reproduce these WAVs with: speech index 42, noise seed 7
```
> **Note:** One set of 3 audios produced during one session is available in this repository. 

### 3. Live Streaming — `stream_denoise.py`

Runs the model on live microphone input in real time.

**Pipeline:**

```
mic callback (20 ms blocks)
      │
      ▼
input ring buffer (last 2 s of audio)
      │
      ▼
worker thread — every 100 ms:
     1. Copy the latest 2 s window
     2. STFT → magnitude + phase
     3. U-Net → mask
     4. Apply mask → enhanced magnitude
     5. iSTFT with original phase → enhanced waveform
     6. Take the newest 100 ms of output
     7. Write to output ring buffer
      │
      ▼
output ring buffer
      │
      ▼
headphone callback (20 ms blocks)
```

**Interactive demo (B / D / Q):**

| Key | Action                                                 |
|-----|--------------------------------------------------------|
| B   | **BYPASS** — headphones play the raw mic (noisy)      |
| D   | **DENOISE** — headphones play the model output (clean)|
| Q   | Quit                                                   |

This toggle lets a demo audience hear the before/after in real time.

**To run:**
```bash
python stream_denoise.py
```

Example console output:

```
Compute       : mps
Input device  : [2] MacBook Air Microphone
Output device : [1] OnePlus Nord Buds 3r
Sample rate   : 16000 Hz
Block         : 20 ms
Hop           : 100 ms
Latency       : ~70 ms + compute

Keys:   B = BYPASS (raw mic)     D = DENOISE (model output)     Q = quit
Currently: DENOISE

Speak into the mic. Ctrl+C to stop.

[DENOISE] in= -42.3 dBFS | out= -46.5 dBFS | compute= 13.5 ms | lat~ 84 ms
[DENOISE] in= -22.0 dBFS | out= -23.5 dBFS | compute= 12.6 ms | lat~ 83 ms
>>> BYPASS  : headphones = RAW mic <<<
[BYPASS ] in= -22.3 dBFS | out= -22.3 dBFS | compute= 13.2 ms | lat~ 83 ms
>>> DENOISE : headphones = MODEL output <<<
[DENOISE] in= -42.3 dBFS | out= -46.5 dBFS | compute= 13.5 ms | lat~ 84 ms
```

**Reading the metrics:**

| Field      | Meaning                                                |
|------------|--------------------------------------------------------|
| `in=`      | Mic input level (dBFS) averaged over the last second   |
| `out=`     | Headphone output level (dBFS) averaged over 1 second   |
| `compute=` | Model inference time for the last window               |
| `lat~`     | Round-trip latency estimate: `HOP/2 + compute + BLOCK` |

**Recommended setup:**
- Wired headphones (Bluetooth adds latency and drops samples).
- Phone playing battlefield noise at 30–50 cm from the mic.
- `INPUT_DEVICE = "MacBook Air Microphone"` (force laptop mic if using Bluetooth for output).

---

## 📊 Results

### Mean performance across the held-out test set (207 Label 0 files)

| Metric                                   | Value        |
|------------------------------------------|--------------|
| Mean SI-SNR (noisy input)                | **+4.94 dB** |
| Mean SI-SNR (denoised output)            | **+5.83 dB** |
| Mean SI-SNR improvement                  | **+0.89 dB** |
| Files evaluated                          | 207          |

These are averages across all 207 Label 0 files in `test.csv`. Each file is mixed with randomly sampled battlefield noise at a random SNR between -5 and +15 dB. The improvement is positive on average — the model reduces noise.

> **Note:** The mean varies slightly between runs (~±0.15 dB) because the noise for each file is re-randomized each run. The exported per-file numbers are byte-identical if you enter the same speech index and noise seed.

### Best-case per-file result

On the specific file exported with a favorable noise seed
(`dataset/test/147/20.wav`, SNR = -4.33 dB):

| Metric                          | Value        |
|---------------------------------|--------------|
| SI-SNR (noisy input)            | -4.25 dB     |
| SI-SNR (denoised output)        | +0.89 dB     |
| **Per-file improvement**        | **+5.14 dB** |

The model takes a heavily degraded speech clip and lifts it into clean speech. The dataset-wide average is lower because many test files are not as noisy to begin with, and there is less room for improvement.

### Live latency

| Metric                           | Value        |
|----------------------------------|--------------|
| Sample rate                      | 16,000 Hz    |
| Audio block (BLOCK)              | 20 ms        |
| Inference hop (HOP)              | 100 ms       |
| Model compute (M1 GPU via MPS)   | ~13 ms       |
| Round-trip latency               | ~83 ms       |

Sub-100 ms latency is achievable on commodity laptop hardware without a dedicated GPU.

---

## ⚠️ Model Limitations

The model is a working prototype, not a production system. Known limits:

| Limitation                            | Consequence                                               |
|---------------------------------------|-----------------------------------------------------------|
| ~150k parameters                      | Lower quality than SOTA (~2–30 M params)                  |
| Magnitude-only mask, noisy phase      | Cannot correct phase distortion                           |
| Fixed 2 s window, no streaming state  | ~80 ms algorithmic latency; small chunk-boundary artifacts|
| Trained only on MAD dataset           | Weak generalization to unseen noise (HVAC, keyboard)      |
| Domain gap (`.wav` → live mic)        | Live output weaker than test-set numbers suggest          |
| Modest +0.89 dB mean SI-SNR           | Improvement is audible but not dramatic                   |
| No dereverberation                    | Room echo passes through                                  |
| Sigmoid mask only (no amplification)  | Can only suppress, cannot boost weak speech               |
| Single-channel input                  | No multi-mic beamforming                                  |

---
## V2 Improvements Over V1

V2 introduces a more advanced **complex-domain U-Net** architecture compared with the magnitude-only U-Net used in V1.

### Model Improvements

- Increased U-Net base channels from **16 → 24** for greater model capacity.
- Predicts a **complex-valued mask** instead of only a magnitude mask.
- Processes both **real and imaginary components** of the STFT.
- Enhances the complete complex spectrum before reconstruction instead of directly reusing the noisy phase.
- Uses a larger model to better handle complex and non-stationary military noise.

### Training Improvements

V2 uses a more comprehensive training objective:

- Complex spectral loss
- Magnitude loss
- Log-magnitude loss
- Waveform-domain L1 loss
- Differentiable SI-SNR loss
- AdamW optimizer
- Weight decay
- Random clean-speech gain augmentation
- Automatic Mixed Precision (AMP) for faster GPU training

### V1 vs V2

| Feature | V1 | V2 |
|---|---|---|
| U-Net Base Channels | 16 | **24** |
| Spectral Representation | Magnitude only | **Complex (Real + Imaginary)** |
| Mask | Magnitude mask | **Complex mask** |
| Phase | Reuses noisy phase | **Modified through complex masking** |
| Loss | Mainly L1 magnitude loss | **Multi-component spectral + waveform + SI-SNR losses** |
| Optimizer | Adam | **AdamW** |
| Augmentation | Basic | **Clean-speech gain augmentation** |
| GPU Training | Basic | **AMP optimized** |

### Performance Comparison

Independent evaluation on the **207 test speech files**:

| Model | Noisy SI-SNR | Enhanced SI-SNR | SI-SNR Improvement |
|---|---:|---:|---:|
| V1 | 4.94 dB | 5.83 dB | +0.89 dB |
| **V2** | 4.65 dB | **6.27 dB** | **+1.62 dB** |

V2 achieved a **+1.62 dB SI-SNR improvement** on the test set, compared with **+0.89 dB for V1**.

### Architecture Upgrade

```text
V1:
Noisy Audio
    ↓
STFT
    ↓
Magnitude
    ↓
U-Net
    ↓
Magnitude Mask
    ↓
Noisy Phase
    ↓
iSTFT
    ↓
Enhanced Audio
