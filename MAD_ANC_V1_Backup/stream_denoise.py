"""
Live streaming denoiser — B/D toggle for the demo.

Pipeline:
    mic (16 kHz mono float32)
      -> input ring buffer (2 s)
      -> every HOP: STFT -> U-Net mask -> iSTFT
      -> newest HOP samples to output ring
      -> headphones

Press:
    B  →  BYPASS : headphones hear the raw mic (before denoising)
    D  →  DENOISE: headphones hear the model output (default)
    Q  →  quit
    Ctrl+C also stops.
"""

import os
import queue
import sys
import threading
import time
import numpy as np
import soundfile as sf
import sounddevice as sd
import torch

from train import (
    SAMPLE_RATE, SEGMENT_LEN, CHECKPOINT_PATH,
    SpectralUNet, stft_mag_phase, istft_from_mag_phase,
    seed_everything, SEED,
)


# ============================================================================
# Config
# ============================================================================
INPUT_DEVICE   = "MacBook Air Microphone"    # exact name from sd.query_devices()
OUTPUT_DEVICE  = None                        # None = system default output

BLOCK          = int(SAMPLE_RATE * 0.020)    # 20 ms audio callback
HOP            = int(SAMPLE_RATE * 0.100)    # 100 ms between inferences

RECORD_WAV     = "stream_session.wav"        # None to disable
SHOW_LEVELS    = True

USE_MPS        = True                        # Apple Silicon GPU if available


# ============================================================================
# Model loading
# ============================================================================
def load_model(device):
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint '{CHECKPOINT_PATH}' not found. Train first."
        )
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    base = ckpt.get("base", 16)
    model = SpectralUNet(base=base).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, int(ckpt.get("segment_len", SEGMENT_LEN))


# ============================================================================
# Keyboard listener (macOS / Linux)
# ============================================================================
def start_keyboard_listener(state):
    """Non-blocking single-key reader. Press B, D, or Q."""
    try:
        import tty, termios, select
    except ImportError:
        print("[warn] keyboard toggle unavailable on this platform.")
        return

    def _loop():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not state["stop"]:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    c = sys.stdin.read(1).lower()
                    if c == "b" and not state["bypass"]:
                        state["bypass"] = True
                        print("\n>>> BYPASS  : headphones = RAW mic <<<\n")
                    elif c == "d" and state["bypass"]:
                        state["bypass"] = False
                        print("\n>>> DENOISE : headphones = MODEL output <<<\n")
                    elif c == "q":
                        state["stop"] = True
                        return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    threading.Thread(target=_loop, daemon=True).start()


# ============================================================================
# Streaming engine
# ============================================================================
class StreamDenoiser:

    def __init__(self, model, segment_len, device, state):
        self.model       = model
        self.segment_len = segment_len
        self.device      = device
        self.state       = state

        # Input ring: last `segment_len` samples of mic audio
        self.in_ring      = np.zeros(segment_len, dtype=np.float32)
        self.in_lock      = threading.Lock()
        self.samples_seen = 0

        # Output ring for playback
        self.out_ring   = np.zeros(segment_len * 2, dtype=np.float32)
        self.out_lock   = threading.Lock()
        self.out_write  = 0

        # Recording
        self.recording  = RECORD_WAV is not None
        self.record_q   = queue.Queue(maxsize=400)

        # Housekeeping
        self.stop           = threading.Event()
        self.in_rms_acc     = 0.0
        self.out_rms_acc    = 0.0
        self.rms_count      = 0
        self.last_infer_dur = 0.0

    # ------------------------------------------------------------- audio cb
    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(f"[audio status] {status}", file=sys.stderr)

        x = indata[:, 0].copy()

        # --- Push mic into input ring ---
        with self.in_lock:
            if frames >= self.segment_len:
                self.in_ring[:] = x[-self.segment_len:]
            else:
                self.in_ring = np.roll(self.in_ring, -frames)
                self.in_ring[-frames:] = x
            self.samples_seen += frames

        # --- Drain output ring to headphones ---
        with self.out_lock:
            L = len(self.out_ring)
            start = self.out_write % L
            end   = start + frames
            if end <= L:
                y = self.out_ring[start:end].copy()
            else:
                y = np.concatenate([self.out_ring[start:],
                                    self.out_ring[:end - L]])
            self.out_write = (self.out_write + frames) % L

        y = np.clip(y, -1.0, 1.0)
        outdata[:, 0] = y
        if outdata.shape[1] > 1:
            outdata[:, 1] = y

        if self.recording:
            try:
                self.record_q.put_nowait(y.copy())
            except queue.Full:
                pass
        if SHOW_LEVELS:
            self.in_rms_acc  += float(np.sqrt(np.mean(x * x) + 1e-12))
            self.out_rms_acc += float(np.sqrt(np.mean(y * y) + 1e-12))
            self.rms_count   += 1

    # ------------------------------------------------------------- worker
    def _worker(self):
        last_infer = 0
        while not self.stop.is_set():
            if self.samples_seen - last_infer < HOP:
                time.sleep(0.005)
                continue
            last_infer = self.samples_seen

            with self.in_lock:
                window = self.in_ring.copy()

            t0 = time.perf_counter()

            # Run model on the noisy window
            x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
            mag, phase = stft_mag_phase(x)
            with torch.no_grad():
                mask = self.model(mag.unsqueeze(1))
            est_mag = mask.squeeze(1) * mag
            est_wav = istft_from_mag_phase(est_mag, phase,
                                           length=self.segment_len)
            if self.device.type == "mps":
                torch.mps.synchronize()
            est = (est_wav.squeeze(0).squeeze(0)
                          .cpu().numpy().astype(np.float32))
            est = np.clip(est, -1.0, 1.0)

            self.last_infer_dur = (time.perf_counter() - t0) * 1000.0

            # Choose what goes to headphones
            if self.state["bypass"]:
                output_full = window            # raw mic
            else:
                output_full = est               # model output

            tail = output_full[-HOP:].astype(np.float32)

            with self.out_lock:
                L = len(self.out_ring)
                start = self.out_write % L
                end   = start + HOP
                if end <= L:
                    self.out_ring[start:end] = tail
                else:
                    first = L - start
                    self.out_ring[start:] = tail[:first]
                    self.out_ring[:end - L] = tail[first:]

    # ------------------------------------------------------------- recorder
    def _recorder(self):
        with sf.SoundFile(RECORD_WAV, mode="w",
                          samplerate=SAMPLE_RATE,
                          channels=1, subtype="PCM_16") as f:
            while not self.stop.is_set() or not self.record_q.empty():
                try:
                    block = self.record_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                f.write(block)

    # ------------------------------------------------------------- reporter
    def _reporter(self):
        while not self.stop.is_set():
            time.sleep(1.0)
            if self.rms_count == 0:
                continue
            in_rms  = self.in_rms_acc  / self.rms_count
            out_rms = self.out_rms_acc / self.rms_count
            self.in_rms_acc = self.out_rms_acc = 0.0
            self.rms_count  = 0
            in_db  = 20 * np.log10(in_rms  + 1e-8)
            out_db = 20 * np.log10(out_rms + 1e-8)
            mode   = "BYPASS " if self.state["bypass"] else "DENOISE"
            lat_ms = HOP / SAMPLE_RATE * 1000 / 2 \
                     + self.last_infer_dur \
                     + BLOCK / SAMPLE_RATE * 1000
            print(f"[{mode}] in={in_db:6.1f} dBFS | out={out_db:6.1f} dBFS "
                  f"| compute={self.last_infer_dur:5.1f} ms "
                  f"| lat~{lat_ms:4.0f} ms")

    # ------------------------------------------------------------- run
    def run(self):
        stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK,
            dtype="float32",
            channels=1,
            device=(INPUT_DEVICE, OUTPUT_DEVICE),
            callback=self._callback,
        )

        threads = [threading.Thread(target=self._worker,   daemon=True)]
        if SHOW_LEVELS:
            threads.append(threading.Thread(target=self._reporter, daemon=True))
        if self.recording:
            threads.append(threading.Thread(target=self._recorder, daemon=True))
        for t in threads:
            t.start()

        # ---- Look up the actual audio devices sounddevice opened ----
        try:
            in_dev_idx, out_dev_idx = stream.device
            in_info  = sd.query_devices(in_dev_idx, "input")
            out_info = sd.query_devices(out_dev_idx, "output")
            in_name  = f"[{in_dev_idx}] {in_info['name']}"
            out_name = f"[{out_dev_idx}] {out_info['name']}"
        except Exception as e:
            in_name  = f"unknown ({e})"
            out_name = f"unknown ({e})"

        print(f"Compute       : {self.device}")
        print(f"Input device  : {in_name}")
        print(f"Output device : {out_name}")
        print(f"Sample rate   : {SAMPLE_RATE} Hz")
        print(f"Block         : {BLOCK / SAMPLE_RATE * 1000:.0f} ms")
        print(f"Hop           : {HOP / SAMPLE_RATE * 1000:.0f} ms")
        print(f"Latency       : ~"
              f"{HOP / SAMPLE_RATE * 1000 / 2 + BLOCK / SAMPLE_RATE * 1000:.0f} ms "
              f"+ compute")
        if self.recording:
            print(f"Recording to  : {RECORD_WAV}")
        print()
        print("Keys:   B = BYPASS (raw mic)     "
              "D = DENOISE (model output)     Q = quit")
        print("Currently: DENOISE")
        print("\nSpeak into the mic. Ctrl+C to stop.\n")

        stream.start()
        try:
            while not self.state["stop"]:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            stream.stop()
            stream.close()
            self.stop.set()
            self.state["stop"] = True
            for t in threads:
                t.join(timeout=1.5)

        print("Stopped.")


# ============================================================================
# Entry point
# ============================================================================
def main():
    seed_everything(SEED)

    if USE_MPS and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model, segment_len = load_model(device)

    state = {"bypass": False, "stop": False}
    start_keyboard_listener(state)

    engine = StreamDenoiser(model, segment_len, device, state)
    engine.run()


if __name__ == "__main__":
    main()