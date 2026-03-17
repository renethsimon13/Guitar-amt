# 🎸 Guitar Music Transcription — Transformer-based AMT Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com)
[![Dataset](https://img.shields.io/badge/Dataset-GuitarSet-blueviolet)](https://guitarset.weebly.com/)


> Automated polyphonic guitar transcription using a Transformer encoder trained on the GuitarSet dataset. Jointly predicts note onsets, MIDI pitch, and velocity from log-mel spectrograms of individual guitar strings.

---

## 📌 Overview

Automatic Music Transcription (AMT) is the task of converting audio recordings into symbolic music notation (e.g. MIDI). This project tackles AMT for guitar by:

1. Extracting per-string audio tracks from GuitarSet's hex-pickup recordings
2. Computing log-mel spectrograms as input features
3. Training a Transformer encoder to jointly predict **onset**, **pitch** (128 MIDI classes + silence token), and **velocity** at each audio frame
4. Evaluating with frame-level pitch accuracy and note-level precision / recall

---

## 🗂️ Repository Structure

```
music-transcription-guitarset/
│
├── music_transcription_guitarset.ipynb   # Main pipeline notebook (Google Colab)
└── README.md                             # This file
```

---

## 🧠 Model Architecture

A 3-layer Transformer encoder with three parallel output heads operating on every frame:

| Component | Detail |
|-----------|--------|
| Input | Log-mel spectrogram `(T=512 frames, 128 mel bins)` |
| Embedding | Linear projection → `D_MODEL=128` + learnable positional encoding |
| Encoder | 3× TransformerEncoderLayer (`N_HEADS=4`, `D_FF=256`, `dropout=0.1`) |
| Onset head | Linear → 1 logit (BCEWithLogitsLoss) |
| Pitch head | Linear → 129 logits — 128 MIDI pitches + silence (CrossEntropyLoss) |
| Velocity head | Linear → Sigmoid → scalar (MSELoss) |
| **Total params** | **~3.26M (~13 MB)** |

---

## 📦 Dataset — GuitarSet

[GuitarSet](https://guitarset.weebly.com/) contains 360 annotated guitar recordings across 6 players, 5 musical styles (Jazz, Rock, Pop, BN, SS), and multiple keys. Each recording provides 6 isolated string tracks via hex-pickup, giving **2,160 monophonic samples** (~970 minutes of audio) after processing.

| Split | Samples |
|-------|---------|
| Train | 1,728 (80%) |
| Val   | 216 (10%) |
| Test  | 216 (10%) |

**Dataset access:** GuitarSet is publicly available at [guitarset.weebly.com](https://guitarset.weebly.com/). Upload the extracted folder to Google Drive at `MyDrive/Guitarset Dataset/` before running the notebook.

---

## ⚙️ Setup

This notebook runs on **Google Colab**. No local installation required.

1. Open `music_transcription_guitarset.ipynb` in Google Colab
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Upload GuitarSet to Google Drive at `MyDrive/Guitarset Dataset/`
4. Run cells top to bottom — Cell 2 installs all dependencies automatically

**Dependencies installed by the notebook:**

```
librosa · soundfile · pretty_midi · jams
torch · torchvision · torchaudio
matplotlib · seaborn · scikit-learn · tqdm
fluidsynth / timidity (MIDI synthesis fallbacks)
```

---

## 🚀 Pipeline Walkthrough

| # | Section | Description |
|---|---------|-------------|
| 1 | Environment Check | Verify Python + Colab GPU |
| 2 | Package Installation | Install all deps (~3–5 min, once) |
| 3 | Configuration & Core Definitions | Config dataclass, model, dataset, loss, training utilities |
| 4 | Dataset Loading | Mount Drive, parallel-copy GuitarSet to local storage |
| 5 | GuitarSet Parsing | Match audio files to JAMS annotations (360/360) |
| 6 | Data Processing & Splits | Extract 2,160 string samples, 80/10/10 split, Drive backup |
| 7 | Model Initialisation | Random weight init, AdamW + CosineAnnealing setup |
| 8 | Training | 10 epochs with early stopping + checkpoint resume after disconnect |
| 9 | Visualise Training Curves | Total loss + component loss breakdown |
| 10 | Inference & Evaluation | Note-by-note comparison, frame-level accuracy, onset/pitch/velocity plots |

---

## 📊 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Sample rate | 22,050 Hz |
| Mel bins | 128 |
| Max frames | 512 |
| Batch size | 32 |
| Epochs | 10 (+ early stopping, patience=4) |
| Learning rate | 3e-4 (AdamW) |
| LR schedule | CosineAnnealing |
| Weight decay | 1e-5 |
| Loss weights | onset=1.0, pitch=10.0, velocity=1.0 |

---

## 💾 Disconnect Recovery

Colab sessions disconnect frequently. This notebook is built to survive that:

- **Processed dataset** is backed up to `MyDrive/guitarset_processed/` after Cell 6 — restored automatically on re-run, skipping the 10–15 min processing step
- **Checkpoints** are backed up to `MyDrive/guitarset_checkpoints/` every 2 epochs — training resumes from the last completed epoch automatically
- **Drive copy** (Cell 4) skips already-copied files, so re-running after a disconnect takes ~30 seconds instead of 5 minutes

---

## 📈 Results

Results from training from scratch on GuitarSet (10 epochs, no pretraining):

| Metric | Value |
|--------|-------|
| Best val loss | ~31.7 |
| Train/val gap | <1.0 (good generalisation) |
| Frame-level pitch accuracy | ~6–10% |
| Onset detection | Learns timing pattern, confidence below threshold |
| Pitch prediction | Tracks correct MIDI range and melodic contour |
| Velocity prediction | Captures average level and note boundaries |

> **Note:** Performance is limited by training from scratch without synthetic pretraining. Pitch loss is still declining at epoch 10 — training for 20 epochs and reducing the pitch loss weight from 10.0 to 5.0 in `MultiTaskLoss` are the recommended next steps.

---

## 👤 Author

**Reneth Raj Simon**  
MS Computer Science — Columbia University  
[GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-profile)
