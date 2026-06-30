# Foot Interaction

Real-time foot-gesture recognition using VR trackers. The system captures 3D tracker positions via OpenVR (SteamVR), classifies gestures on the fly with a trained SVM, and visualizes everything in an interactive Pygame window. Sessions can be recorded to disk and replayed later for analysis.

---

## What it does

- **Live 3D visualization** — renders XZ (top-down), XY (front), and ZY (side) projections of one or more VR trackers.
- **Real-time gesture classification** — a rolling-window pipeline runs PCA → image encoding → SVM and confirms gestures via configurable debounce logic.
- **Session recording & playback** — captures full tracker sessions to `.npz` files and replays them with pause, seek, and speed controls.
- **ML training pipeline** — builds a labeled image dataset from raw recordings and trains an SVM classifier with grid-search cross-validation.

---

## Recognized gestures

| Label | Name  |
|-------|-------|
| 1     | One   |
| 2     | Two   |
| 3     | Three |

Raw training recordings are stored under `data/raw/1/`, `data/raw/2/`, and `data/raw/3/` by class.

---

## Requirements

- Python 3.9+
- SteamVR (only required for live OpenVR mode)

Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `numpy`, `scikit-learn`, `pygame`, `openvr`, `matplotlib`, `pillow`.

---

## Running the application

Launching without arguments opens an **interactive terminal menu** where every option starts with its default parameters:

```bash
python main.py
```

```
=============================================
          VR TRACKER VISUALIZER
=============================================
 1. OpenVR           (Real Hardware / SteamVR)
 2. Mock             (Simulated Trackers)
 3. Recording Viewer (Analyze NPZ data)
 4. Build ML Dataset
 5. Train Model
 6. Exit
---------------------------------------------
Select a mode [1-6]:
```

After each mode finishes you are returned to the menu automatically.

You can also skip the menu by passing the mode as a command-line argument:

### 1. OpenVR — real hardware

Requires SteamVR running with trackers connected and paired.

```bash
python main.py openvr
```

### 2. Mock — simulated trackers

Runs two simulated trackers (Lissajous and circular patterns). No hardware needed — useful for testing the UI and classification pipeline.

```bash
python main.py mock
```

### 3. Recording Viewer — playback

Loads a `.npz` recording file and replays it. A file-picker dialog opens at startup so you can select any recorded session.

```bash
python main.py recording
```

---

## Recording sessions

In **OpenVR** or **Mock** mode you can record the live tracker stream at any time:

| Key | Action                                 |
|-----|----------------------------------------|
| `R` | Start recording (press once to begin)  |
| `R` | Stop recording and save to disk        |

Recordings are saved as `.npz` files in the working directory with a timestamped filename. Each file stores tracker positions, rotations, and timestamps for every captured frame and can be opened directly with the **Recording Viewer**.

---

## Viewer keybindings

| Key       | Action                                   |
|-----------|------------------------------------------|
| `G`       | Toggle gesture classification on / off   |
| `W` / `S` | Increase / decrease prediction window    |
| `E` / `Q` | Increase / decrease prediction step      |
| `Z` / `X` | Increase / decrease debounce count       |
| `R`       | Start / stop recording                   |
| `Esc`     | Quit                                     |

---

## ML pipeline

### 1 — Build the base dataset

Processes every `.npz` file under `data/raw/{1,2,3}/` through the full preprocessing chain and saves a labeled image dataset.

```bash
python -m ml.pipeline
```

Options (all have defaults):

```bash
python -m ml.pipeline --size 64                                        # image resolution (default: 64)
python -m ml.pipeline --raw_dir data/raw --out_dir data/processed      # I/O directories
```

Output: `data/processed/dataset.npz` — arrays `X` (N × size × size, float32) and `y` (N,).

### 2 — Augment the dataset

Expands each sample with horizontal, vertical, and combined flips, multiplying the dataset by 4×.

```bash
python -m ml.dataset_augmented
```

Output: `data/processed/dataset_augmented.npz`

### 3 — Validate and select hyperparameters

80/20 stratified split, augments the training half only, then runs GridSearchCV over SVM hyperparameters and prints accuracy, confusion matrix, and classification report.

```bash
python -m ml.trainer_svm_validation
```

### 4 — Train the final model

Retrains on the full augmented dataset using the best hyperparameters found in the previous step.

```bash
python -m ml.trainer_svm
```

Options:

```bash
python -m ml.trainer_svm --dataset data/processed/dataset_augmented.npz \
                          --source_model ml/models/svm_model.pkl
```

Output: `ml/models/svm_model_final.pkl`

---

## Preprocessing pipeline

Every sample — at training time and at inference time — goes through the same chain:

```
N × [x, y, z]  →  PCA (2D projection)  →  normalize & center  →  encode as image  →  flatten  →  SVM
```

1. **PCA** — projects the 3D tracker trajectory onto its two principal components.
2. **Normalization** — centers, scales uniformly to `[−1, 1]`, and corrects PCA sign ambiguity.
3. **Image encoding** — rasterizes the normalized trajectory into a square float32 image (default 64 × 64) where pixel intensity encodes temporal progress.

---

## Real-time classifier

`RealtimeGestureClassifier` (`ml/realtime_classifier.py`) maintains a rolling buffer and applies the same preprocessing pipeline on every prediction step.

| Parameter         | Default | Description                                         |
|-------------------|---------|-----------------------------------------------------|
| `window_size`     | 105     | Rolling buffer length in frames                     |
| `step_size`       | 20      | Frames skipped between prediction attempts          |
| `debounce_count`  | 4       | Consecutive matching predictions required to confirm|
| `cooldown_frames` | 2       | Prediction steps blocked after a confirmed gesture  |

A gesture label is only emitted once `debounce_count` consecutive predictions agree. After confirmation the classifier enters a cooldown period to prevent repeated triggers for the same gesture.

The viewer loads `ml/models/svm_model_final.pkl` at startup. If the file is missing, classification is silently disabled.

---

## Project structure

```
foot_interaction/
├── main.py                          # Entry point with interactive menu
├── requirements.txt
├── ml/
│   ├── realtime_classifier.py       # Live SVM inference with debounce & cooldown
│   ├── trainer_svm.py               # Final model training
│   ├── trainer_svm_validation.py    # GridSearch cross-validation
│   ├── dataset_builder.py           # dataset.npz builder
│   ├── dataset_augmented.py         # 4× flip augmentation
│   └── models/                      # Saved .pkl model files
├── src/
│   ├── visualization/               # Pygame viewer and rendering utilities
│   ├── recordings/                  # Session recorder and playback engine
│   ├── tracker/                     # OpenVR, mock, and base tracker interfaces
│   └── gesture_processing/          # PCA projection and image encoding
└── data/
    ├── raw/                         # Raw recordings organized by class (1/, 2/, 3/)
    └── processed/                   # Processed datasets (.npz)
```

---

## Notes

- For live hardware, ensure SteamVR is fully initialized and trackers are paired before launching.
- Recording files are saved to the working directory by default.
- Raw training recordings must be placed in `data/raw/1/`, `data/raw/2/`, and `data/raw/3/` before running the dataset pipeline.
