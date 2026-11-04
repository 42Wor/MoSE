# Running MoSE on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![GPU](https://img.shields.io/badge/Runtime-T4%20%7C%20A100%20GPU-green)
![License](https://img.shields.io/badge/License-MIT-blue)

A complete, production-ready guide to training, evaluating, and visualizing the **Mixture of Smaller Experts (MoSE)** causal language model on [Google Colab](https://colab.research.google.com/).

---

## 1. Overview & Colab Advantages

While MoSE is designed to train on budget workstation GPUs (such as the 4 GB NVIDIA Quadro T1000), Google Colab provides a free cloud GPU environment with substantial advantages:

| Spec / Resource | Local Quadro T1000 | Colab Free Tier (NVIDIA T4) | Colab Pro (A100 / L4) |
| :--- | :--- | :--- | :--- |
| **VRAM** | 4 GB GDDR6 | 15–16 GB GDDR6 | 24–40 GB HBM2 / GDDR6 |
| **Max Batch Size** | 16 (`seq_len=256`) | 32–64 (`seq_len=256`) | 128+ (`seq_len=256`) |
| **Step Latency** | ~38 ms / step | ~12–18 ms / step | ~3–5 ms / step |
| **Training Speed** | ~8,000 tokens/sec | ~25,000 tokens/sec | ~90,000+ tokens/sec |
| **Network Speed** | Local ISP speed | 1+ Gbps datacenter pipe | 1+ Gbps datacenter pipe |
| **HuggingFace Stream** | ~10–15 mins for 500 MB | ~1–2 mins for 500 MB | < 1 min |

> [!TIP]
> **Drive FUSE Tip:** Always store the dataset binary memory maps (`train.bin`, `val.bin`) on Colab's fast local NVMe filesystem (`/content/MoSE/data/binary/`). Save checkpoints (`*.pt`) to Google Drive (`/content/drive/MyDrive/MoSE_Checkpoints/`) so your progress is preserved if the Colab session disconnects.

---

## 2. Quickstart: 3-Minute Pipeline Smoke Test

If you want to verify the model architecture, dynamic routing, and FP16 mixed precision on Colab's GPU in under 3 minutes wi                 thout downloading the full dataset:

```bash
# 1. Clone repository
!git clone https://github.com/YOUR_USERNAME/MoSE.git /content/MoSE
%cd /content/MoSE

# 2. Install dependencies
!pip install -r requirements.txt -q

# 3. Quick verification run with dummy synthetic memory maps (50 steps)
!python train.py --dummy_data --max_steps 50 --batch_size 16

# 4. Run evaluation benchmark
!python evaluate.py --checkpoint mose_python_model.pt
```

---

## 3. Cell-by-Cell Notebook Guide (Copy-Paste Ready)

Create a new notebook at [colab.research.google.com](https://colab.research.google.com/) and paste the following cells in order.

### Runtime Configuration
Before running any cells, set your runtime to GPU:
1. Navigate to **Runtime** > **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (or **A100 GPU** if using Colab Pro).
3. Click **Save**.

---

### Cell 1: Hardware & Environment Verification

Verify that PyTorch recognizes Colab's CUDA hardware and check available VRAM:

```python
# Cell 1: Check GPU environment
!nvidia-smi

import torch
print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name     : {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
```

---

### Cell 2: Mount Google Drive (For Checkpoint Persistence)

Mounting Google Drive ensures that checkpoints, tokenizers, and logs survive Colab runtime resets:

```python
# Cell 2: Mount Google Drive
from google.colab import drive
import os

drive.mount('/content/drive')

# Create persistent checkpoint backup directory on Drive
drive_backup_dir = "/content/drive/MyDrive/MoSE_Checkpoints"
os.makedirs(drive_backup_dir, exist_ok=True)
print(f"Checkpoints will be backed up to: {drive_backup_dir}")
```

---

### Cell 3: Clone Repository or Setup Project Workspace

```python
# Cell 3: Setup Project Directory
import os
import shutil

# Target directory on fast local NVMe
WORKDIR = "/content/MoSE"

if not os.path.exists(WORKDIR):
    # Option A: Clone from your GitHub repository
    # !git clone https://github.com/YOUR_USERNAME/MoSE.git {WORKDIR}

    # Option B: Or initialize fresh workspace if uploading files manually
    os.makedirs(WORKDIR, exist_ok=True)
    print(f"Created {WORKDIR}. If using manual upload, upload project files here.")

%cd {WORKDIR}
```

> [!NOTE]
> If you have not pushed MoSE to a remote Git repository, you can simply drag-and-drop the MoSE project folder (or a zipped `MoSE.zip`) into Colab's file browser pane on the left, then unpack with `!unzip -q MoSE.zip -d /content/MoSE`.

---

### Cell 4: Install Dependencies

```python
# Cell 4: Install required Python packages
!pip install -r requirements.txt -q
!pip install pyngrok -q  # Optional: for public Streamlit web app tunneling
```

---

### Cell 5: Run Data Pipeline (Download, Clean, FIM, Tokenize)

Stream real Python code from Hugging Face, apply Fill-in-the-Middle (FIM) transformation, train the code BPE tokenizer, and write binary memory maps:

```python
# Cell 5: Complete Data Pipeline

# Step 5.1: Stream ~500 MB raw Python code from Hugging Face into Parquet shards
!python data_pipeline/download_dataset.py --target_mb 500.0

# Step 5.2: Filter, clean, and apply 50% Fill-in-the-Middle (FIM) into JSONL chunks
!python data_pipeline/preprocess_chunks.py --chunk_size_mb 50.0 --fim_rate 0.5

# Step 5.3: Train 10,000-vocabulary BPE tokenizer and compile flat uint16 memory maps
!python data_pipeline/tokenize_to_bin.py --vocab_size 10000 --train_ratio 0.9

# Verify generated binary files
import os
from pathlib import Path
for p in ["data/binary/train.bin", "data/binary/val.bin", "data/binary/meta.json", "data/binary/tokenizer.json"]:
    if os.path.exists(p):
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"✓ Found {p} ({size_mb:.2f} MB)")
    else:
        print(f"✗ Missing {p}")
```

---

### Cell 6: Train MoSE Model on Colab GPU

Launch training with automatic mixed precision (FP16), cosine decay with linear warmup, Switch Transformer load-balancing auxiliary loss, and zero-copy memory-mapped loading:

```python
# Cell 6: Train MoSE
# On Colab's T4 GPU (16 GB), you can easily scale batch size to 32 or keep 16
!python train.py \
    --batch_size 32 \
    --lr 5e-4 \
    --max_steps 15000 \
    --checkpoint mose_python_model.pt

# Copy best checkpoint and tokenizer to Google Drive for safety
!cp mose_python_model.pt /content/drive/MyDrive/MoSE_Checkpoints/
!cp data/binary/tokenizer.json /content/drive/MyDrive/MoSE_Checkpoints/
!cp data/binary/meta.json /content/drive/MyDrive/MoSE_Checkpoints/
print("✓ Checkpoint and tokenizer backed up to Google Drive successfully!")
```

#### Resuming an Interrupted Training Run
If Colab disconnects or you hit the runtime limit:
```bash
# Copy checkpoint back from Google Drive (if restarting fresh runtime)
!cp /content/drive/MyDrive/MoSE_Checkpoints/mose_python_model.pt /content/MoSE/

# Resume from saved step
!python train.py --resume mose_python_model.pt --batch_size 32 --max_steps 15000
```

---

### Cell 7: Model Evaluation & Expert Routing Analysis

Evaluate code infilling performance and print the ASCII expert routing matrix across Python keywords:

```python
# Cell 7: Run evaluation benchmarks
!python evaluate.py --checkpoint mose_python_model.pt
```

Expected output includes:
1. **FIM Code Infilling Test:** The model receives `<PRE>def add(a, b):\n    <SUF>\n    return result\n<MID>` and generates `result = a + b`.
2. **ASCII Routing Matrix:** Visualizes which of the 16 micro-experts activate for keywords (`def`, `import`, `for`, `class`, `return`, `try`, `async`).

---

### Cell 8: Launch Interactive MoSE Streamlit Dashboard on Colab

You can run the interactive Streamlit dashboard (`app.py`) directly from Colab and view it in your browser using Cloudflare Tunnel or LocalTunnel:

#### Option 8A: Via Cloudflare Tunnel (No account required, recommended)
```python
# Cell 8A: Run Streamlit with Cloudflare Tunnel
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb > /dev/null

import subprocess
import time

# Launch Streamlit in background
streamlit_proc = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])
time.sleep(3)

# Launch Cloudflare Tunnel
print("Starting Cloudflare Tunnel... Click the trycloudflare.com link below when it appears:")
!cloudflared tunnel --url http://127.0.0.1:8501
```

#### Option 8B: Via LocalTunnel (Alternative)
```python
# Cell 8B: Run Streamlit with LocalTunnel
!npm install -g localtunnel -q
import subprocess
import time

subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])
time.sleep(3)

# Get your Colab public IP (required as password for localtunnel)
!curl -s https://ipv4.icanhazip.com

print("\nEnter the above IP address in the LocalTunnel endpoint page:")
!npx localtunnel --port 8501
```

---

## 4. Hyperparameter Recommendations on Colab

| Parameter | Local 4 GB GPU | Colab Free T4 (16 GB) | Colab Pro A100 (40 GB) | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `--batch_size` | `16` | `32` or `64` | `128` | T4 has 4x more VRAM; larger batches stabilize gradient estimation. |
| `--lr` | `5e-4` | `6e-4` | `8e-4` | Slightly higher LR scales well with larger batch sizes. |
| `--max_steps` | `15000` | `15000` | `25000` | Faster compute allows converging to lower perplexity. |
| `gradient_accumulation_steps` | `2` | `2` | `1` | Can be set to 1 on A100 because physical batch size is large enough. |

---

## 5. File Structure in Colab

```text
/content/
├── drive/MyDrive/MoSE_Checkpoints/   # Persistent backup on Google Drive
│   ├── mose_python_model.pt
│   └── tokenizer.json
│
└── MoSE/                             # Fast local NVMe SSD workspace
    ├── config.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    ├── app.py
    ├── requirements.txt
    ├── mose_python_model.pt
    ├── data/
    │   ├── raw/                      # Downloaded Parquet files (~500 MB)
    │   ├── chunks/                   # Preprocessed JSONL chunks with FIM
    │   └── binary/                   # Flat uint16 memory maps
    │       ├── train.bin             # ~56 MB
    │       ├── val.bin               # ~3.1 MB
    │       ├── meta.json
    │       └── tokenizer.json
    └── data_pipeline/
        ├── download_dataset.py
        ├── preprocess_chunks.py
        └── tokenize_to_bin.py
```

---

## 6. Colab Best Practices & Troubleshooting

### Keeping Your Session Alive
Google Colab may disconnect if idle. If training for several hours:
- Keep the browser tab open.
- Make sure checkpoints are saving periodically (default: every 200 steps).

### Preventing Out-of-Memory (OOM)
- While the MoSE model is ultra-compact (~5.9M total parameters, ~3.1M active parameters), memory usage on T4 is minimal (< 3 GB VRAM with `batch_size=32`).
- If you increase `--batch_size` beyond 64 or expand `seq_len` beyond 512, monitor VRAM using `!nvidia-smi`.

### Dataset Storage Notice
Do **not** place `train.bin` directly on `/content/drive/MyDrive/`. Google Drive uses a FUSE network mount; reading millions of random token slices via `np.memmap` over FUSE causes severe network latency. Keep binary files under `/content/MoSE/data/binary/` on the local VM disk.

