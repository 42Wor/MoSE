# MoSE (Mixture of Smaller Experts): Fine-Grained Sparse Transformers for Resource-Constrained Hardware

A modular PyTorch implementation of a **Mixture of Smaller Experts (MoSE)** causal language model optimized for fine-grained Python code generation and **Fill-in-the-Middle (FIM)** infilling on edge workstation GPUs (such as the 4 GB NVIDIA Quadro T1000).

---

## 1. Architecture Diagram

```text
                                 [Input Token IDs]
                                         │
                         ┌───────────────▼──────────────┐
                         │   wte (Token) + wpe (Pos)    │ (d_model = 192)
                         └───────────────┬──────────────┘
                                         │
                      ┌──────────────────┴──────────────────┐
                      │    Transformer Layer (Repeated x 4) │
                      │                                     │
                      │  ┌───────────────────────────────┐  │
                      │  │   Pre-LayerNorm Attention     │  │
                      │  │   - 6 Heads (head_dim = 32)   │  │
                      │  │   - Causal Autoregressive     │  │
                      │  └───────────────┬───────────────┘  │
                      │                  │ + Residual       │
                      │  ┌───────────────▼───────────────┐  │
                      │  │        Pre-LayerNorm          │  │
                      │  └───────┬───────────────┬───────┘  │
                      │          │               │          │
                      │          │               │          │
                      │  ┌───────▼───────┐ ┌─────▼───────┐  │
                      │  │ Shared Expert │ │ Top-2 Gate  │  │
                      │  │ (Baseline FFN)│ │ Router      │  │
                      │  └───────┬───────┘ └─────┬───────┘  │
                      │          │               │          │
                      │          │     ┌─────────┴─────────┐│
                      │          │     │ Micro-Experts     ││
                      │          │     │ [E0, E1, ..., E15]││
                      │          │     │ (Top-2 Selected)  ││
                      │          │     └─────────┬─────────┘│
                      │          │               │          │
                      │          └───────┬───────┘          │
                      │                  │ + Residual       │
                      └──────────────────┼──────────────────┘
                                         ▼
                             [Final LayerNorm & Head]
                                         │
                             [Next Token Prediction]
```

---

## 2. Theoretical Motivation: Combinatorial Expressivity vs. Memory Footprint

Traditional Mixture of Experts (MoE) uses few monolithic experts (e.g., 8 experts with $d_{\text{ff}} = 4 \times d_{\text{model}}$). This incurs high parameter overhead and risks poor expert utilization on small GPUs.

**MoSE** inverts this paradigm by deploying a larger swarm of ultra-compact micro-experts ($N = 16$, $d_{\text{expert}} = 128$) alongside 1 unrouted shared base expert:
* **Combinatorial Subdomain Specialization:** Routing top-2 out of 16 micro-experts yields $\binom{16}{2} = 120$ distinct expert combinations per layer ($120^4 \approx 207\text{ Million}$ dynamic paths across 4 layers), providing vast representational capacity for code domains (control flow, syntax, standard library, algorithms, docstrings).
* **Guaranteed Syntax Invariance:** The shared expert absorbs high-frequency indentation, punctuation, and structural boilerplate, allowing the micro-experts to specialize purely in distinct semantic domains.
* **Low Active Parameter Footprint:** While total parameter count is ~5.9M, only ~3.1M parameters activate per token (46.6% sparsity), ensuring microsecond inference latency and ultra-low compute requirements.

---

## 3. Hardware Profile: Why 4 GB VRAM Suffices

* **Zero-RAM Memory-Mapped Data Loading (`numpy.memmap`):** Training data is stored as flat continuous `uint16` binary token arrays. Batches are read directly from disk page cache with zero CPU RAM accumulation.
* **Peak VRAM < 2.5 GB:** With `batch_size = 16`, `seq_len = 256`, and FP16 Automatic Mixed Precision (`torch.amp.autocast`), model weights require only ~12 MB, optimizer states ~24 MB, and activation memory < 600 MB. Over 1.5 GB of VRAM remains free.
* **Load-Balancing Auxiliary Loss:** Switch Transformer auxiliary loss prevents expert collapse and maintains balanced routing across all 16 micro-experts.

---

## 4. Step-by-Step Pipeline Execution

### Prerequisites
```bash
pip install -r requirements.txt
```

### Stage 1: Stream and Download Raw Python Data
Stream ~500 MB of deduplicated Python code from Hugging Face into compressed Parquet shards:
```bash
python data_pipeline/download_dataset.py --target_mb 500.0
```

### Stage 2: Clean Code and Apply Fill-in-the-Middle (FIM)
Filter lines > 1,000 characters and files > 50 KB. Apply FIM transformation (`<PRE>`, `<SUF>`, `<MID>`) to 50% of records, saving into 50 MB JSONL chunks:
```bash
python data_pipeline/preprocess_chunks.py --chunk_size_mb 50.0 --fim_rate 0.5
```

### Stage 3: Train BPE Tokenizer & Write Binary Memory Maps
Tokenize chunks using a 10,000-vocabulary Byte-Pair Encoding tokenizer and write flat 1D `uint16` files (`train.bin` 90%, `val.bin` 10%):
```bash
python data_pipeline/tokenize_to_bin.py --vocab_size 10000 --train_ratio 0.9
```

### Stage 4: Local Model Training
Launch FP16 training with zero-copy memory-mapped loading:
```bash
python train.py --batch_size 16 --lr 5e-4 --max_steps 15000
```
*(For quick pipeline verification without downloading, pass `--dummy_data --max_steps 10`)*.

### Stage 5: Evaluation & Micro-Expert Analysis
Benchmark FIM code infilling and generate the ASCII expert routing matrix across Python keywords:
```bash
python evaluate.py --checkpoint mose_python_model.pt
```

### Stage 6: Interactive MoSE Explorer Dashboard (Streamlit + Plotly)
Launch the interactive web dashboard for real-time token-by-token routing inspection, dynamic layer heatmaps, and publication-ready vector figures:
```bash
streamlit run app.py
```
* **Interactive Token Ribbon:** Click any token to trace its exact routing path across all 4 Transformer layers.
* **Layer-by-Layer Heatmaps:** Visualize sparse routing weights or dense softmax distributions for Layers 0–3, exportable as high-res SVGs for LaTeX figures.
* **Global Load Balancing:** Verify routing entropy and expert utilization histograms.
* **Per-Expert Vocabulary Table:** Discover which tokens each of the 16 micro-experts attracts.

