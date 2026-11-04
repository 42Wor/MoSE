### Table VI: Dual-Hardware Deployment Profile (Quadro T1000 vs. NVIDIA L4)

| Metric / Resource | NVIDIA Quadro T1000 (Mobile Edge) | NVIDIA L4 (Cloud Accelerator) |
| :--- | :---: | :---: |
| **Architecture** | Turing (TU117) | Ada Lovelace (AD104) |
| **Nominal VRAM** | 4.00 GB GDDR6 | 24.00 GB GDDR6 |
| **Peak Training VRAM** | 860 MB (21.5% capacity) | 1,420 MB (5.9% capacity) |
| **Precision** | FP16 AMP (`torch.amp`) | FP16 AMP (`torch.amp`) |
| **Micro-Batch $\times$ Accumulation** | $16 \times 2$ (8,192 tok/step) | $32 \times 2$ (16,384 tok/step) |
| **Step Latency** | 38.2 ms | 7.2 ms |
| **Throughput** | 5,420 tokens/s | 30,550 tokens/s (5.6x) |
| **Training Horizon** | Steps 0 to 3,600 | Steps 3,600 to 15,000 |
| **Achieved Perplexity** | 75.11 (Step 3,600) | 23.62 (Step 15,000) |
