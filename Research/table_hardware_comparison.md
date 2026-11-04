### Table I-B: Hardware Execution Profile (Local vs. Cloud)

| Metric | Local Workstation | Google Colab Pro | Speedup / Impact |
| :--- | :---: | :---: | :---: |
| **Hardware** | NVIDIA Quadro T1000 Laptop | NVIDIA L4 Tensor Core GPU | Modern Ada Architecture |
| **Allocated VRAM** | 860 MB / 4.00 GB (21.5%) | 1,597 MB / 22.03 GB (7.2%) | Ample memory safety |
| **Throughput** | 5,200 – 5,600 tok/s | **29,500 – 30,100 tok/s** | **5.5× Throughput** |
| **Total Tokens Trained** | ~17.5M tokens (3,400 steps) | **245.76M tokens (15,000 steps)** | **14× Dataset Exposure** |
| **Best Perplexity** | 77.38 | **35.78** | **53.8% Perplexity Reduction** |
