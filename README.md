# LDM: Efficient Multi-Parameter Optimization for High-Performance Data Transfers

Lightning Data Mover (LDM) is a learning-driven data transfer framework that **jointly coordinates pipelining, parallelism, and concurrency** to maximize end-to-end throughput while keeping overhead low.

This repository accompanies our SC 2026 submission:

> **Unified Multi-Parameter Data Transfer Optimization for Heterogeneous Workloads**  
> (Authors), SC 2026

<!-- Add after camera-ready / public release -->
<!-- [![DOI](https://zenodo.org/badge/DOI/<YOUR_DOI>.svg)](https://doi.org/<YOUR_DOI>) -->

---

## Workflow at a glance

LDM follows a three-phase workflow:

1. **Exploration + logging (about 5 minutes)**  
   Run a short memory-to-memory transfer with random concurrency to log throughput and concurrency signals.

2. **Offline training (minutes)**  
   Configure the simulator from the exploration log and train a PPO agent offline until convergence.

3. **Production transfer (online adaptation)**  
   Run real transfers while the trained policy adaptively adjusts concurrency by stalling or reopening workers.

---

## Quick start (two‑machine setup)

| Phase | Receiver (Node B) | Sender (Node A) |
|-------|-------------------|-----------------|
| **Common prep** | <ul><li>`git clone <LDM_REPO_URL>`</li><li>`cd <LDM_REPO_DIR>`</li><li>`chmod +x setup.sh && ./setup.sh`</li><li>`chmod +x host_tune.sh && sudo ./host_tune.sh`</li><li>`source venv/bin/activate`</li><li>`pip install -r requirements.txt`</li><li>Edit **`config_receiver.py`** (IP / port etc.).</li></ul> | <ul><li>Same steps 1–6 as receiver.</li><li>Edit **`config_sender.py`** (IP / port etc.).</li></ul> |
| **Exploration run** | `python receiver.py` | <ul> <li>Generate dataset (if needed).</li> <li>Set <code>"mode": "random"</code> and a new <code>"model_version"</code> in <code>config_sender.py</code>.</li><li><code>python sender.py</code></li></ul> |
| **Offline training** | *(no action)* | `python offline.py` &nbsp;→ trains PPO (≈4 min) |
| **Production / inference** | `python receiver.py` (keep running) | 1. Set `"mode": "inference"` in `config_sender.py`.<br>2. `python sender.py` |

> **Tip :** `receiver.py` must be running **before any transfer** phase.  
> All tunables (ports, buffer sizes, max threads, etc.) live in
> `config_receiver.py` and `config_sender.py`.

---

## Dependencies

* Linux kernel ≥ 5.x  
* Python 3.9 – 3.12  
* `torch`, `numpy`, `gymnasium`, `tqdm`, `pandas`, `matplotlib`  
* (Optional) `iperf3` for link‑capacity sanity checks

All versions are pinned in **`requirements.txt`**.

---

## Reproducing paper results

1. Use **CloudLab** or **Fabric** nodes with ≥ 10 Gb s⁻¹ NICs.  
2. Follow the *Quick start* table above to train (≈8 min) and then
   run a transfer.  
3. Parse the logs (will be saved in the same directory) to replicate the figures of the paper.

---

## Citing

This section will be updated soon.
<!-- If you build on AutoMDT, please cite:

```bibtex
@inproceedings{Swargo2025AutoMDT,
  title     = {A Modular DRL Architecture for High-Performance and Low-Overhead Data Transfers},
  author    = {Rasman Mubtasim Swargo and Md Arifuzzaman and Engin Arslan},
  booktitle = {Proc. INDIS (SC Workshop)},
  year      = {2025}
}

``` -->

---

## License

MIT — see `LICENSE` for details.
