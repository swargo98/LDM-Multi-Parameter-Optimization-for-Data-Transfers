# Chunk-size sensitivity of LDM on a FABRIC node pair

Experiments run on 2026-11-10 and 2026-11-11 · LDM commit `8742d62` plus a small configuration hook and an experiment runner (§3.2)

## 1. Summary

- **Question.** The LDM paper uses a 512 MB chunk size for all its main results; a 256 MB chunk appears only in the large-file ablation (§5.6). Does LDM's performance depend on chunk size? We tested 128, 256, 512 and 1024 MB on the four Table 2 datasets: 4 chunk sizes × 4 datasets × 5 repetitions = 80 transfers.
- **Throughput: no measurable effect.** For every dataset, throughput at the four chunk sizes is the same within run-to-run noise. One-way ANOVA p-values are 0.60 (A), 0.28 (B), 0.68 (C) and 0.12 (D).
- **Policy behaviour: a strong, consistent effect.** The PPO policy is retrained for each chunk size, and policies trained for smaller chunks open more streams. On datasets A–C the mean is 24.3 streams at 128 MB versus 20.1 at 1024 MB; on D it is 29.9 versus 22.8. In every dataset p < 10⁻¹⁷. The extra streams bring no throughput, because the aggregate rate is limited at about 37 Gb/s on this path (§5).
- **Limits on these conclusions.** They hold for one path (6 ms RTT), CUBIC congestion control, memory-to-memory transfers and the paper's per-thread caps. With n = 5, differences smaller than roughly 3 Gb/s (about 10%) on A and B could go undetected (§5).

## 2. Background: where chunk size enters LDM

`max_file_chunk_in_MB` is used in two places:

1. **Sender.** `sender.py` splits each file into chunk-sized work items. Workers pull these items from a shared queue, so a large file is sent in parallel by several streams.
2. **Simulator.** `fpp_simulator.py` sizes its simulated work units by the chunk size. The PPO policy is trained in this simulator, so **the learned policy depends on chunk size**.

The receiver does not use the chunk size. Because of point 2, each chunk size gets its own trained policy. Changing only the sender's setting under a 512 MB policy would not test the full system.

## 3. Setup

### 3.1 Testbed

| Item | Value |
|---|---|
| Testbed | FABRIC, one sender–receiver node pair, a data-plane link with 100 Gb/s nominal capacity |
| Hosts | 32 vCPU and 125 GiB RAM each, no GPU |
| Path | RTT 6.0 ms, MTU 9000, TCP buffers up to 512 MB |
| Congestion control | **CUBIC** with `fq_codel`. BBR is not available on these nodes, and enabling it would change the system-wide TCP setup for other users of the shared nodes. |
| Transfer mode | **Memory-to-memory** (`file_transfer: False`): the sender reads file names and sizes and sends zero-filled buffers; the receiver discards the data |
| Measured aggregate ceiling | about 37 Gb/s (peak 39.4 Gb/s) in both exploration runs |

The node pair is shared with other experiments. LDM ran only in windows when the nodes were confirmed otherwise idle, and the runner refused to start if it saw traffic on the data interface, I/O on the watched disks, or `fio`/`iperf3` processes.

**Why memory-to-memory.** The paper does not say whether its evaluation transfers were memory-to-memory or disk-to-disk. Its reported rates (30–42 Gb/s, and 29 Gb/s on a 1.57 TB dataset) exceed what each endpoint's single 1 TB Intel P4510 can write (about 9 Gb/s) or hold, so they were most likely memory-to-memory. Here, the nodes' NVMe drives are in use by another project. Datasets were therefore created as **sparse files** with the paper's exact file counts and sizes, which take almost no disk space.

### 3.2 Software

- LDM at commit `8742d62` (upstream `main`) with two local additions, neither of which changes LDM's algorithm:
  - `config_sender.py`: an `LDM_CONFIG_OVERRIDES` environment variable for per-run configuration overrides (chunk size, mode, model paths, per-thread cap).
  - `experiments/chunk_sweep.py`: a runner that creates the datasets, runs exploration, training and transfers in separate run directories, records results and summarises them.
- Python environment: PyTorch on CPU.
- LDM defaults kept: K = 1.02, B = 10, probing interval 5 s.

### 3.3 Datasets (paper Table 2)

| Dataset | Files | Size |
|---|---|---|
| A (uniform) | 4,096 × 256 MiB | 1024 GiB |
| B (uniform + large) | 804 × 1 GiB + 4 × 55 GiB | 1024 GiB |
| C (small) | 238,770 × uniform 100 KiB–1 MiB | 128.0 GiB |
| D (mixed) | 2 × 12 GiB + 3,872 × 10–100 MiB + 23,624 × 100 KiB–1 MiB | 245.4 GiB |

Sizes within each range were drawn uniformly at random with a fixed seed (2026). The paper lists D as 256 GB. Uniform draws within its stated ranges give 245 GiB, so the paper probably used a different size distribution.

How chunking applies to each dataset:

| Dataset | Effect of chunk size |
|---|---|
| A | Files are split only at 128 MB (2 chunks per file). |
| B | Files are split at every chunk size. |
| C | Every file is smaller than every chunk size, so chunking never applies. C therefore works as a **built-in control**. |
| D | Only the 2 large files are split at every chunk size. |

### 3.4 Per-thread caps (matching paper §5.3)

| Datasets | Cap per stream |
|---|---|
| A, B, C | 3 Gb/s (`network_limit` 3000, i.e. 3000 × 2²⁰ bit/s ≈ 3.15 Gb/s) |
| D | none (uncapped) |

The paper states 3 Gb/s for A and B on NEWY–CERN. It does not state a cap for C, but its C results (throughput ÷ concurrency ≈ 3.0 Gb/s) imply one.

### 3.5 Procedure

The paper's three-phase workflow (§4.6) was used for each cap.

1. **Exploration.** One 5-minute random-concurrency, memory-to-memory run per cap, on dataset A at 512 MB chunks. The file list was repeated 20 times so data never ran out. These runs give the simulator its inputs:

   | Run | Bandwidth | Per-stream rate | Optimal concurrency |
   |---|---|---|---|
   | 3 Gb/s cap | 37,753 Mb/s | 2,185 Mb/s | ≈ 17 |
   | Uncapped | 37,882 Mb/s | 1,858 Mb/s | ≈ 20 |

   Throughout, "optimal concurrency" means the simulator's theoretical value, bandwidth ÷ per-stream rate.

2. **Training.** `offline.py` trained one PPO policy for each combination of chunk size and cap: 4 × 2 = 8 policies. Training used the paper's convergence rule: reach 90% of the theoretical maximum reward, then continue for 1,000 episodes without improvement. All 8 runs converged.

   | Chunk | Training time, uncapped | Training time, 3 Gb/s cap |
   |---|---|---|
   | 128 MB | 1116 s (≈ 31k episodes) | 610 s |
   | 256 MB | 524 s | 531 s |
   | 512 MB | 290 s | 343 s |
   | 1024 MB | 285 s | 245 s |

   Smaller chunks need more episodes to converge. The paper reports 156 s on average (range 33–222 s) at 512 MB.

3. **Transfers.** Each run sent the dataset once in inference mode, with the matching policy choosing concurrency.
   - Design: 5 repetitions × 4 datasets × 4 chunk sizes = 80 transfers.
   - Order: within each repetition and dataset, the order of chunk sizes was shuffled with a fixed seed. This avoids confounding chunk size with time of day.
   - Pause: 10 s between transfers.
   - Timing: 2026-11-11, 01:52–06:30 and 16:13–16:51 UTC. The run was interrupted once (§5) and resumed; the interrupted transfer was discarded and rerun.

### 3.6 Metrics

- **Throughput** = dataset size × 8 ÷ transfer time, in Gb/s (10⁹ bit/s). Transfer time is the sender's own measurement, from the start of the transfer until the last chunk is done.
  - This uses the dataset's true size, not the sender's byte counter, which slightly undercounts (§5).
  - The sender's own figure differs from this by at most 0.15 Gb/s per cell.
- **Streams** = the mean of the per-second concurrency during the transfer, ignoring seconds with zero active workers.
- **Tests.** For each dataset:
  - one-way ANOVA across the four chunk sizes;
  - Kruskal–Wallis (a test that does not assume normal data) as a check;
  - Welch t-tests of each chunk size against 512 MB, the paper's default.

## 4. Results

### 4.1 Throughput (Gb/s, mean ± SD, n = 5)

| Dataset | 128 MB | 256 MB | 512 MB | 1024 MB | ANOVA p | Kruskal–Wallis p |
|---|---|---|---|---|---|---|
| A | 32.71 ± 1.47 | 30.97 ± 3.56 | 32.21 ± 1.54 | 32.59 ± 1.53 | 0.60 | 0.86 |
| B | 31.29 ± 1.96 | 32.58 ± 1.61 | 33.15 ± 0.83 | 32.58 ± 1.29 | 0.28 | 0.45 |
| C | 5.84 ± 0.05 | 5.87 ± 0.08 | 5.88 ± 0.02 | 5.87 ± 0.07 | 0.68 | 0.55 |
| D | 24.07 ± 0.44 | 23.66 ± 0.61 | 23.81 ± 0.80 | 22.74 ± 1.29 | 0.12 | 0.13 |

Welch t-tests against 512 MB:

| Dataset | 128 MB | 256 MB | 1024 MB |
|---|---|---|---|
| A | p = 0.61 | p = 0.51 | p = 0.70 |
| B | p = 0.10 | p = 0.51 | p = 0.43 |
| C | p = 0.12 | p = 0.73 | p = 0.81 |
| D | p = 0.55 | p = 0.76 | p = 0.16 |

No comparison is significant at α = 0.05, even before correcting for multiple comparisons.

- **Weakest contrasts.**
  - B at 128 MB is the lowest B mean, with p = 0.10 against 512 MB.
  - D at 1024 MB is the lowest D mean (p = 0.16 against 512 MB), pulled down by one slow run (20.7 Gb/s); without that run it would be 23.3 Gb/s.
- **A at 256 MB.** The large SD comes from one slow first-repetition run (25.5 Gb/s).

### 4.2 Streams chosen by the policy (mean ± SD across the 5 runs)

| Dataset | 128 MB | 256 MB | 512 MB | 1024 MB | ANOVA p |
|---|---|---|---|---|---|
| A | 24.3 ± 0.1 | 23.4 ± 0.1 | 20.5 ± 0.1 | 20.1 ± 0.1 | 2 × 10⁻²⁰ |
| B | 24.4 ± 0.1 | 23.5 ± 0.2 | 20.4 ± 0.1 | 20.0 ± 0.1 | 1 × 10⁻²⁰ |
| C | 24.2 ± 0.1 | 23.2 ± 0.1 | 20.6 ± 0.1 | 20.1 ± 0.1 | 9 × 10⁻²⁰ |
| D (uncapped) | 29.9 ± 0.2 | 26.6 ± 0.3 | 22.0 ± 0.3 | 22.8 ± 0.2 | 1 × 10⁻¹⁸ |

Within a single transfer, concurrency varied by only about 0.7–1.2 streams (SD). The policies settle quickly and stay near one value. The choice depends on the policy, not the dataset. In A, B and C (one cap, hence one exploration), each chunk size's policy picks the same number of streams whatever the dataset.

### 4.3 Transfer times (mean ± SD)

| Dataset | 128 MB | 256 MB | 512 MB | 1024 MB |
|---|---|---|---|---|
| A | 269 ± 12 s | 287 ± 36 s | 274 ± 13 s | 270 ± 13 s |
| B | 282 ± 17 s | 271 ± 14 s | 265 ± 7 s | 270 ± 11 s |
| C | 188 ± 2 s | 187 ± 3 s | 187 ± 1 s | 187 ± 2 s |
| D | 88 ± 2 s | 89 ± 2 s | 89 ± 3 s | 93 ± 6 s |

### 4.4 Context: the paper's CUBIC results

The paper's CUBIC results are on a different path (MICH–MASS: 35 Gb/s, 33 ms RTT, paper Table 5), so the comparison below is context only, not a reproduction.

| Dataset | Paper LDM: Gb/s @ streams | This study, 512 MB: Gb/s @ streams |
|---|---|---|
| A | 23.8 @ 27.1 | 32.2 @ 20.5 |
| B | 21.1 @ 19.5 | 33.2 @ 20.4 |
| C | 7.4 @ 3.2 | 5.9 @ 20.6 |
| D | 19.7 @ 10.5 | 23.8 @ 22.0 |

The 5× shorter RTT here plausibly explains the higher A, B and D rates under CUBIC.

**Dataset C is the notable difference.** The paper's LDM sends C with about 3 streams; our policies use 20–24 streams and are about 20% slower. C reaches the same rate at every stream count we observed (20–24). This suggests C is limited by per-file handling on the sender rather than by the network.

## 5. Caveats and limitations

1. **Aggregate ceiling of about 37 Gb/s, cause unknown.** A and B run at 31–33 Gb/s, just under this ceiling. The ceiling may come from the path or from LDM's Python sender. For example, every `send()` updates a shared dictionary through a single `multiprocessing.Manager` process. If the limit is on the sender, it could hide chunk-size effects that would appear at higher rates. An `iperf3` baseline on the link would settle this.
2. **Statistical power.** With n = 5 and the observed SDs, a two-sample comparison at α = 0.05 with 80% power detects differences of about:

   | Dataset | Detectable difference |
   |---|---|
   | A, B | 3 Gb/s (about 10%) |
   | D | 1.6 Gb/s (about 7%) |
   | C | 0.15 Gb/s (about 2.5%) |

   Smaller chunk-size effects on A, B and D cannot be ruled out.
3. **Exploration only at 512 MB.** Both explorations ran on dataset A at 512 MB chunks, one per cap, as the paper's workflow does (one exploration per path). The simulator inputs are therefore the same for all chunk sizes; chunk size reaches the policy only through the simulator's work-unit sizes.
4. **Differences between the code and the paper.** These are in upstream LDM, not introduced here:
   - The per-stream rate is estimated as the median of the top 51 per-second values, where the paper says the maximum.
   - The simulator hard-codes K = 1.002, while the configuration and paper use K = 1.02.
5. **Byte-counter race in `sender.py`.** The line `file_offsets[file_id] += sent` on a shared `multiprocessing.Array` is not atomic. When several workers send chunks of the same file at once, some updates are lost.
   - Effect: the sender's reported total falls short by up to 5.3 GiB of 1024 GiB on B at 128 MB, and the shortfall grows with more chunks per file.
   - No data is lost: a transfer ends only when every chunk has been marked done, and that count is kept under a lock.
   - Throughput in this report uses the true dataset size (§3.6).
   - The same counter feeds the throughput the policy observes, so the policy sees a slightly low rate on split files.
6. **CUBIC ramp-up.** New CUBIC connections on this path need about 8 s to reach full rate, longer than the 5 s probing interval. Each change in concurrency therefore takes more than one probe to show its full effect.
7. **Scope.**
   - One path, CUBIC only, memory-to-memory only, and only the paper's caps; disk I/O was not exercised.
   - Chunk size could matter more in other settings: a lower per-stream cap (the paper's 1.5 Gb/s for A and B on MICH–MASS), BBR, disk-to-disk transfers, or a longer RTT.
8. **Interruption.** The first sweep run stopped after 69 transfers, when the controlling session ended. The transfer in progress was discarded, and the remaining 11 ran about 10 hours later. The last repetition's results fall within the ranges of the earlier ones.

## 6. Possible follow-ups

The follow-ups below are in rough order of cost.

1. **`iperf3` baseline on the link.** This would show whether the 37 Gb/s ceiling is the path or LDM's sender.
2. **Repeat at a 1.5 Gb/s per-stream cap.** This is the paper's MICH–MASS value for A and B. The per-stream limit then binds well before the aggregate ceiling, which is where chunk size is more likely to matter. It needs 2 explorations, 4–8 trainings and about 5 hours of transfers.
3. **Per-chunk-size explorations.** Explore at each chunk size instead of only 512 MB.
4. **BBR, or disk-to-disk on NVMe.** Each needs approval first, because it changes system-wide network settings or storage that the shared nodes' other users depend on.

## 7. Data and reproduction

The data and models are on the sender node, not in this repository.

**Data locations:**

| Path | Contents |
|---|---|
| `~/ldm-runs/results.csv` | One row per transfer. The `gbps` column is the sender's own figure; recompute from the dataset size as in §3.6. |
| `~/ldm-runs/summary.csv` | Per-cell means: `gbps_mean` (corrected), `gbps_logged_mean`, streams |
| `~/ldm-runs/transfer/<dataset>/cs<chunk>/rep<NN>_<time>/` | Per transfer: console log, per-second throughput and concurrency log, cwnd log, run metadata |
| `~/ldm-runs/explore/` | The two exploration runs |
| `~/ldm-runs/train/` | Training runs. These total 2.5 GB, mostly periodic checkpoints in `models/`. |
| `~/ldm-runs/models/` | The 8 final policies (13 MB) |

**Commands**, run from `~/LDM` and only when nothing else is using the nodes:

```
venv/bin/python experiments/chunk_sweep.py datasets                          # sparse datasets in ~/ldm-data/src
venv/bin/python experiments/chunk_sweep.py prepare --host <receiver> --confirm-idle   # explorations + 8 trainings
venv/bin/python experiments/chunk_sweep.py sweep   --host <receiver> --confirm-idle   # 80 transfers, resumable
venv/bin/python experiments/chunk_sweep.py summarize
```

- **Receiver:** run `receiver.py` in memory-to-memory mode on the receiver (port 50028 by default). Start it by hand for each session and stop it afterwards.
- **Site settings:** the receiver address, the interface and disks to watch, paths the datasets must stay out of, and environment variables to drop can go in an optional `experiments/site.json`, which git ignores. The runner's docstring lists the keys.

## Appendix: per-run throughput (Gb/s, repetitions 1–5)

| Dataset | Chunk | Rep 1 | Rep 2 | Rep 3 | Rep 4 | Rep 5 |
|---|---|---|---|---|---|---|
| A | 128 | 30.6 | 34.1 | 33.7 | 31.8 | 33.4 |
| A | 256 | 25.5 | 33.4 | 29.4 | 34.0 | 32.7 |
| A | 512 | 30.7 | 30.7 | 34.1 | 32.1 | 33.4 |
| A | 1024 | 31.6 | 34.1 | 32.7 | 34.1 | 30.6 |
| B | 128 | 30.1 | 30.7 | 29.2 | 32.2 | 34.2 |
| B | 256 | 29.9 | 34.2 | 32.6 | 33.0 | 33.2 |
| B | 512 | 31.8 | 33.9 | 33.3 | 33.0 | 33.7 |
| B | 1024 | 32.7 | 30.5 | 34.1 | 33.1 | 32.5 |
| C | 128 | 5.9 | 5.8 | 5.9 | 5.8 | 5.9 |
| C | 256 | 6.0 | 5.8 | 5.9 | 5.8 | 5.9 |
| C | 512 | 5.9 | 5.9 | 5.9 | 5.9 | 5.9 |
| C | 1024 | 5.8 | 5.9 | 5.9 | 6.0 | 5.8 |
| D | 128 | 23.9 | 24.7 | 24.2 | 23.9 | 23.6 |
| D | 256 | 23.8 | 22.6 | 24.0 | 23.7 | 24.2 |
| D | 512 | 24.3 | 24.0 | 23.0 | 24.8 | 23.0 |
| D | 1024 | 23.3 | 22.3 | 23.6 | 20.7 | 23.8 |
