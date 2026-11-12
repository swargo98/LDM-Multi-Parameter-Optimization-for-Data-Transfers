#!/usr/bin/env python3
"""Chunk-size sweep for LDM: 128, 256, 512 and 1024 MB chunks.

Memory-to-memory pipeline (file_transfer=False), following the paper's setup:

  datasets   Create sparse copies of the paper's Table 2 datasets (A-D) under
             ~/ldm-data/src/. In memory-to-memory mode sender.py only reads file
             names and sizes, so sparse files cost no disk space.
  check      Pre-flight report: datasets, explorations, models, receiver, activity.
  smoke      ~10 s end-to-end test: 8 fixed streams, 16 GiB.
  explore    5-minute random-concurrency run (paper §4.6), one per per-thread cap.
  train      offline.py PPO training for one (chunk size, cap) pair, using that
             cap's exploration log. The offline simulator reads
             max_file_chunk_in_MB, so every chunk size gets its own policy.
  prepare    explore for every cap in use, then train every (chunk size, cap).
  transfer   One inference transfer of one dataset at one chunk size.
  sweep      All transfers: reps x datasets x chunk sizes (chunk order shuffled
             per rep and dataset). Results go to ~/ldm-runs/results.csv; transfers
             already recorded as ok are skipped, so a sweep can be resumed.
  summarize  Mean +/- std throughput and concurrency per (dataset, chunk size).

explore, train, prepare, transfer and sweep load the network link or the CPU. On
shared nodes this can disturb other experiments, so they refuse to run unless
--confirm-idle is given (you have checked that nothing else is using the nodes),
and before every run they also refuse if they see traffic on the data interface,
I/O on the watched disks, or fio/iperf3 processes. receiver.py must already be
running on the receiver (it serves every chunk size; it never needs restarting).

Site-specific settings go in an optional experiments/site.json, which git ignores:
  {"host": "10.0.0.2",                    receiver data-plane address (default for --host)
   "port": "50028",                       receiver port (default for --port)
   "data_iface": "eth1",                  interface to watch (default: the route to host)
   "watch_disks": ["sdb"],                block devices whose I/O blocks a run
   "forbidden_data_prefixes": ["/mnt/x"], paths the datasets must never be created under
   "watch_dirs": ["/home/me/other"],      check lists processes running under these paths
   "strip_env": ["DATA", "OUT"]}          environment variables removed for LDM's processes

Per-thread caps follow the paper (§5.3). A and B are capped (3 Gb/s on
NEWY-CERN, 1.5 Gb/s on MICH-MASS; 3 Gb/s is used here). The paper doesn't state
C's cap, but LDM's throughput / concurrency for C is 3.0 Gb/s per stream on both
paths (Tables 3-4), which points to a 3 Gb/s cap. D is uncapped. Values are in
sender.py's network_limit units (Mbit/s per thread, -1 = no cap); override with
--caps, e.g. --caps A=1500,B=1500.
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

LDM_DIR = Path(__file__).resolve().parent.parent
PYTHON = LDM_DIR / "venv" / "bin" / "python"
HOME = Path.home()
DATA_ROOT = HOME / "ldm-data" / "src"
RUNS_ROOT = HOME / "ldm-runs"
MODELS_DIR = RUNS_ROOT / "models"
RESULTS_CSV = RUNS_ROOT / "results.csv"

CHUNKS_MB = [128, 256, 512, 1024]
DEFAULT_CAPS = {"A": 3000, "B": 3000, "C": 3000, "D": -1}
EXPLORE_DATASET = "A"     # exploration only needs enough data for 5 minutes
EXPLORE_CHUNK_MB = 512    # the paper's chunk size; exploration measures the path, not chunking
EXPLORE_MULTIPLIER = 20   # sender.py's default for random mode

SITE_FILE = Path(__file__).resolve().parent / "site.json"   # optional, untracked; see docstring
SITE = {"host": None, "port": "50028", "data_iface": None, "watch_disks": [],
        "forbidden_data_prefixes": [], "watch_dirs": [], "strip_env": []}
if SITE_FILE.exists():
    SITE.update(json.loads(SITE_FILE.read_text()))

KiB, MiB, GiB = 1024, 1024 ** 2, 1024 ** 3
# Paper Table 2 (binary units: A and B come to exactly 1 TiB, C to 128 GiB).
# Each group: (file count, min size, max size); sizes are uniform in [min, max].
DATASETS = {
    "A": [(4096, 256 * MiB, 256 * MiB)],
    "B": [(804, 1 * GiB, 1 * GiB), (4, 55 * GiB, 55 * GiB)],
    "C": [(238_770, 100 * KiB, 1 * MiB)],
    "D": [(2, 12 * GiB, 12 * GiB), (3872, 10 * MiB, 100 * MiB), (23_624, 100 * KiB, 1 * MiB)],
}

RESULT_FIELDS = ["finished", "dataset", "chunk_mb", "cap", "rep", "status", "gbps",
                 "sender_mbps", "total_gib", "time_s", "cc_mean", "cc_std", "wall_s",
                 "model_version", "git", "run_dir"]
TOTAL_RE = re.compile(r"Total: ([\d.]+) GB, Time: ([\d.]+) sec, Throughput: ([\d.]+) Mbps")


# ---------------------------------------------------------------- helpers
def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def cap_tag(cap):
    return "capnone" if cap < 0 else f"cap{cap}"


def model_version(chunk, cap):
    return f"cs{chunk}_{cap_tag(cap)}"


def model_paths(mv):
    return (MODELS_DIR / f"{mv}_offline_policy.pth", MODELS_DIR / f"{mv}_offline_value.pth")


def have_model(mv):
    return all(p.exists() for p in model_paths(mv))


def git_rev():
    def git(*a):
        return subprocess.run(["git", "-C", str(LDM_DIR), *a], capture_output=True, text=True).stdout.strip()
    try:
        return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
    except OSError:
        return "unknown"


def manifest(name, root=DATA_ROOT):
    p = root / f"{name}.manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def require_dataset(name):
    m = manifest(name)
    if m is None:
        return f"dataset {name} missing (run: datasets --only {name})"
    if m.get("scale") != 1:
        return f"dataset {name} in {DATA_ROOT} is a scaled test copy (scale {m.get('scale')}); recreate with --force"
    return None


def find_exploration_log(cap):
    """Newest exploration log for this cap from a run that stopped cleanly."""
    for p in sorted((RUNS_ROOT / "explore" / cap_tag(cap)).glob("*/timed_log_network_ppo_explore_*.csv"), reverse=True):
        meta = p.parent / "run.json"
        if meta.exists() and json.loads(meta.read_text()).get("status") == "stopped-at-marker":
            return p
    return None


# ---------------------------------------------------------------- safety
def data_iface(host):
    """Interface to watch: site.json's data_iface, else the one the kernel routes host through."""
    if SITE["data_iface"]:
        return SITE["data_iface"]
    try:
        out = subprocess.run(["ip", "-o", "route", "get", str(host)], capture_output=True, text=True).stdout.split()
        return out[out.index("dev") + 1]
    except (OSError, ValueError, IndexError):
        return None


def iface_bytes(iface):
    base = Path(f"/sys/class/net/{iface}/statistics")
    return int((base / "rx_bytes").read_text()) + int((base / "tx_bytes").read_text())


def disk_sectors():
    total = 0
    for line in Path("/proc/diskstats").read_text().splitlines():
        f = line.split()
        if len(f) > 9 and f[2] in SITE["watch_disks"]:
            total += int(f[5]) + int(f[9])
    return total


def activity(host, seconds=2.0):
    """Reasons to think something else is using the link or the watched disks right now."""
    reasons = []
    for name in ("fio", "iperf3"):
        out = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True).stdout.split()
        if out:
            reasons.append(f"{name} running (pid {' '.join(out)})")
    iface = data_iface(host) if host else None
    b0, s0 = iface_bytes(iface) if iface else 0, disk_sectors()
    time.sleep(seconds)
    b1, s1 = iface_bytes(iface) if iface else 0, disk_sectors()
    net, disk = (b1 - b0) / seconds / 1e6, (s1 - s0) * 512 / seconds / 1e6
    if net > 10:
        reasons.append(f"{iface} carrying {net:.0f} MB/s")
    if disk > 10:
        reasons.append(f"{'+'.join(SITE['watch_disks'])} doing {disk:.0f} MB/s of I/O")
    return reasons


def processes_in_watch_dirs():
    found = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cwd = os.readlink(p / "cwd")
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        if any(cwd.startswith(d) for d in SITE["watch_dirs"]):
            found.append(f"pid {p.name} in {cwd}: {cmd[:90]}")
    return found


def receiver_reachable(host, port):
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


def gate(args, need_receiver=True):
    """Refuse to load the link/CPU unless the user confirmed the nodes are idle and nothing looks busy."""
    if not args.confirm_idle:
        die("this step loads the network link or the CPU. Check that nothing else is using the nodes, "
            "then rerun with --confirm-idle")
    if need_receiver:
        need_host(args)
    busy = activity(args.host)
    if busy:
        die("possible other activity: " + "; ".join(busy) + ". Not starting.")
    if need_receiver and not receiver_reachable(args.host, args.port):
        die(f"receiver {args.host}:{args.port} is not reachable. Start receiver.py there first")
    if not PYTHON.exists():
        die(f"{PYTHON} not found (create the venv first)")


# ---------------------------------------------------------------- child processes
def child_env(overrides):
    env = {k: v for k, v in os.environ.items() if k not in SITE["strip_env"]}
    env.update(PYTHONUNBUFFERED="1", MPLBACKEND="Agg", TQDM_DISABLE="1",
               LDM_CONFIG_OVERRIDES=json.dumps(overrides))
    return env


def kill_group(proc):
    """Stop the child and everything it spawned (mp workers, Manager)."""
    for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            pass


def run_child(cmd, run_dir, overrides, timeout_s, label, stop_marker=None):
    """Run cmd in its own session with run_dir as cwd; log to run_dir/console.log.

    Returns (status, seconds). status: ok | exit<N> | timeout | stopped-at-marker.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(c) for c in cmd]
    meta = {"label": label, "cmd": cmd, "overrides": overrides, "git": git_rev(), "start": stamp()}
    t0, status, last = time.time(), "ok", {"line": ""}
    marker = threading.Event()
    with open(run_dir / "console.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=run_dir, env=child_env(overrides), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)

        def pump():
            for line in proc.stdout:
                log.write(line)
                if any(k in line for k in ("Throughput @", "Average Reward", "Total:")):
                    last["line"] = line.strip()
                if stop_marker and stop_marker in line:
                    marker.set()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        next_note = t0 + 30
        try:
            while proc.poll() is None:
                if marker.is_set():
                    status = "stopped-at-marker"
                    break
                if time.time() - t0 > timeout_s:
                    status = "timeout"
                    break
                if time.time() >= next_note:
                    print(f"    [{label}] {time.time() - t0:5.0f}s  {last['line'][-100:]}", flush=True)
                    next_note += 30
                time.sleep(0.5)
        except KeyboardInterrupt:
            status = "interrupted"
        finally:
            kill_group(proc)
            reader.join(timeout=5)
    if status == "ok" and proc.returncode not in (0, None):
        status = f"exit{proc.returncode}"
    meta.update(end=stamp(), seconds=round(time.time() - t0, 1), status=status, returncode=proc.returncode)
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1))
    if status == "interrupted":
        raise KeyboardInterrupt
    return status, meta["seconds"]


def need_host(args):
    if not args.host:
        die("no receiver address: pass --host or set \"host\" in experiments/site.json")


def sender_cmd(args, dataset):
    need_host(args)
    return [PYTHON, LDM_DIR / "sender.py", "--host", args.host, "--port", args.port,
            "--data_dir", f"{DATA_ROOT / dataset}/"]


def show(cmd, overrides):
    print("    LDM_CONFIG_OVERRIDES='" + json.dumps(overrides) + "' \\\n      " + " ".join(map(str, cmd)))


# ---------------------------------------------------------------- datasets
def cmd_datasets(args):
    root = Path(args.root).expanduser().resolve()
    if str(root).startswith(tuple(SITE["forbidden_data_prefixes"])):
        die(f"{root} is under a path listed in forbidden_data_prefixes ({SITE_FILE}); choose another --root")
    root.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        groups = [[max(1, round(n * args.scale)), lo, hi] for n, lo, hi in DATASETS[name]]
        spec = {"dataset": name, "groups": groups, "seed": args.seed, "scale": args.scale}
        old, d = manifest(name, root), root / name
        if old and {k: old.get(k) for k in spec} == spec and not args.force:
            print(f"{name}: present ({old['files']} files, {old['total_bytes'] / GiB:.1f} GiB apparent); skipping")
            continue
        if d.exists() and any(d.iterdir()) and not args.force:
            die(f"{d} exists but doesn't match the spec; rerun with --force to recreate it")
        nfiles = sum(g[0] for g in groups)
        if os.statvfs(root).f_favail < nfiles + 100_000:
            die(f"not enough free inodes on {root} for {nfiles} files")
        shutil.rmtree(d, ignore_errors=True)
        (root / f"{name}.manifest.json").unlink(missing_ok=True)
        d.mkdir(parents=True)
        rng, total = random.Random(f"{args.seed}-{name}"), 0
        for g, (count, lo, hi) in enumerate(groups):
            for i in range(count):
                size = lo if lo == hi else rng.randint(lo, hi)
                fd = os.open(d / f"g{g}_{i:06d}", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                try:
                    os.ftruncate(fd, size)  # sparse: apparent size only, no blocks written
                finally:
                    os.close(fd)
                total += size
        (root / f"{name}.manifest.json").write_text(
            json.dumps({**spec, "files": nfiles, "total_bytes": total, "created": stamp()}, indent=1))
        print(f"{name}: created {nfiles} sparse files, {total / GiB:.1f} GiB apparent, in {d}")


# ---------------------------------------------------------------- explore / train
def explore(args, cap):
    mv = f"explore_{cap_tag(cap)}"
    run_dir = RUNS_ROOT / "explore" / cap_tag(cap) / stamp()
    overrides = {"mode": "random", "model_version": mv, "max_file_chunk_in_MB": EXPLORE_CHUNK_MB,
                 "network_limit": cap, "file_transfer": False, "multiplier": EXPLORE_MULTIPLIER}
    cmd = sender_cmd(args, EXPLORE_DATASET)
    print(f"explore {cap_tag(cap)} -> {run_dir}")
    problem = require_dataset(EXPLORE_DATASET)
    if args.dry_run:
        show(cmd, overrides)
        if problem:
            print(f"    (missing: {problem})")
        return
    if problem:
        die(problem)
    gate(args)
    # After its 6 random episodes, sender.py keeps sending at concurrency 2 until
    # the data runs out, so stop it as soon as the random phase ends.
    status, secs = run_child(cmd, run_dir, overrides, timeout_s=15 * 60, label=f"explore {cap_tag(cap)}",
                             stop_marker="Training finished.")
    if status != "stopped-at-marker":
        die(f"exploration ended with status {status}; see {run_dir / 'console.log'}")
    # sender.py also prints the marker after 30 s of zero throughput (e.g. receiver
    # unreachable), so require real traffic for most of the 5 minutes.
    log = run_dir / f"timed_log_network_ppo_{mv}.csv"
    rows = [r for r in csv.reader(log.open()) if len(r) >= 4] if log.exists() else []
    active = sum(1 for r in rows if float(r[2]) > 0)
    if active < 200:
        meta = json.loads((run_dir / "run.json").read_text())
        meta.update(status="invalid-exploration", active_seconds=active)
        (run_dir / "run.json").write_text(json.dumps(meta, indent=1))
        die(f"exploration had traffic in only {active} seconds; not usable. See {run_dir / 'console.log'}")
    sys.path.insert(0, str(LDM_DIR))
    from log_stats import extract_log_metrics  # what offline.py will feed the simulator
    m = extract_log_metrics(mv, base_dir=str(run_dir))
    print(f"    done in {secs:.0f}s, {active} s with traffic. Simulator inputs: bandwidth "
          f"{m['network_bandwidth']} Mbps, per-thread {m['network_throughput_per_thread']} Mbps")


def cmd_smoke(args):
    """~10 s end-to-end check: 8 fixed streams send 16 x 1 GiB sparse files (method 'probe')."""
    d = DATA_ROOT / "smoke"
    if not d.exists():
        d.mkdir(parents=True)
        for i in range(16):
            with open(d / f"s{i:02d}", "wb") as f:
                f.truncate(1 * GiB)
    run_dir = RUNS_ROOT / "smoke" / stamp()
    overrides = {"method": "probe", "fixed_probing": {"bsize": 10, "thread": 8}, "model_version": "smoke",
                 "network_limit": 3000, "file_transfer": False, "multiplier": 1, "max_file_chunk_in_MB": 512}
    cmd = sender_cmd(args, "smoke")
    print(f"smoke -> {run_dir}")
    if args.dry_run:
        show(cmd, overrides)
        return
    gate(args)
    status, secs = run_child(cmd, run_dir, overrides, timeout_s=5 * 60, label="smoke")
    row = parse_run(run_dir, "smoke")
    ok = status == "ok" and "gbps" in row
    print(f"    {'OK' if ok else 'FAILED (' + status + ')'}: {row.get('total_gib', '?')} GiB in {row.get('time_s', '?')} s "
          f"= {row.get('gbps', '?')} Gb/s with 8 streams capped at 3 Gb/s (expect about 24)")
    if not ok:
        die(f"see {run_dir / 'console.log'}")


def train(args, chunk, cap):
    mv = model_version(chunk, cap)
    run_dir = RUNS_ROOT / "train" / mv / stamp()
    overrides = {"model_version": mv, "max_file_chunk_in_MB": chunk, "network_limit": cap}
    cmd = [PYTHON, LDM_DIR / "offline.py"]
    expl = find_exploration_log(cap)
    print(f"train {mv} -> {run_dir}")
    if args.dry_run:
        show(cmd, overrides)
        print(f"    exploration log: {expl or '(none yet: produced by explore --cap %d)' % cap}")
        return
    if expl is None:
        die(f"no finished exploration for {cap_tag(cap)}; run: explore --cap {cap}")
    gate(args, need_receiver=False)
    run_dir.mkdir(parents=True)
    shutil.copy(expl, run_dir / f"timed_log_network_ppo_{mv}.csv")  # offline.py reads it from its cwd
    (run_dir / "exploration_source.txt").write_text(f"{expl}\n")
    status, secs = run_child(cmd, run_dir, overrides, timeout_s=args.train_timeout * 60, label=f"train {mv}")
    produced = [run_dir / "best_models" / p.name for p in model_paths(mv)]
    if not all(p.exists() for p in produced):
        die(f"training produced no best model ({status}); see {run_dir / 'console.log'}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for src, dst in zip(produced, model_paths(mv)):
        shutil.copy(src, dst)
    (MODELS_DIR / f"{mv}.json").write_text(json.dumps(
        {"train_run": str(run_dir), "status": status, "seconds": secs, "exploration": str(expl)}, indent=1))
    note = "" if status == "ok" else f" (training {status}: kept its best model so far)"
    print(f"    {mv} saved to {MODELS_DIR} after {secs:.0f}s{note}")


def cmd_explore(args):
    explore(args, args.cap)


def cmd_train(args):
    train(args, args.chunk, args.cap)


def cmd_prepare(args):
    caps = sorted({args.caps[d] for d in args.datasets})
    for cap in caps:
        if find_exploration_log(cap) and not args.redo:
            print(f"explore {cap_tag(cap)}: done already ({find_exploration_log(cap)})")
        else:
            explore(args, cap)
    for cap in caps:
        for chunk in args.chunks:
            mv = model_version(chunk, cap)
            if have_model(mv) and not args.redo:
                print(f"train {mv}: model exists already")
            else:
                train(args, chunk, cap)


# ---------------------------------------------------------------- transfers
def parse_run(run_dir, mv):
    row = {}
    text = (run_dir / "console.log").read_text(errors="replace") if (run_dir / "console.log").exists() else ""
    totals = TOTAL_RE.findall(text)
    if totals:
        gib, secs, mbps = map(float, totals[-1])
        row.update(total_gib=gib, time_s=secs, sender_mbps=mbps,
                   gbps=round(gib * GiB * 8 / secs / 1e9, 3) if secs else "")
    timed = run_dir / f"timed_log_network_ppo_{mv}.csv"
    if timed.exists():
        # active workers per second, ignoring start-up/shut-down seconds with none active
        cc = [float(r[3]) for r in csv.reader(timed.open()) if len(r) >= 4 and float(r[3]) > 0]
        if cc:
            row.update(cc_mean=round(statistics.mean(cc), 2),
                       cc_std=round(statistics.pstdev(cc), 2))
    return row


def append_result(row):
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    new = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def done_already():
    if not RESULTS_CSV.exists():
        return set()
    with RESULTS_CSV.open() as f:
        return {(r["dataset"], int(r["chunk_mb"]), int(r["rep"])) for r in csv.DictReader(f) if r["status"] == "ok"}


def transfer(args, dataset, chunk, rep):
    cap = args.caps[dataset]
    mv = model_version(chunk, cap)
    policy, value = model_paths(mv)
    run_dir = RUNS_ROOT / "transfer" / dataset / f"cs{chunk}" / f"rep{rep:02d}_{stamp()}"
    overrides = {"mode": "inference", "model_version": mv, "max_file_chunk_in_MB": chunk,
                 "network_limit": cap, "file_transfer": False, "multiplier": 1,
                 "inference_policy_model": str(policy), "inference_value_model": str(value)}
    cmd = sender_cmd(args, dataset)
    problems = [p for p in (require_dataset(dataset), None if have_model(mv) else f"model {mv} not trained") if p]
    print(f"transfer {dataset} cs{chunk} rep{rep} ({cap_tag(cap)}) -> {run_dir}", flush=True)
    if args.dry_run:
        show(cmd, overrides)
        for p in problems:
            print(f"    (missing: {p})")
        return None
    if problems:
        die("; ".join(problems))
    gate(args)
    status, secs = run_child(cmd, run_dir, overrides, timeout_s=args.transfer_timeout * 60,
                             label=f"{dataset} cs{chunk} rep{rep}")
    row = {"finished": stamp(), "dataset": dataset, "chunk_mb": chunk, "cap": cap, "rep": rep,
           "status": status, "wall_s": secs, "model_version": mv, "git": git_rev(),
           "run_dir": str(run_dir), **parse_run(run_dir, mv)}
    if status == "ok" and "gbps" not in row:
        row["status"] = "no-total"  # the sender exited without its final summary line
    append_result(row)
    print(f"    {row['status']}: {row.get('gbps', '?')} Gb/s, mean concurrency {row.get('cc_mean', '?')}", flush=True)
    return row


def cmd_transfer(args):
    transfer(args, args.dataset, args.chunk, args.rep)


def cmd_sweep(args):
    plan = []
    for rep in range(1, args.reps + 1):
        for ds in args.datasets:
            order = list(args.chunks)
            random.Random(f"order-{rep}-{ds}").shuffle(order)  # spread time-of-day effects
            plan += [(ds, ch, rep) for ch in order]
    skip = set() if args.no_skip else done_already()
    todo = [p for p in plan if p not in skip]
    print(f"sweep: {len(plan)} transfers planned, {len(plan) - len(todo)} already done, {len(todo)} to run")
    if not args.dry_run:
        missing = sorted({model_version(ch, args.caps[ds]) for ds, ch, _ in todo
                          if not have_model(model_version(ch, args.caps[ds]))})
        if missing:
            die("models not trained yet: " + ", ".join(missing) + " (run: prepare --confirm-idle)")
    for i, (ds, ch, rep) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] ", end="")
        row = transfer(args, ds, ch, rep)
        if row and row["status"] != "ok" and args.stop_on_error:
            die(f"stopping after a failed transfer; see {row['run_dir']}")
        if not args.dry_run and i < len(todo):
            time.sleep(args.pause)


# ---------------------------------------------------------------- reporting
def cmd_summarize(args):
    if not RESULTS_CSV.exists():
        die(f"no results yet ({RESULTS_CSV})")
    groups = {}
    with RESULTS_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["status"] == "ok" and r["gbps"]:
                groups.setdefault((r["dataset"], int(r["chunk_mb"])), []).append(r)
    # sender.py's byte counter (file_offsets[fid] += sent on a shared mp.Array) loses updates when
    # several workers send chunks of the same file, so its "Total"/"Throughput" read slightly low.
    # A transfer only ends once every chunk is marked done, so the whole dataset was sent:
    # gbps_mean uses the dataset size from its manifest / transfer time; gbps_logged_mean is sender.py's own figure.
    out = []
    print(f"{'dataset':7} {'chunk':>6} {'n':>3} {'Gb/s mean':>10} {'± std':>7} {'logged':>7} {'cc mean':>8} {'± std':>7}")
    for (ds, ch), rows in sorted(groups.items()):
        m = manifest(ds)
        logged = [float(r["gbps"]) for r in rows]
        g = [m["total_bytes"] * 8 / float(r["time_s"]) / 1e9 for r in rows] if m and all(r["time_s"] for r in rows) else logged
        c = [float(r["cc_mean"]) for r in rows if r["cc_mean"]]
        rec = {"dataset": ds, "chunk_mb": ch, "n": len(rows),
               "gbps_mean": round(statistics.mean(g), 2), "gbps_std": round(statistics.stdev(g), 2) if len(g) > 1 else 0,
               "gbps_logged_mean": round(statistics.mean(logged), 2),
               "cc_mean": round(statistics.mean(c), 1) if c else "", "cc_std": round(statistics.stdev(c), 1) if len(c) > 1 else 0}
        out.append(rec)
        print(f"{ds:7} {ch:>6} {rec['n']:>3} {rec['gbps_mean']:>10} {rec['gbps_std']:>7} {rec['gbps_logged_mean']:>7} {rec['cc_mean']:>8} {rec['cc_std']:>7}")
    with (RUNS_ROOT / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]) if out else ["dataset"])
        w.writeheader()
        w.writerows(out)
    print(f"written to {RUNS_ROOT / 'summary.csv'}")


def cmd_check(args):
    print(f"LDM {LDM_DIR} at {git_rev()}; python {'ok' if PYTHON.exists() else 'MISSING'}")
    for name in DATASETS:
        m = manifest(name)
        print(f"  dataset {name}: " + (f"{m['files']} files, {m['total_bytes'] / GiB:.1f} GiB, scale {m['scale']}" if m else "missing"))
    caps = sorted({args.caps[d] for d in DATASETS})
    for cap in caps:
        print(f"  exploration {cap_tag(cap)}: {find_exploration_log(cap) or 'not yet'}")
        print("  models " + cap_tag(cap) + ": " + ", ".join(
            f"cs{c}={'yes' if have_model(model_version(c, cap)) else 'no'}" for c in CHUNKS_MB))
    if args.host:
        print(f"  receiver {args.host}:{args.port}: {'reachable' if receiver_reachable(args.host, args.port) else 'NOT reachable'}")
    else:
        print("  receiver: not set (--host or site.json)")
    busy = activity(args.host)
    watched = [x for x in (data_iface(args.host) if args.host else None, *SITE["watch_disks"]) if x]
    print("  activity: " + ("; ".join(busy) if busy else f"none seen on {' / '.join(watched) or 'nothing watched'} in 2 s"))
    for line in processes_in_watch_dirs():
        print(f"  note: {line}")
    if RESULTS_CSV.exists():
        print(f"  results: {len(done_already())} ok transfers in {RESULTS_CSV}")


# ---------------------------------------------------------------- CLI
def parse_caps(text):
    caps = dict(DEFAULT_CAPS)
    for part in filter(None, (text or "").split(",")):
        key, val = part.split("=")
        caps[key.strip().upper()] = int(val)
    return caps


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", default=SITE["host"], help="receiver data-plane address (default: site.json)")
    common.add_argument("--port", default=str(SITE["port"]), help="receiver port (default: site.json, else 50028)")
    common.add_argument("--caps", type=parse_caps, default=dict(DEFAULT_CAPS),
                        help="per-thread caps in Mbit/s, e.g. A=1500,B=1500 (-1 = none). Default A,B,C=3000 D=-1")
    common.add_argument("--confirm-idle", action="store_true",
                        help="confirm that nothing else is using the nodes (required for steps that load link/CPU)")
    common.add_argument("--dry-run", action="store_true", help="print what would run, run nothing")
    common.add_argument("--train-timeout", type=int, default=45, help="minutes per training run")
    common.add_argument("--transfer-timeout", type=int, default=60, help="minutes per transfer")
    chunks = lambda s: [int(x) for x in s.split(",")]
    names = lambda s: [x.strip().upper() for x in s.split(",")]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("datasets", parents=[common], help="create sparse paper datasets")
    p.add_argument("--only", type=names, default=list(DATASETS))
    p.add_argument("--root", default=str(DATA_ROOT))
    p.add_argument("--scale", type=float, default=1.0, help="file-count scale, for testing only")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--force", action="store_true", help="recreate datasets that differ from the spec")
    p.set_defaults(func=cmd_datasets)

    sub.add_parser("check", parents=[common], help="pre-flight report").set_defaults(func=cmd_check)
    sub.add_parser("smoke", parents=[common], help="10-second end-to-end test").set_defaults(func=cmd_smoke)

    p = sub.add_parser("explore", parents=[common], help="5-minute exploration for one cap")
    p.add_argument("--cap", type=int, required=True)
    p.set_defaults(func=cmd_explore)

    p = sub.add_parser("train", parents=[common], help="offline training for one chunk size and cap")
    p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--cap", type=int, required=True)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("prepare", parents=[common], help="all explorations and trainings")
    p.add_argument("--chunks", type=chunks, default=CHUNKS_MB)
    p.add_argument("--datasets", type=names, default=list(DATASETS))
    p.add_argument("--redo", action="store_true", help="redo steps that already have results")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("transfer", parents=[common], help="one transfer")
    p.add_argument("--dataset", type=str.upper, required=True, choices=list(DATASETS))
    p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--rep", type=int, default=1)
    p.set_defaults(func=cmd_transfer)

    p = sub.add_parser("sweep", parents=[common], help="all transfers (resumable)")
    p.add_argument("--chunks", type=chunks, default=CHUNKS_MB)
    p.add_argument("--datasets", type=names, default=list(DATASETS))
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--pause", type=int, default=10, help="seconds between transfers")
    p.add_argument("--no-skip", action="store_true", help="rerun transfers already recorded as ok")
    p.add_argument("--stop-on-error", action="store_true")
    p.set_defaults(func=cmd_sweep)

    sub.add_parser("summarize", parents=[common], help="aggregate results.csv").set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    for key in ("datasets", "only"):
        bad = [d for d in getattr(args, key, []) if d not in DATASETS]
        if bad:
            die(f"unknown dataset(s): {', '.join(bad)}")
    args.func(args)


if __name__ == "__main__":
    main()
