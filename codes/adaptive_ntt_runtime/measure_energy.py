"""
Energy comparison for the adaptive NTT runtime, via NVML (pynvml).

There is no nvml.h in our user-space CUDA toolchain, so energy cannot be read from inside the
C++ binary. Instead this Python parent samples the GPU's energy counter around the binary while
it executes a workload `reps` times under a chosen policy.

To remove one-time setup/probe energy, we use a TWO-POINT subtraction: run the same policy at R
and 2R reps and take the difference, so any fixed per-process cost cancels:

    compute_energy_per_pass = (E(2R) - E(R)) / R
    compute_wall_per_pass   = (W(2R) - W(R)) / R   (W = the binary's CUDA-event compute wall)

Metrics reported per policy (workload D, the full-grid mix):
    runtime (ms/pass), energy (mJ/pass), energy per NTT (uJ), throughput (NTT/s),
    throughput per watt (NTT/J).
"""
import csv
import os
import re
import subprocess
import sys
import time

import pynvml

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.realpath(os.path.join(BASE, "..", ".."))
BIN = os.path.join(BASE, "adaptive_runtime")
OUT = os.path.join(PROJ, "results", "raw", "adaptive_energy_comparison.csv")

WORKLOAD = "D"          # full-grid realistic mix
R = 400                 # base reps; second point is 2R
POLICIES = ["gpu_only", "our_only", "adaptive"]

LINE_RE = re.compile(r"ntts=(\d+)\s+tasks=(\d+).*compute_wall_s=([0-9.]+)")
# stdout also has 'reps=' earlier in the line; capture robustly below.


def run_once(policy, reps):
    """Run the binary; return (energy_mJ_delta, compute_wall_s, ntts) using NVML energy counter."""
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    # Prefer the hardware total-energy counter (Volta+; Turing 2080 Ti supports it).
    try:
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)  # mJ, monotonic since driver load
        use_counter = True
    except pynvml.NVMLError:
        use_counter = False
        e0 = None

    samples = []  # (t, power_W) if we must integrate power instead
    proc = subprocess.Popen([BIN, "energy", WORKLOAD, policy, str(reps)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=PROJ, text=True)
    if not use_counter:
        while proc.poll() is None:
            try:
                p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # W
            except pynvml.NVMLError:
                p = 0.0
            samples.append((time.time(), p))
            time.sleep(0.02)
    out = proc.communicate()[0]

    m = None
    for ln in out.splitlines():
        if ln.startswith("ENERGY_RUN"):
            mm = re.search(r"ntts=(\d+).*compute_wall_s=([0-9.]+)", ln)
            ntts = int(mm.group(1)); wall = float(mm.group(2))
            m = (ntts, wall)
    if m is None:
        sys.stderr.write(out + "\n")
        raise RuntimeError(f"no ENERGY_RUN line for policy={policy} reps={reps}")
    ntts, wall = m

    if use_counter:
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        energy_mj = float(e1 - e0)
    else:
        # integrate sampled power over elapsed wall (trapezoid)
        energy_j = 0.0
        for i in range(1, len(samples)):
            dt = samples[i][0] - samples[i - 1][0]
            energy_j += 0.5 * (samples[i][1] + samples[i - 1][1]) * dt
        energy_mj = energy_j * 1000.0
    return energy_mj, wall, ntts


def main():
    pynvml.nvmlInit()
    name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
    if isinstance(name, bytes):
        name = name.decode()
    try:
        pynvml.nvmlDeviceGetTotalEnergyConsumption(pynvml.nvmlDeviceGetHandleByIndex(0))
        method = "nvmlDeviceGetTotalEnergyConsumption"
    except pynvml.NVMLError:
        method = "power-integration (counter unsupported)"
    print(f"GPU: {name} | energy method: {method} | workload {WORKLOAD}, R={R}/{2*R}")

    rows = []
    for pol in POLICIES:
        e_r, w_r, ntts = run_once(pol, R)
        e_2r, w_2r, _ = run_once(pol, 2 * R)
        d_energy_mj = (e_2r - e_r) / R          # mJ per workload pass
        d_wall_s = (w_2r - w_r) / R             # s per workload pass
        if d_wall_s <= 0:
            d_wall_s = w_r / R                  # fallback if counter noise made the diff <=0
        per_ntt_uj = (d_energy_mj * 1000.0) / ntts          # uJ per NTT
        power_w = (d_energy_mj / 1000.0) / d_wall_s
        thr_ntts_s = ntts / d_wall_s
        thr_per_watt = thr_ntts_s / power_w if power_w > 0 else float("nan")
        rows.append({
            "policy": pol,
            "workload": WORKLOAD,
            "ntts_per_pass": ntts,
            "runtime_ms_per_pass": f"{d_wall_s*1000:.4f}",
            "energy_mj_per_pass": f"{d_energy_mj:.3f}",
            "avg_power_w": f"{power_w:.2f}",
            "energy_per_ntt_uj": f"{per_ntt_uj:.4f}",
            "throughput_ntts_per_s": f"{thr_ntts_s:.1f}",
            "throughput_per_watt_ntts_per_j": f"{thr_per_watt:.2f}",
        })
        print(f"{pol:9s} runtime={d_wall_s*1000:8.4f} ms  energy={d_energy_mj:8.2f} mJ  "
              f"power={power_w:6.2f} W  E/NTT={per_ntt_uj:8.4f} uJ  thr/W={thr_per_watt:8.2f} NTT/J")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT}")
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
