#!/usr/bin/env python3
"""octreg on the I58 brainstem pair, from the two original files. Any OS, any machine.

    python bench/run_i58.py                                                       # register + evaluate
    STEPS="ablate evaluate report" python bench/run_i58.py                        # bash
    $env:STEPS = "ablate evaluate report"; python bench/run_i58.py                # PowerShell

Steps in order: register (python -m octreg register OCT MRI -o OUT), ablate (bench/ablate.py --starts), evaluate
(bench/evaluate.py) and report (bench/report.py, which rewrites bench/BENCHMARK.md, bench/figures and bench/results/I58).
Overrides, all optional: STEPS, OUT, ABL, PREV_MAIN, DEVICE, CODE. The roots come from bench/paths.py (OCTREG_PROJECT_ROOT,
OCTREG_DATA_ROOT, OCTREG_I58_DIR) and CODE defaults to the repository of this file. The steps run with the interpreter that runs
this file.

OUT and ABL default to bench_runs/I58/octreg and bench_runs/I58/octreg_ablate. PREV_MAIN has no default. When it names a second
octreg run, the evaluate step also measures the pose distance to it (bench/evaluate.py --previous).

Each step writes NAME.log, NAME.time (wall time and exit status, the peak memory of a run is in its own result.json)
and, when nvidia-smi is on the PATH, NAME.gpu_mib into ${OUT}_logs, with a start and a done line
per step in chain.log. The runner writes its PID (POSIX: its process group) to ${OUT}_logs/runner.pid while it runs and
refuses to start while another registration job is running, since one GPU takes one at a time.

register and ablate refuse a default OUT or ABL that already holds a run, so a second run needs a new OUT and a new ABL.
The report step has no such guard. It rewrites bench/BENCHMARK.md from whatever OUT and ABL name.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from paths import BENCH_RUNS, MRI_I58, OCT_I58                        # bench/paths.py: the run and data roots

POSIX = os.name == "posix"
ENV = os.environ.get
CODE = Path(ENV("CODE") or Path(__file__).resolve().parents[1])
BENCH = BENCH_RUNS / "I58"
OUT = Path(ENV("OUT") or BENCH / "octreg")
ABL = Path(ENV("ABL") or BENCH / "octreg_ablate")
PREV_MAIN = Path(ENV("PREV_MAIN")) if ENV("PREV_MAIN") else None    # a second run for the pose distance, only when set
LOGS = Path(str(OUT).rstrip("/" + os.sep) + "_logs")
STEPS = (ENV("STEPS") or "register evaluate").split()
DEVICE = ENV("DEVICE") or "cuda"
KILL = LOGS / "runner.pid"                    # the running chain, so a second one refuses to start
BUSY = re.compile(r'(^|/)python[0-9.]*(\.exe)?"? .*(-m octreg|bench/(ablate|evaluate)\.py)',
                  0 if POSIX else re.I)


def say(msg, stamp=True):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}" if stamp else msg
    print(line, flush=True)
    with open(LOGS / "chain.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def command_lines():
    """Command lines of the running processes (Windows: of the python processes only), with forward slashes."""
    if POSIX:
        return subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True).stdout.splitlines()
    try:
        import psutil
        lines = [" ".join(p.info["cmdline"]) for p in psutil.process_iter(["pid", "name", "cmdline"])
                 if p.info["pid"] != os.getpid() and p.info["cmdline"] and (p.info["name"] or "").lower().startswith("python")]
    except ImportError:
        query = "Get-CimInstance Win32_Process -Filter 'Name like ''python%''' | ForEach-Object { $_.CommandLine }"
        lines = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True).stdout.splitlines()
    return [line.replace(os.sep, "/") for line in lines]


def run(cmd, log, env):
    """cmd from CODE with stdout and stderr to log -> (exit status, peak RSS in kB or None)."""
    with open(log, "wb") as f:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=CODE, env=env)
    try:
        return proc.wait()
    finally:
        if proc.returncode is None:                      # interrupted
            proc.kill()


def step(name, args):
    """python ARGS: output to NAME.log, wall time to NAME.time, GPU memory samples to NAME.gpu_mib. The peak memory of the
    step is in the run's own result.json (octreg.register.peak_rss_gb), which is what bench/report.py prints."""
    say(f"{name} start: python {' '.join(str(a) for a in args)}")
    smi, sampler = shutil.which("nvidia-smi"), None
    if smi:
        with open(LOGS / f"{name}.gpu_mib", "wb") as f:
            sampler = subprocess.Popen([smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-l", "5"],
                                       stdout=f, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    env = {**os.environ, "OMP_NUM_THREADS": "8"}
    if not POSIX:
        env.setdefault("PYTHONUTF8", "1")                # UTF-8 logs and text files whatever the ANSI code page
    t0 = time.time()
    try:
        rc = run([sys.executable, *args], LOGS / f"{name}.log", env)
    finally:
        if sampler:
            sampler.terminate()
            sampler.wait()
    m, s = divmod(time.time() - t0, 60)
    with open(LOGS / f"{name}.time", "w") as f:
        f.write(f"\tElapsed (wall clock) time (h:mm:ss or m:ss): {int(m)}:{s:05.2f}\n"
                f"\tExit status: {rc}\n")
    say(f"{name} done rc={rc}, {int(m)}:{s:05.2f} (m:ss)")
    if rc:
        for line in (LOGS / f"{name}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-5:]:
            say(line, stamp=False)
    return rc


def main():
    if sys.stdout is not None:
        sys.stdout.reconfigure(errors="replace")
    for f in (OCT_I58, MRI_I58):            # before any mkdir, so that wrong roots create no directories, and with a plain
        if not f.is_file():                 # print, since say() needs the log directory
            print(f"missing input {f} (OCTREG_I58_DIR or OCTREG_DATA_ROOT, see bench/paths.py)", flush=True)
            return 1
    if not CODE.is_dir():
        print(f"no code copy at {CODE}", flush=True)
        return 1
    for s, var, d, f in (("register", "OUT", OUT, "result.json"), ("ablate", "ABL", ABL, "ablations.json")):
        if s in STEPS and not ENV(var) and (d / f).is_file():      # never overwrite a run by default
            print(f"{s}: the default {var} {d} already holds a run: set {var} to a new directory", flush=True)
            return 1
    LOGS.mkdir(parents=True, exist_ok=True)
    busy = sum(bool(BUSY.search(line)) for line in command_lines())
    if busy:
        say(f"another python -m octreg, bench/ablate.py or bench/evaluate.py process is running ({busy}): not starting")
        return 1
    KILL.parent.mkdir(parents=True, exist_ok=True)
    KILL.write_text(f"{os.getpgid(0) if POSIX else os.getpid()}\n")
    try:
        for s in STEPS:
            if s == "register":
                args = ["-m", "octreg", "register", OCT_I58, MRI_I58, "-o", OUT, "--device", DEVICE]
            elif s == "ablate":
                args = ["bench/ablate.py", "--out", ABL, "--starts"]
                args += ["--main", OUT] if (OUT / "T_oct2mri.txt").is_file() else []
                args += ["--device", DEVICE]
            elif s == "evaluate":
                args = ["bench/evaluate.py", OUT]
                if PREV_MAIN and (PREV_MAIN / "T_oct2mri.txt").is_file():
                    args += ["--previous", PREV_MAIN]
                elif PREV_MAIN:
                    say(f"evaluate: PREV_MAIN {PREV_MAIN} holds no T_oct2mri.txt, so no pose distance")
                if (ABL / "prep/texture/oct_mask.nii.gz").is_file():
                    args += ["--masks", ABL / "prep/texture"]
                else:
                    say(f"evaluate: no {ABL / 'prep/texture'} yet, so no mask volume and no outline agreement "
                        "(run the ablate step first)")
            elif s == "report":
                args = ["bench/report.py", "--main", OUT, "--ablate", ABL, "--logs", LOGS, "-o", "bench/BENCHMARK.md",
                        "--figures", "bench/figures", "--store", "bench/results/I58"]
            else:
                say(f"unknown step '{s}' (register | ablate | evaluate | report)")
                return 1
            if step(s, args):
                return 1
        say(f"all steps done: {' '.join(STEPS)}")
        return 0
    finally:
        KILL.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
