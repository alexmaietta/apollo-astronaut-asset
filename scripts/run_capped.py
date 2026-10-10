"""Run a command and kill it (and its children) if their combined resident memory exceeds a limit.

    python scripts/run_capped.py 8 -- <command ...>     # limit in GiB

Heavy GPU sims/renders on this machine can otherwise push it into a global OOM that also kills the editor.
"""
import os
import signal
import subprocess
import sys
import time


def rss_kib(pid):
    total, todo = 0, [pid]
    while todo:
        p = todo.pop()
        try:
            with open(f"/proc/{p}/status") as f:
                total += next(int(line.split()[1]) for line in f if line.startswith("VmRSS:"))
            with open(f"/proc/{p}/task/{p}/children") as f:
                todo += [int(c) for c in f.read().split()]
        except (FileNotFoundError, StopIteration, ProcessLookupError):
            pass
    return total


def main():
    limit = float(sys.argv[1]) * 1024 * 1024
    cmd = sys.argv[sys.argv.index("--") + 1:]
    proc = subprocess.Popen(cmd, start_new_session=True)
    peak = 0
    while proc.poll() is None:
        r = rss_kib(proc.pid)
        peak = max(peak, r)
        if r > limit:
            os.killpg(proc.pid, signal.SIGKILL)
            print(f"run_capped: killed, resident memory {r / 1048576:.1f} GiB > {limit / 1048576:.1f} GiB", file=sys.stderr)
            sys.exit(137)
        time.sleep(0.25)
    print(f"run_capped: peak resident memory {peak / 1048576:.1f} GiB", file=sys.stderr)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
