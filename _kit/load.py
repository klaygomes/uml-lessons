import os
import re
import subprocess

LOAD_PER_CORE_LIMIT = 0.66
MIN_FREE_MEMORY_PERCENT = 30


def free_memory_percent() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    match = re.search(r"free percentage: (\d+)%", out)
    return int(match.group(1)) if match else 0


def main() -> None:
    cores = os.cpu_count() or 1
    load1, load5, _ = os.getloadavg()
    free = free_memory_percent()
    limit = cores * LOAD_PER_CORE_LIMIT
    ok = load1 < limit and free >= MIN_FREE_MEMORY_PERCENT
    print(f"{'OK' if ok else 'WAIT'}  load {load1:.1f}/{load5:.1f} (limit {limit:.1f} on {cores} cores)  memory free {free}% (min {MIN_FREE_MEMORY_PERCENT}%)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
