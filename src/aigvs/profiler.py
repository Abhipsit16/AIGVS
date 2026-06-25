from collections import defaultdict

PROFILE = defaultdict(float)
COUNTS = defaultdict(int)

def add(name, ms):
    PROFILE[name] += ms
    COUNTS[name] += 1

def report():
    print("\n===== AIGVS PROFILE =====")
    for k in sorted(PROFILE):
        avg = PROFILE[k] / max(COUNTS[k], 1)
        print(
            f"{k:30s}"
            f" total={PROFILE[k]:8.2f} ms"
            f" avg={avg:8.4f} ms"
            f" calls={COUNTS[k]}"
        )
