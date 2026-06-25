from collections import defaultdict

PROFILE = defaultdict(float)
COUNTS = defaultdict(int)

def add(name, ms):
    PROFILE[name] += ms
    COUNTS[name] += 1

def report():
    print("\n==============================================")
    print("              AIGVS LATENCY PROFILE")
    print("==============================================")
    
    # We need to know total queries to average the counters
    total_queries = max(COUNTS.get("rust_unified_search_call", 1), 1)
    
    for k in sorted(PROFILE):
        # Treat counters differently than millisecond timings
        if k in ["attractors_visited_per_query", "leaves_visited_per_query", "total_candidates_scored"]:
            avg = PROFILE[k] / total_queries
            print(f"{k:30s} total={int(PROFILE[k]):<8d} avg={avg:8.2f} per query")
        else:
            avg = PROFILE[k] / max(COUNTS[k], 1)
            print(f"{k:30s} total={PROFILE[k]:8.2f} ms | avg={avg:8.4f} ms | calls={COUNTS[k]}")
            
    print("==============================================\n")