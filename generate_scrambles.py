#!/usr/bin/env python3
"""
generate_scrambles.py
Generates the NOON Scramble puzzle bank from the existing NOON word square bank.

Each output puzzle:
  - Four valid NOON words (symmetric 4×4 word square)
  - Target swap count (3, 4, or 5) — the EXACT minimum swaps to solve
  - Scrambled grid: flat 16-element list of chars

Verification:
  An A* search with an admissible mismatch heuristic confirms that the true
  minimum number of swaps to restore the solved state equals the target.
  Any scramble solvable in fewer swaps is discarded and regenerated.

Output: scramble_puzzles.json
"""

import json
import heapq
import os
import random
import time

# ── Paths ─────────────────────────────────────────────────────────────────────

HERE        = os.path.dirname(os.path.abspath(__file__))
BANK_FILE   = os.path.join(HERE, 'noon_puzzles_final.json')
OUTPUT_FILE = os.path.join(HERE, 'scramble_puzzles.json')

# ── Config ─────────────────────────────────────────────────────────────────────

TARGET_COUNT = 300      # Puzzles to generate (covers 250+ days with buffer)
SEED         = 2026     # Reproducible run
MAX_RETRIES  = 500      # Per-puzzle generation attempts before giving up on a source

# Swap-count pattern cycling across the bank.
# Weights: 3-swap=25%, 4-swap=37.5%, 5-swap=37.5% for balanced difficulty.
SWAP_CYCLE = [3, 4, 5, 4, 5, 4, 5, 3]

# ── Load NOON bank ─────────────────────────────────────────────────────────────

with open(BANK_FILE) as f:
    NOON_BANK = json.load(f)

print(f"Loaded {len(NOON_BANK):,} NOON puzzles from {os.path.basename(BANK_FILE)}")

# Filter out any invalid word squares (non-symmetric) that slipped through curation
def is_valid_square(words):
    g = [list(w) for w in words]
    return all(g[r][c] == g[c][r] for r in range(4) for c in range(4))

NOON_BANK = [p for p in NOON_BANK if is_valid_square(p['words'])]
print(f"After symmetry filter: {len(NOON_BANK):,} valid puzzles")

# ── Grid helpers ───────────────────────────────────────────────────────────────

def words_to_grid(words):
    """4 four-letter words → 16-element tuple (row-major)."""
    return tuple(c for w in words for c in w)

def do_swap(grid, i, j):
    """Return new grid with positions i and j exchanged."""
    g = list(grid)
    g[i], g[j] = g[j], g[i]
    return tuple(g)

# ── A* verification ────────────────────────────────────────────────────────────

def h(state, goal):
    """
    Admissible heuristic: ceil(positional mismatches / 2).
    Each swap can fix at most 2 misplaced positions, so this never overestimates.
    """
    return (sum(1 for i in range(16) if state[i] != goal[i]) + 1) // 2

def astar_min_swaps(start, goal, max_depth):
    """
    A* search from `start` to `goal`.
    Returns the exact minimum number of swaps needed, or None if > max_depth.
    Skips swaps of identical letters (no-op moves that don't change the state).
    """
    if start == goal:
        return 0

    h0 = h(start, goal)
    if h0 > max_depth:
        return None

    # heap: (f = g + h, g, state)
    heap   = [(h0, 0, start)]
    closed = {}  # state → min g when it was expanded

    while heap:
        f, g, state = heapq.heappop(heap)

        if state in closed:
            continue
        closed[state] = g

        if state == goal:
            return g

        if g >= max_depth:
            continue

        for i in range(15):
            for j in range(i + 1, 16):
                if state[i] == state[j]:
                    continue  # swapping identical letters is a no-op
                ns = do_swap(state, i, j)
                if ns in closed:
                    continue
                ng  = g + 1
                nh  = h(ns, goal)
                nf  = ng + nh
                if nf <= max_depth:
                    heapq.heappush(heap, (nf, ng, ns))

    return None  # not reachable within max_depth

# ── Scramble builder ───────────────────────────────────────────────────────────

def build_scramble(words, target, rng):
    """
    Tries up to MAX_RETRIES times to generate a scrambled grid that requires
    EXACTLY `target` swaps to restore the solved state.

    Strategy:
      1. Apply `target` random swaps that change the state (no-op swaps skipped).
         Immediate reversals are also skipped to avoid trivial cancellations.
      2. Verify via A* that the true minimum equals `target`.
      3. Discard and retry if verification fails.

    Returns a 16-element list (the scrambled grid) or None on exhaustion.
    """
    solved = words_to_grid(words)

    for _ in range(MAX_RETRIES):
        grid     = solved
        last_ij  = None
        valid    = True

        for _ in range(target):
            # Candidates: swaps that actually change the state and aren't an immediate reversal
            candidates = [
                (i, j)
                for i in range(15)
                for j in range(i + 1, 16)
                if grid[i] != grid[j] and (i, j) != last_ij
            ]
            if not candidates:
                valid = False
                break
            i, j    = rng.choice(candidates)
            grid    = do_swap(grid, i, j)
            last_ij = (i, j)

        if not valid or grid == solved:
            continue  # All swaps cancelled each other out

        # A* verification: true minimum must be exactly `target`
        true_min = astar_min_swaps(grid, solved, target + 1)
        if true_min == target:
            return list(grid)

    return None  # Failed after MAX_RETRIES

# ── Main generation loop ───────────────────────────────────────────────────────

rng      = random.Random(SEED)
pool     = NOON_BANK[:]
rng.shuffle(pool)

puzzles  = []
stats    = {3: 0, 4: 0, 5: 0}
failures = 0
pool_idx = 0

print(f"\nGenerating {TARGET_COUNT} scramble puzzles (target: {TARGET_COUNT})…")
print(f"Swap distribution: {SWAP_CYCLE} (repeating cycle)\n")

t_start = time.perf_counter()

while len(puzzles) < TARGET_COUNT:
    src    = pool[pool_idx % len(pool)]
    pool_idx += 1
    target = SWAP_CYCLE[len(puzzles) % len(SWAP_CYCLE)]

    scrambled = build_scramble(src['words'], target, rng)
    if scrambled is None:
        failures += 1
        continue

    puzzles.append({
        'words':     src['words'],
        'swaps':     target,
        'scrambled': scrambled,
    })
    stats[target] += 1

    n = len(puzzles)
    if n % 50 == 0:
        elapsed = time.perf_counter() - t_start
        rate    = n / elapsed
        eta     = (TARGET_COUNT - n) / rate if rate > 0 else 0
        print(f"  {n:>3}/{TARGET_COUNT}  "
              f"3s={stats[3]:>3}  4s={stats[4]:>3}  5s={stats[5]:>3}  "
              f"failures={failures:>3}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

elapsed = time.perf_counter() - t_start

# ── Day-offset ─────────────────────────────────────────────────────────────────
#
# NOON daily index   : days_since_launch % len(NOON_BANK)   [12,009 puzzles]
# Scramble daily idx : (days_since_launch + offset) % len(puzzles)
#
# Setting offset = half the Scramble bank size means day 0 in each game maps to
# different puzzles (NOON shows puzzle 0, Scramble shows puzzle ~150).
# Because the two banks are independent word squares, there is no word-square
# overlap to worry about, but the offset further ensures Scramble never follows
# the same progression as NOON day-for-day.

offset = len(puzzles) // 2

# ── Write output ───────────────────────────────────────────────────────────────

output = {
    'meta': {
        'total':    len(puzzles),
        'offset':   offset,
        'counts':   stats,
        'note':     (
            'Daily puzzle index = (days_since_launch + offset) % total. '
            'Offset = half bank size so Scramble day-0 ≠ NOON day-0.'
        ),
    },
    'puzzles': puzzles,
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, separators=(',', ':'))

file_kb = os.path.getsize(OUTPUT_FILE) / 1024

# ── Summary ────────────────────────────────────────────────────────────────────

print(f"\n{'─' * 54}")
print(f"✓  {len(puzzles)} puzzles → {os.path.basename(OUTPUT_FILE)}  ({file_kb:.0f} KB)")
print(f"   3-swap : {stats[3]:>3} puzzles")
print(f"   4-swap : {stats[4]:>3} puzzles")
print(f"   5-swap : {stats[5]:>3} puzzles")
print(f"   Failures (exhausted retries): {failures}")
print(f"   Day offset for Scramble: {offset}")
print(f"   Total time: {elapsed:.1f}s")
print(f"{'─' * 54}")

# ── Spot-check ─────────────────────────────────────────────────────────────────

print("\nSpot-check (first puzzle of each swap count):")
seen = set()
for p in puzzles:
    t = p['swaps']
    if t in seen:
        continue
    seen.add(t)
    solved   = words_to_grid(p['words'])
    sc       = tuple(p['scrambled'])
    verified = astar_min_swaps(sc, solved, t + 1)
    rows     = [p['scrambled'][r*4:(r+1)*4] for r in range(4)]
    print(f"\n  words={p['words']}  swaps={t}  verified_min={verified}")
    print(f"  solved  : {list(solved)}")
    print(f"  scrambled: {p['scrambled']}")
    assert verified == t, f"VERIFICATION FAILED: expected {t}, got {verified}"
print("\nAll spot-checks passed ✓")
