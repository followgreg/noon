#!/usr/bin/env python3
"""
generate_scramble_puzzles.py

Generates Scramble puzzles from the NOON word square bank.

Scramble rules:
  - Corner positions 0, 3, 12, 15 (top-left, top-right, bottom-left, bottom-right)
    are ALWAYS fixed — they remain in their solved positions and are never swapped
  - Only the 12 non-corner positions are eligible for swapping during scramble
  - Each puzzle applies 3, 4, or 5 random swaps using only non-corner positions
  - BFS (across ALL 16 positions) verifies that the true minimum swaps to restore
    the solved state equals the target; if fewer swaps suffice, discard and retry

Output: scramble_puzzles.json
"""

import json
import os
import random
import time
import heapq

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
BANK_FILE = os.path.join(HERE, 'noon_puzzles_final.json')
OUT_FILE  = os.path.join(HERE, 'scramble_puzzles.json')

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_COUNT = 300    # >= 250 required; extra gives a comfortable buffer
SEED         = 2027   # different from existing generate_scrambles.py
MAX_RETRIES  = 1000   # attempts per source word-square before moving on

# ── Grid positions ────────────────────────────────────────────────────────────
CORNERS     = (0, 3, 12, 15)          # flat indices of the four corner tiles
CORNER_SET  = set(CORNERS)
NON_CORNERS = [i for i in range(16) if i not in CORNER_SET]

# C(12,2) = 66 non-corner swap pairs — used for SCRAMBLING only
NC_PAIRS = [
    (i, j)
    for idx, i in enumerate(NON_CORNERS)
    for j in NON_CORNERS[idx + 1:]
]

# C(16,2) = 120 all-position swap pairs — used for BFS VERIFICATION
ALL_PAIRS = [(i, j) for i in range(16) for j in range(i + 1, 16)]

# Swap-count cycle: even thirds 3 / 4 / 5
SWAP_CYCLE = [3, 4, 5]


# ── Grid helpers ──────────────────────────────────────────────────────────────

def is_valid_square(words):
    g = [list(w) for w in words]
    return all(g[r][c] == g[c][r] for r in range(4) for c in range(4))

def words_to_grid(words):
    """Return a flat 16-char tuple from four 4-letter words."""
    return tuple(c for w in words for c in w)

def grid_swap(grid, i, j):
    """Return a new tuple with positions i and j exchanged."""
    g = list(grid)
    g[i], g[j] = g[j], g[i]
    return tuple(g)


# ── A* verification ───────────────────────────────────────────────────────────
# A* with an admissible heuristic finds the EXACT same optimal (minimum) solution
# as BFS, but prunes states BFS blindly expands.  At depth 4–5, BFS across
# C(16,2)=120 swap pairs builds a frontier of millions of states and becomes
# impractical (hours + gigabytes).  A* at the same depth takes milliseconds.
#
# Heuristic: h(state) = ceil(positional_mismatches / 2)
#   Each swap can fix at most 2 misplaced tiles, so h never overestimates
#   the true minimum — the heuristic is admissible and A* is optimal.
#   Search uses all C(16,2) = 120 position pairs, as specified.

def h(state, goal):
    return (sum(1 for i in range(16) if state[i] != goal[i]) + 1) // 2

def astar_min_swaps(start, goal, max_depth):
    """
    A* from `start` to `goal` across all 16-position swap pairs.
    Returns the exact minimum swaps needed, or None if > max_depth.
    Skips swaps of identical letters (no-ops that don't change the state).
    """
    if start == goal:
        return 0

    h0 = h(start, goal)
    if h0 > max_depth:
        return None

    # heap entries: (f = g + h, g, state)
    heap   = [(h0, 0, start)]
    closed = {}   # state → best g seen when expanded

    while heap:
        f, g, state = heapq.heappop(heap)

        if state in closed:
            continue
        closed[state] = g

        if state == goal:
            return g

        if g >= max_depth:
            continue

        for i, j in ALL_PAIRS:
            if state[i] == state[j]:
                continue                  # no-op
            ns  = grid_swap(state, i, j)
            if ns in closed:
                continue
            ng  = g + 1
            nh  = h(ns, goal)
            nf  = ng + nh
            if nf <= max_depth:
                heapq.heappush(heap, (nf, ng, ns))

    return None   # not reachable within max_depth


# ── Scramble builder ──────────────────────────────────────────────────────────

def build_scramble(words, target, rng):
    """
    Repeatedly apply `target` random non-corner swaps, then verify with BFS
    that the true minimum equals `target`. Returns the scrambled grid tuple,
    or None if MAX_RETRIES is exhausted.
    """
    solved = words_to_grid(words)

    for _ in range(MAX_RETRIES):
        grid    = solved
        last_ij = None

        for _ in range(target):
            # Pick a non-corner swap that:
            #   (a) actually changes the grid  (no same-letter no-ops)
            #   (b) doesn't immediately undo the previous swap
            candidates = [
                (i, j) for i, j in NC_PAIRS
                if grid[i] != grid[j] and (i, j) != last_ij
            ]
            if not candidates:
                break                       # degenerate — give up this attempt
            i, j    = rng.choice(candidates)
            grid    = grid_swap(grid, i, j)
            last_ij = (i, j)
        else:
            # All `target` swaps were successfully applied
            if grid == solved:
                continue                   # they all cancelled out

            # Safety: corners must still sit in their solved positions
            if any(grid[c] != solved[c] for c in CORNERS):
                continue                   # should never fire, but be safe

            # BFS verification: true minimum must be EXACTLY target
            true_min = astar_min_swaps(grid, solved, target)
            if true_min == target:
                return grid
            # true_min < target (too easy) or None (can't solve) — retry

    return None


# ── Load and filter bank ──────────────────────────────────────────────────────

with open(BANK_FILE) as f:
    raw_bank = json.load(f)

print(f"Loaded {len(raw_bank):,} puzzles from {os.path.basename(BANK_FILE)}")

bank = [p for p in raw_bank if is_valid_square(p['words'])]
print(f"After symmetry filter: {len(bank):,} valid word squares\n")

# ── Generation loop ───────────────────────────────────────────────────────────

rng      = random.Random(SEED)
pool     = bank[:]
rng.shuffle(pool)

puzzles  = []
stats    = {3: 0, 4: 0, 5: 0}
failures = 0
pool_idx = 0
t_start  = time.perf_counter()

print(f"Generating {TARGET_COUNT} puzzles — corners fixed at positions {CORNERS}...\n")

while len(puzzles) < TARGET_COUNT:
    src    = pool[pool_idx % len(pool)]
    pool_idx += 1
    target = SWAP_CYCLE[len(puzzles) % 3]

    scrambled = build_scramble(src['words'], target, rng)

    if scrambled is None:
        failures += 1
        continue

    solved  = words_to_grid(src['words'])
    corners = [solved[c] for c in CORNERS]  # sanity-reference corner letters

    puzzles.append({
        'words':          src['words'],
        'required_swaps': target,
        'scrambled':      list(scrambled),
        'corners':        corners,
    })
    stats[target] += 1

    n = len(puzzles)
    if n % 25 == 0:
        elapsed = time.perf_counter() - t_start
        rate    = n / elapsed
        eta     = (TARGET_COUNT - n) / rate if rate > 0 else 0
        print(f"  {n:>3}/{TARGET_COUNT}  "
              f"3-swap={stats[3]:>3}  4-swap={stats[4]:>3}  5-swap={stats[5]:>3}  "
              f"failures={failures:>3}  "
              f"elapsed={elapsed:.1f}s  ETA={eta:.0f}s")

elapsed = time.perf_counter() - t_start

# ── Day-offset ────────────────────────────────────────────────────────────────
offset = len(puzzles) // 2

# ── Write output ──────────────────────────────────────────────────────────────

output = {
    'meta': {
        'total':          len(puzzles),
        'offset':         offset,
        'counts':         stats,
        'corners_fixed':  list(CORNERS),
        'note': (
            'Daily index = (days_since_launch + offset) % total. '
            'Corner positions 0,3,12,15 are always locked to their solved values.'
        ),
    },
    'puzzles': puzzles,
}

with open(OUT_FILE, 'w') as f:
    json.dump(output, f, separators=(',', ':'))

kb = os.path.getsize(OUT_FILE) / 1024

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'─' * 54}")
print(f"✓  {len(puzzles)} puzzles → {os.path.basename(OUT_FILE)}  ({kb:.0f} KB)")
print(f"   3-swap : {stats[3]:>3} puzzles")
print(f"   4-swap : {stats[4]:>3} puzzles")
print(f"   5-swap : {stats[5]:>3} puzzles")
print(f"   Failures (MAX_RETRIES exhausted): {failures}")
print(f"   Day offset: {offset}")
print(f"   Total time: {elapsed:.1f}s")
print(f"{'─' * 54}")

# ── Spot-check first 5 output entries ────────────────────────────────────────

print("\nFirst 5 entries in scramble_puzzles.json:")
for i, p in enumerate(puzzles[:5]):
    solved   = words_to_grid(p['words'])
    sc       = tuple(p['scrambled'])
    verified = astar_min_swaps(sc, solved, p['required_swaps'] + 1)
    corners_ok = all(sc[c] == solved[c] for c in CORNERS)
    print(f"\n  [{i}] words={p['words']}  required_swaps={p['required_swaps']}")
    print(f"       corners={p['corners']}  corners_ok={corners_ok}")
    print(f"       solved   : {''.join(solved)}")
    print(f"       scrambled: {''.join(sc)}")
    print(f"       bfs_verified_min={verified}  match={verified == p['required_swaps']}")
    assert verified == p['required_swaps'], f"VERIFICATION FAILED at index {i}"

print("\nAll spot-checks passed ✓")
