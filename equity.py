"""Time-budgeted Monte Carlo equity against an opponent RANGE (not just a random hand).

The range is described by:
  top_frac  - opponent's preflop range = the best `top_frac` share of hands (by preflop percentile)
  filters   - postflop actions, e.g. ["bet", "raise"]: hands that 'missed' the board are only kept
              with probability `air_keep` (bluffs), weak pairs with `weak_keep`.
"""
from __future__ import annotations

import time

from cards import PAIR, draw_outs, evaluate, hand_class


def _opp_hand_strength(o1, o2, board, board_val_cat):
    """0 = air, 1 = weak made hand / draw, 2 = strong (top pair+ or better)."""
    if not board:
        return 2
    v = evaluate([o1, o2] + board)
    cat = v >> 20
    if cat > board_val_cat:
        if cat >= 2:             # two pair or better using a hole card
            return 2
        pair_rank = (v >> 16) & 15
        top_board = max(c >> 2 for c in board)
        return 2 if pair_rank >= top_board else 1
    d = draw_outs([o1, o2], board)
    return 1 if d["outs"] >= 8 else 0


def range_equity(hole, board, percentile: dict, rng, top_frac: float = 1.0, filters=(),
                 air_keep: float = 0.35, weak_keep: float = 0.7, time_budget: float = 0.5,
                 max_iters: int = 4000, min_iters: int = 300) -> tuple[float, int]:
    """Return (equity in [0,1], iterations). Ties count half."""
    known = set(hole) | set(board)
    deck = [c for c in range(52) if c not in known]
    need = 5 - len(board)
    board_cat = (evaluate(board) >> 20) if len(board) >= 3 else -1
    edge = 0.08                                  # soft edge on the preflop range cut
    keep_air = 1.0
    keep_weak = 1.0
    for f in filters:
        if f == "raise":
            keep_air *= air_keep * 0.6
            keep_weak *= weak_keep * 0.6
        elif f in ("bet", "bigbet"):
            keep_air *= air_keep * (0.7 if f == "bigbet" else 1.0)
            keep_weak *= weak_keep
        elif f == "call":
            keep_air *= 0.5
    use_filter = bool(filters) and len(board) >= 3
    start = time.perf_counter()
    wins = 0.0
    n = 0
    my_hole = list(hole)
    while n < max_iters:
        if n >= min_iters and (n & 127) == 0 and time.perf_counter() - start > time_budget:
            break
        # --- sample an opponent hand from the weighted range (rejection sampling) ---
        for _ in range(60):
            o1, o2 = rng.sample(deck, 2)
            p = percentile[hand_class(o1, o2)]
            if p > top_frac:
                if p > top_frac + edge or rng.random() > (top_frac + edge - p) / edge:
                    continue
            if use_filter:
                s = _opp_hand_strength(o1, o2, list(board), board_cat)
                if s == 0 and rng.random() > keep_air:
                    continue
                if s == 1 and rng.random() > keep_weak:
                    continue
            break
        rest = [c for c in deck if c != o1 and c != o2]
        runout = rng.sample(rest, need) if need else []
        full = list(board) + runout
        a = evaluate(my_hole + full)
        b = evaluate([o1, o2] + full)
        if a > b:
            wins += 1.0
        elif a == b:
            wins += 0.5
        n += 1
    return wins / max(n, 1), n
