"""Cards and a fast pure-Python 7-card hand evaluator (no external libraries, no file I/O).

Card encoding used internally: c = rank_index * 4 + suit, rank_index 0..12 = 2..A, suit 0..3.
Higher evaluate() value = better hand.
"""
from __future__ import annotations

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_OF = {r: i for i, r in enumerate(RANKS)}
SUIT_OF = {s: i for i, s in enumerate(SUITS)}
FULL_DECK = list(range(52))

# hand categories
HIGH, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = ["high card", "pair", "two pair", "three of a kind", "straight", "flush",
                  "full house", "four of a kind", "straight flush"]


def _build_tables():
    straight_high = [-1] * 8192
    for m in range(8192):
        for top in range(12, 3, -1):            # A-high .. 6-high
            need = 0b11111 << (top - 4)
            if m & need == need:
                straight_high[m] = top
                break
        else:
            if m & 0b1000000001111 == 0b1000000001111:   # wheel A-2-3-4-5
                straight_high[m] = 3
    popcount = [bin(m).count("1") for m in range(8192)]
    top5 = [0] * 8192
    for m in range(8192):
        v, n = 0, 0
        for r in range(12, -1, -1):
            if m >> r & 1:
                v = (v << 4) | r
                n += 1
                if n == 5:
                    break
        v <<= 4 * (5 - n)
        top5[m] = v
    return straight_high, popcount, top5


STRAIGHT_HIGH, POPCOUNT, TOP5 = _build_tables()


def card_from_any(x) -> int:
    """Parse common card formats: 'As', 'AS', 'as', '10h', 'Th', ('A','s'), {'rank':'A','suit':'s'},
    or an int already in our encoding. Raises ValueError if unknown."""
    if isinstance(x, bool):
        raise ValueError(x)
    if isinstance(x, int):
        if 0 <= x < 52:
            return x
        raise ValueError(x)
    if isinstance(x, dict):
        return card_from_any(str(x.get("rank", "")) + str(x.get("suit", "")))
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return card_from_any(str(x[0]) + str(x[1]))
    s = str(x).strip()
    if not s:
        raise ValueError(x)
    s = s.replace("10", "T")
    suit_ch = s[-1].lower()
    rank_ch = s[0].upper()
    suit_map = {"c": 0, "d": 1, "h": 2, "s": 3, "♣": 0, "♦": 1, "♥": 2, "♠": 3}
    if rank_ch not in RANK_OF or suit_ch not in suit_map:
        raise ValueError(x)
    return RANK_OF[rank_ch] * 4 + suit_map[suit_ch]


def card_str(c: int) -> str:
    return RANKS[c >> 2] + SUITS[c & 3]


def evaluate(cards) -> int:
    """Evaluate 5-7 cards. Returns an int: (category << 20) | kickers (4 bits each)."""
    sm0 = sm1 = sm2 = sm3 = 0
    counts = [0] * 13
    for c in cards:
        r = c >> 2
        counts[r] += 1
        s = c & 3
        if s == 0:
            sm0 |= 1 << r
        elif s == 1:
            sm1 |= 1 << r
        elif s == 2:
            sm2 |= 1 << r
        else:
            sm3 |= 1 << r
    flush_mask = 0
    for m in (sm0, sm1, sm2, sm3):
        if POPCOUNT[m] >= 5:
            flush_mask = m
            break
    if flush_mask:
        sh = STRAIGHT_HIGH[flush_mask]
        if sh >= 0:
            return (STRAIGHT_FLUSH << 20) | (sh << 16)
    quad = trip = trip2 = pair1 = pair2 = -1
    for r in range(12, -1, -1):
        n = counts[r]
        if n == 4:
            quad = r
        elif n == 3:
            if trip < 0:
                trip = r
            elif trip2 < 0:
                trip2 = r
        elif n == 2:
            if pair1 < 0:
                pair1 = r
            elif pair2 < 0:
                pair2 = r
    rank_mask = sm0 | sm1 | sm2 | sm3
    if quad >= 0:
        kick = rank_mask & ~(1 << quad)
        return (QUADS << 20) | (quad << 16) | ((TOP5[kick] >> 16) << 12)
    if trip >= 0 and (trip2 >= 0 or pair1 >= 0):
        p = max(trip2, pair1)
        return (FULL_HOUSE << 20) | (trip << 16) | (p << 12)
    if flush_mask:
        return (FLUSH << 20) | TOP5[flush_mask]
    sh = STRAIGHT_HIGH[rank_mask]
    if sh >= 0:
        return (STRAIGHT << 20) | (sh << 16)
    if trip >= 0:
        kick = rank_mask & ~(1 << trip)
        return (TRIPS << 20) | (trip << 16) | ((TOP5[kick] >> 12) << 8)
    if pair1 >= 0 and pair2 >= 0:
        kick = rank_mask & ~(1 << pair1) & ~(1 << pair2)
        return (TWO_PAIR << 20) | (pair1 << 16) | (pair2 << 12) | ((TOP5[kick] >> 16) << 8)
    if pair1 >= 0:
        kick = rank_mask & ~(1 << pair1)
        return (PAIR << 20) | (pair1 << 16) | ((TOP5[kick] >> 8) << 4)
    return (HIGH << 20) | TOP5[rank_mask]


def category(value: int) -> int:
    return value >> 20


def hand_class(c1: int, c2: int) -> str:
    """169-class name, e.g. 'AKs', 'T9o', '77'."""
    r1, r2 = c1 >> 2, c2 >> 2
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return RANKS[r1] * 2
    return RANKS[r1] + RANKS[r2] + ("s" if (c1 & 3) == (c2 & 3) else "o")


def draw_outs(hole, board) -> dict:
    """Simple draw detection for semi-bluffing: flush draw, open-ended / gutshot straight draws."""
    cards = list(hole) + list(board)
    suits = [0, 0, 0, 0]
    for c in cards:
        suits[c & 3] += 1
    flush_draw = any(n == 4 for n in suits) and any(((h & 3) == s and n == 4) for h in hole for s, n in enumerate(suits))
    mask = 0
    for c in cards:
        mask |= 1 << (c >> 2)
    if mask >> 12 & 1:
        mask |= 0          # ace handled in straight table already for made hands
    oesd = gut = False
    if STRAIGHT_HIGH[mask] < 0:
        completions = 0
        for r in range(13):
            if not mask >> r & 1 and STRAIGHT_HIGH[mask | (1 << r)] >= 0:
                completions += 1
        oesd = completions >= 2
        gut = completions == 1
    outs = (9 if flush_draw else 0) + (8 if oesd else (4 if gut else 0))
    if flush_draw and (oesd or gut):
        outs -= 2
    return {"flush_draw": flush_draw, "oesd": oesd, "gutshot": gut, "outs": outs}
