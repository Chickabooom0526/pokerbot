"""Adapter between the competition engine and the bot core.

>>> THIS IS THE ONLY FILE THAT DEPENDS ON THE ENGINE'S EXACT FORMAT. <<<
It accepts many common key names out of the box. When you see the official starter template,
check three things (marked CHECK below):
  1. which keys hold the cards / pot / stacks / amounts,
  2. whether 'raise' amounts are "raise TO" (total bet) or "raise BY" (increment),
  3. what the engine expects back (dict, tuple, string ...).
"""
from __future__ import annotations

from cards import card_from_any
from bot import Action, Obs

# CHECK 2: True if the engine's raise amount means "total bet this street" (most engines).
RAISE_IS_TO = True
# CHECK 3: how to return actions: "dict" -> {"action": "raise", "amount": 120}
#                                  "tuple" -> ("raise", 120)     "str" -> "raise 120"
ACTION_FORMAT = "dict"

# CHECK 1: aliases tried in order for each field
KEYS = {
    "hole": ["hole_cards", "hole", "my_cards", "hand", "cards", "private_cards", "pocket", "your_cards"],
    "board": ["board", "community_cards", "community", "board_cards", "table_cards", "shared_cards"],
    "pot": ["pot", "pot_size", "total_pot"],
    "to_call": ["to_call", "amount_to_call", "call_amount", "cost_to_call", "owed"],
    "my_stack": ["stack", "my_stack", "chips", "my_chips", "remaining_stack", "your_stack"],
    "opp_stack": ["opponent_stack", "opp_stack", "villain_stack", "their_stack", "other_stack"],
    "my_bet": ["my_bet", "bet", "current_bet_me", "my_street_bet", "committed", "my_pip"],
    "opp_bet": ["opponent_bet", "opp_bet", "their_bet", "opp_street_bet", "opp_pip"],
    "bb": ["big_blind", "bb", "bigblind"],
    "button": ["is_button", "button", "is_dealer", "dealer", "is_small_blind", "in_position"],
    "legal": ["legal_actions", "valid_actions", "allowed_actions", "actions", "available_actions"],
    "min_raise": ["min_raise_to", "min_raise", "minimum_raise", "raise_min"],
    "max_raise": ["max_raise_to", "max_raise", "maximum_raise", "raise_max"],
    "hand_id": ["hand_id", "hand_number", "deal_id", "hand_no", "game_id"],
    "history": ["history", "action_history", "actions_history", "log"],
}


def _get(state, field, default=None):
    for k in KEYS[field]:
        if isinstance(state, dict) and k in state and state[k] is not None:
            return state[k]
        if not isinstance(state, dict) and hasattr(state, k):
            v = getattr(state, k)
            if v is not None:
                return v
    return default


def _cards(x):
    if x is None:
        return []
    if isinstance(x, str):
        x = x.replace(",", " ").split() if (" " in x or "," in x) else [x[i:i + 2] for i in range(0, len(x), 2)]
    out = []
    for c in x:
        try:
            out.append(card_from_any(c))
        except ValueError:
            pass
    return out


def _legal(x) -> set:
    if not x:
        return {"fold", "check", "call", "raise"}
    names = set()
    for a in x:
        s = (a.get("action") or a.get("type") or a.get("name")) if isinstance(a, dict) else str(a)
        s = str(s).lower()
        if s in ("bet", "raise", "allin", "all_in", "all-in"):
            names.add("raise")
        elif s in ("fold", "check", "call"):
            names.add(s)
    return names or {"fold", "check", "call", "raise"}


def to_obs(state) -> Obs:
    hole = _cards(_get(state, "hole"))
    board = _cards(_get(state, "board"))
    bb = float(_get(state, "bb", 2) or 2)
    my_bet = float(_get(state, "my_bet", 0) or 0)
    opp_bet = float(_get(state, "opp_bet", 0) or 0)
    to_call = _get(state, "to_call")
    to_call = float(to_call) if to_call is not None else max(0.0, opp_bet - my_bet)
    pot = float(_get(state, "pot", my_bet + opp_bet) or 0)
    my_stack = float(_get(state, "my_stack", 100 * bb))
    opp_stack = float(_get(state, "opp_stack", my_stack))
    btn = _get(state, "button")
    is_button = None if btn is None else bool(btn) if not isinstance(btn, str) else btn.lower() in ("btn", "button", "sb", "dealer", "true")
    mn = _get(state, "min_raise")
    mx = _get(state, "max_raise")
    if mn is not None and not RAISE_IS_TO:
        mn = float(mn) + opp_bet          # convert "raise by" into "raise to"
    if mx is not None and not RAISE_IS_TO:
        mx = float(mx) + opp_bet
    hist = _get(state, "history", []) or []
    hid = _get(state, "hand_id")
    try:
        hash(hid)
    except TypeError:
        hid = str(hid)
    return Obs(hole=hole, board=board, pot=pot, to_call=to_call, my_stack=my_stack, opp_stack=opp_stack,
               my_bet=my_bet, opp_bet=opp_bet, big_blind=bb, is_button=is_button,
               legal=_legal(_get(state, "legal")), min_raise_to=None if mn is None else float(mn),
               max_raise_to=None if mx is None else float(mx), hand_id=hid,
               history=list(hist) if isinstance(hist, (list, tuple)) else [])


def from_action(act: Action, obs: Obs):
    kind, amt = act.kind, act.amount_to
    if kind == "raise":
        if not RAISE_IS_TO:
            amt = amt - obs.opp_bet
        amt = int(round(amt)) if abs(amt - round(amt)) < 1e-9 else amt
    else:
        amt = 0
    if ACTION_FORMAT == "tuple":
        return (kind, amt)
    if ACTION_FORMAT == "str":
        return f"{kind} {amt}" if kind == "raise" else kind
    return {"action": kind, "amount": amt}
