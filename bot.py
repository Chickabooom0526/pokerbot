"""Heads-up No-Limit Hold'em bot.

Engine-independent core: the bot works on a canonical `Obs` (see adapter.py) and returns an
`Action(kind, amount_to)`. adapter.py converts the engine's state/actions to/from these.

Rules compliance: no network, no file I/O, no engine internals. All data (preflop tables) is copied
into the object in __init__, so the pickled bot is self-contained. Every decision is wrapped in a
fail-safe and a time budget far below the 10 s limit.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from cards import draw_outs, evaluate, hand_class
from equity import range_equity
from preflop_data import EQUITY, PERCENTILE


# ------------------------------------------------------------------------------------------
# Canonical observation / action
# ------------------------------------------------------------------------------------------
@dataclass
class Obs:
    hole: list                  # 2 ints (cards.card_from_any)
    board: list                 # 0,3,4,5 ints
    pot: float                  # total chips in the middle INCLUDING both players' bets this street
    to_call: float              # chips needed to call
    my_stack: float             # chips behind (not yet committed)
    opp_stack: float
    my_bet: float = 0.0         # committed this street
    opp_bet: float = 0.0
    big_blind: float = 2.0
    is_button: bool | None = None   # True = I'm the button / small blind (acts first preflop)
    legal: set = field(default_factory=lambda: {"fold", "check", "call", "raise"})
    min_raise_to: float | None = None   # minimum total bet ("raise to") this street
    max_raise_to: float | None = None   # maximum total bet this street (all-in)
    hand_id: object = None
    history: list = field(default_factory=list)   # optional [(street, 'me'|'opp', kind, amount_to)]

    @property
    def street(self) -> int:
        return {0: 0, 3: 1, 4: 2, 5: 3}.get(len(self.board), 0)


@dataclass
class Action:
    kind: str                   # 'fold' | 'check' | 'call' | 'raise'  (a bet is a 'raise' with to_call == 0)
    amount_to: float = 0.0      # for 'raise': total bet this street after the action


# ------------------------------------------------------------------------------------------
# Opponent model (Bayesian-smoothed frequencies, learnt within a match)
# ------------------------------------------------------------------------------------------
class OpponentModel:
    def __init__(self):
        # (successes, trials) with priors equivalent to ~8 observations of a typical HU player
        self.pfr = [3.2, 8.0]          # SB raises when it plays (vs limps)
        self.bb_defend = [4.8, 8.0]    # BB continues vs our open
        self.three_bet = [1.2, 8.0]    # BB 3-bets vs our open
        self.fold_to_bet = [3.6, 8.0]  # folds to our flop/turn bet
        self.aggr = [3.2, 8.0]         # bets/raises when given the chance postflop
        self.hands = 0

    @staticmethod
    def rate(x):
        return x[0] / x[1]

    def add(self, stat, success: bool):
        s = getattr(self, stat)
        s[0] += 1.0 if success else 0.0
        s[1] += 1.0

    def profile(self) -> str:
        f, a = self.rate(self.fold_to_bet), self.rate(self.aggr)
        if a > 0.6:
            return "maniac"
        if f < 0.3:
            return "station"
        if f > 0.62:
            return "nit"
        return "balanced"


# ------------------------------------------------------------------------------------------
# The bot
# ------------------------------------------------------------------------------------------
class PokerBot:
    def __init__(self, seed: int | None = None, think_time: float = 0.6, max_iters: int = 4000):
        self.equity_tbl = dict(EQUITY)          # data lives inside the object (pickle-safe)
        self.pct = dict(PERCENTILE)
        self.rng = random.Random(seed)
        self.think_time = think_time
        self.max_iters = max_iters
        self.opp = OpponentModel()
        self.mem = None                         # per-hand memory
        # tunable strategy parameters (kept on the object so they pickle with it)
        self.params = {
            "open_range": 0.80,        # share of hands the button opens
            "open_size_bb": 2.5,
            "bb_raise_vs_limp": 0.30,
            "value_3bet": 0.07,
            "bluff_3bet": 0.22,
            "defend_implied": 0.06,
            "cbet_base": 0.35,
            "value_adj": 0.0,          # added to value-bet thresholds
            "call_margin_adj": 0.0,    # added to calling margins
            "raise_eq": 0.82,          # equity needed to raise for value when facing a bet
            "bluff_scale": 1.0,
        }
        self.log = []                           # optional debug trail (kept short)

    # ---------------------------------------------------------------- public entry point
    def decide(self, obs: Obs) -> Action:
        t0 = time.perf_counter()
        try:
            self._track(obs)
            act = self._decide(obs, t0)
            act = self._legalise(obs, act)
        except Exception as e:                  # never crash: check if possible, else fold
            self.log = (self.log + [f"error: {e!r}"])[-20:]
            act = Action("check") if (obs.to_call <= 0 and "check" in obs.legal) else Action("fold")
            if act.kind == "fold" and "fold" not in obs.legal:
                act = Action("call")
        self._remember_own(obs, act)
        return act

    # ---------------------------------------------------------------- hand memory / tracking
    def _new_hand(self, obs):
        m = self.mem
        if m is not None:
            # opponent folded to our flop/turn bet if the hand ended right after it
            if m["last_action"] == "raise" and m["last_street"] in (1, 2) and not m["opp_allin"]:
                self.opp.add("fold_to_bet", True)
            if m["last_action"] == "raise" and m["last_street"] == 0 and m["we_opened"] and not m["opp_allin"]:
                self.opp.add("bb_defend", False)
        self.opp.hands += 1
        self.mem = {"key": (tuple(obs.hole), obs.hand_id), "street": -1, "opp_raises_pre": 0,
                    "pre_aggressor": None, "opp_aggr": {1: 0, 2: 0, 3: 0}, "opp_checked": {1: False, 2: False, 3: False},
                    "last_action": None, "last_street": 0, "we_opened": False, "seen_streets": set(),
                    "opp_allin": False, "my_last_bet_street": -1, "last_opp_bet": 0.0}

    def _track(self, obs: Obs):
        key = (tuple(obs.hole), obs.hand_id)
        m = self.mem
        if m is None or m["key"] != key or obs.street < m["street"]:
            self._new_hand(obs)
            m = self.mem
        st = obs.street
        bb = obs.big_blind
        first_time_this_street = st not in m["seen_streets"]
        # we bet the previous street and the hand continued -> opponent called
        if first_time_this_street and st >= 1 and m["last_action"] == "raise" and m["last_street"] == st - 1:
            if st - 1 in (1, 2):
                self.opp.add("fold_to_bet", False)
            if st - 1 == 0 and m["we_opened"]:
                self.opp.add("bb_defend", True)
                self.opp.add("three_bet", False)
        m["seen_streets"].add(st)
        if obs.opp_stack <= 0:
            m["opp_allin"] = True
        if st == 0:
            # opponent raised preflop if their bet exceeds the BB and our current bet
            if obs.opp_bet > max(bb, obs.my_bet) + 1e-9 and obs.opp_bet > m["last_opp_bet"] + 1e-9:
                m["opp_raises_pre"] += 1
                m["pre_aggressor"] = "opp"
                if m["we_opened"] and m["opp_raises_pre"] == 1:
                    self.opp.add("three_bet", True)
                    self.opp.add("bb_defend", True)
                if obs.is_button is False and m["opp_raises_pre"] == 1 and not m["we_opened"]:
                    self.opp.add("pfr", True)
            elif obs.is_button is False and st == 0 and obs.to_call <= 0 and m["opp_raises_pre"] == 0 and first_time_this_street:
                self.opp.add("pfr", False)          # SB limped
            m["last_opp_bet"] = obs.opp_bet
        else:
            if obs.to_call > 0:
                if m["opp_aggr"][st] == 0 or obs.opp_bet > m["last_opp_bet"]:
                    m["opp_aggr"][st] += 1
                    self.opp.add("aggr", True)
            elif first_time_this_street and obs.is_button is True and obs.my_bet == 0:
                m["opp_checked"][st] = True          # opponent (OOP) checked to us
                self.opp.add("aggr", False)
            m["last_opp_bet"] = obs.opp_bet
        m["street"] = st

    def _remember_own(self, obs, act):
        m = self.mem
        if m is None:
            return
        m["last_action"] = act.kind
        m["last_street"] = obs.street
        if act.kind == "raise":
            m["my_last_bet_street"] = obs.street
            if obs.street == 0:
                m["pre_aggressor"] = "me"
                if obs.is_button is True and m["opp_raises_pre"] == 0:
                    m["we_opened"] = True
        if obs.street > 0:
            m["last_opp_bet"] = max(m["last_opp_bet"], obs.opp_bet)

    # ---------------------------------------------------------------- helpers
    def _pct(self, obs):
        return self.pct[hand_class(obs.hole[0], obs.hole[1])]

    def _eff_bb(self, obs):
        return min(obs.my_stack + obs.my_bet, obs.opp_stack + obs.opp_bet) / obs.big_blind

    def _opp_range(self, obs):
        """Opponent's preflop range as a top fraction, from their preflop actions."""
        m = self.mem
        k = m["opp_raises_pre"]
        o = self.opp
        if k == 0:
            if obs.is_button:                    # opp is BB and only checked / called our open
                return 0.85 if m["we_opened"] else 1.0
            return 0.9                           # opp (SB) limped
        if k == 1:
            if obs.is_button is False and not m["we_opened"]:
                return min(0.95, max(0.35, o.rate(o.pfr)))      # SB open-raise range
            return min(0.4, max(0.08, o.rate(o.three_bet) * 1.6))  # a 3-bet
        return 0.06 if k == 2 else 0.035

    def _filters(self, obs):
        m = self.mem
        f = []
        for st in (1, 2, 3):
            if st > obs.street:
                break
            n = m["opp_aggr"][st]
            if n >= 2:
                f.append("raise")
            elif n == 1:
                big = st == obs.street and obs.to_call > 0.9 * max(obs.pot - obs.to_call, 1)
                f.append("bigbet" if big else "bet")
            elif m["last_street"] >= st and m["my_last_bet_street"] == st and st < obs.street:
                f.append("call")
        return f

    def _air_keep(self):
        a = self.opp.rate(self.opp.aggr)
        return min(0.75, max(0.15, 0.1 + 0.6 * a))

    def _equity(self, obs, t0, budget_scale=1.0):
        remaining = max(0.05, self.think_time * budget_scale - (time.perf_counter() - t0))
        eq, _ = range_equity(obs.hole, obs.board, self.pct, self.rng, top_frac=self._opp_range(obs),
                             filters=self._filters(obs), air_keep=self._air_keep(),
                             time_budget=remaining, max_iters=self.max_iters)
        return eq

    def _raise_to(self, obs, size_to):
        lo = obs.min_raise_to if obs.min_raise_to is not None else obs.opp_bet + max(obs.big_blind, obs.to_call)
        hi = obs.max_raise_to if obs.max_raise_to is not None else obs.my_bet + obs.my_stack
        if size_to >= 0.7 * hi or lo >= hi:      # don't leave a silly amount behind: just shove
            return Action("raise", hi)
        if float(lo).is_integer() and float(hi).is_integer():
            size_to = round(size_to)            # integer-chip engines
        return Action("raise", min(hi, max(lo, size_to)))

    def _bet_pot(self, obs, frac):
        """Bet/raise sized as a fraction of the pot after calling."""
        if obs.to_call <= 0:
            return self._raise_to(obs, obs.my_bet + frac * obs.pot)
        return self._raise_to(obs, obs.opp_bet + frac * (obs.pot + obs.to_call))

    def _exploit(self):
        """Continuous adjustments from opponent stats (instead of one style label)."""
        o = self.opp
        fold = o.rate(o.fold_to_bet)
        aggr = o.rate(o.aggr)
        P = self.params
        return {
            "bluff_scale": min(1.8, max(0.25, 0.3 + 2.0 * (fold - 0.35))) * P["bluff_scale"],
            "call_margin": 0.05 - 0.12 * (aggr - 0.4) + P["call_margin_adj"],
            "value_thresh": 0.58 + 0.1 * (fold - 0.45) + P["value_adj"],
            "trap": min(0.35, max(0.0, (aggr - 0.5) * 0.8)),
            "open_range": min(0.95, max(0.45, P["open_range"] + 0.3 * (0.55 - o.rate(o.bb_defend))
                                       - 0.4 * max(0.0, o.rate(o.three_bet) - 0.2))),
            "fold": fold, "aggr": aggr,
        }

    def _call_or_check(self, obs):
        return Action("call") if obs.to_call > 0 else Action("check")

    # ---------------------------------------------------------------- strategy
    def _decide(self, obs: Obs, t0: float) -> Action:
        if obs.street == 0:
            return self._preflop(obs, t0)
        return self._postflop(obs, t0)

    def _preflop(self, obs, t0):
        bb = obs.big_blind
        p = self._pct(obs)                       # 0 = best, 1 = worst
        eff = self._eff_bb(obs)
        m = self.mem
        rng = self.rng
        prof = self.opp.profile()
        pot_odds = obs.to_call / (obs.pot + obs.to_call) if obs.to_call > 0 else 0.0

        # ---- short stacks: push / fold -------------------------------------------------------
        if eff <= 13:
            if m["opp_raises_pre"] == 0:
                push_range = min(1.0, 0.30 + 2.8 / max(eff, 1))
                if obs.is_button is False and obs.to_call <= 0:      # BB vs limp
                    push_range *= 0.8
                if p <= push_range:
                    return self._raise_to(obs, obs.my_bet + obs.my_stack)
                return self._call_or_check(obs) if obs.to_call <= 0 else Action("fold")
            eq = self._equity(obs, t0, 0.7)
            if eq >= pot_odds + 0.02:
                return self._raise_to(obs, obs.my_bet + obs.my_stack) if eq > 0.55 else Action("call")
            return Action("fold")

        k = m["opp_raises_pre"]
        # ---- unopened: we are SB/button first to act -------------------------------------------
        if obs.is_button is not False and k == 0 and obs.to_call > 0 and obs.to_call <= 0.5 * bb + 1e-9:
            P = self.params
            open_range = self._exploit()["open_range"]
            if p <= open_range or (p <= open_range + 0.08 and rng.random() < 0.4):
                return self._raise_to(obs, P["open_size_bb"] * bb)
            return Action("fold")

        # ---- BB facing a limp ------------------------------------------------------------------
        if k == 0 and obs.to_call <= 0:
            if p <= self.params["bb_raise_vs_limp"] or (p <= 0.5 and rng.random() < 0.2):
                return self._raise_to(obs, 4.0 * bb)
            return Action("check")

        # ---- facing a raise (open, 3-bet, 4-bet ...) -------------------------------------------
        eq = self._equity(obs, t0, 0.7)
        raise_bb = obs.opp_bet / bb
        if k == 1 and not m["we_opened"]:        # we are BB facing an open
            if p <= self.params["value_3bet"] or (eq > 0.62 and p <= self.params["value_3bet"] + 0.05):
                return self._raise_to(obs, 3.2 * obs.opp_bet)            # value 3-bet
            X = self._exploit()
            if 0.30 <= p <= 0.50 and rng.random() < self.params["bluff_3bet"] * min(1.5, X["bluff_scale"]) and raise_bb <= 3.5:
                return self._raise_to(obs, 3.2 * obs.opp_bet)            # light 3-bet (bluff)
            implied = self.params["defend_implied"] if eff > 40 else 0.03
            if eq + implied >= pot_odds:
                return Action("call")
            return Action("fold")
        # facing a 3-bet or more
        if p <= 0.025 or (k >= 2 and p <= 0.012):
            return self._raise_to(obs, max(2.3 * obs.opp_bet, obs.my_bet + obs.my_stack if eff < 60 else 0))
        if eq >= pot_odds + max(-0.02, self._exploit()["call_margin"] - 0.02):
            return Action("call")
        return Action("fold")

    def _postflop(self, obs, t0):
        rng = self.rng
        m = self.mem
        prof = self.opp.profile()
        eq = self._equity(obs, t0)
        draws = draw_outs(obs.hole, obs.board) if obs.street < 3 else {"outs": 0}
        strong_draw = draws["outs"] >= 8
        pot_odds = obs.to_call / (obs.pot + obs.to_call) if obs.to_call > 0 else 0.0
        spr = obs.my_stack / max(obs.pot, 1)
        fold_rate = self.opp.rate(self.opp.fold_to_bet)
        P = self.params
        X = self._exploit()
        bluff_scale = X["bluff_scale"]
        value_thresh = X["value_thresh"]
        in_position = obs.is_button is True

        # ---------------- nobody has bet: we may check or bet ----------------
        if obs.to_call <= 0:
            if eq >= 0.80:
                if obs.street < 3 and rng.random() < X["trap"]:
                    return Action("check")                   # trap an aggressive player
                return self._bet_pot(obs, 0.75 if spr > 1.5 else 1.0)
            if eq >= value_thresh:
                if in_position or rng.random() < 0.7:
                    return self._bet_pot(obs, 0.55)
                return Action("check")
            if strong_draw and rng.random() < 0.55 * min(1.6, bluff_scale):
                return self._bet_pot(obs, 0.6)
            # c-bet / probe bluffs with air
            we_were_aggressor = m["pre_aggressor"] == "me"
            base = 0.0
            if obs.street == 1 and we_were_aggressor:
                base = P["cbet_base"] + 0.5 * (fold_rate - 0.4)
            elif m["opp_checked"].get(obs.street) and in_position:
                base = 0.25
            elif obs.street == 3 and m["opp_checked"].get(2):
                base = 0.15
            if eq < 0.35 and rng.random() < max(0.0, base) * bluff_scale:
                return self._bet_pot(obs, 0.5 if obs.street == 1 else 0.66)
            return Action("check")

        # ---------------- facing a bet / raise ----------------
        margin = X["call_margin"]
        implied = 0.06 if strong_draw and obs.street < 3 and spr > 1.5 else 0.0
        if eq >= P["raise_eq"] or (eq >= P["raise_eq"] - 0.10 and spr < 1.5):
            if rng.random() < 0.75 or spr < 1.5:
                return self._bet_pot(obs, 0.9)
            return Action("call")
        if eq + implied >= pot_odds + margin:
            if strong_draw and obs.street == 1 and rng.random() < 0.18 * bluff_scale and spr > 2:
                return self._bet_pot(obs, 0.9)                  # semi-bluff raise
            return Action("call")
        if strong_draw and obs.street < 3 and rng.random() < 0.08 * bluff_scale and spr > 2:
            return self._bet_pot(obs, 0.9)
        return Action("fold")

    # ---------------------------------------------------------------- legality guard
    def _legalise(self, obs, act: Action) -> Action:
        legal = obs.legal
        if act.kind == "raise":
            hi = obs.max_raise_to if obs.max_raise_to is not None else obs.my_bet + obs.my_stack
            if "raise" not in legal or hi <= obs.opp_bet or obs.my_stack <= obs.to_call:
                act = Action("call") if obs.to_call > 0 else Action("check")
        if act.kind == "check" and obs.to_call > 0:
            act = Action("call") if "call" in legal else Action("fold")
        if act.kind == "call" and obs.to_call <= 0:
            act = Action("check") if "check" in legal else Action("call")
        if act.kind == "fold" and obs.to_call <= 0 and "check" in legal:
            act = Action("check")                     # never fold when checking is free
        if act.kind not in legal:
            for k in ("check", "call", "fold"):
                if k in legal:
                    return Action(k)
        return act
