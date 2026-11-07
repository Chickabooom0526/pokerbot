# Heads-up NLHE bot

Pure Python, no network, no file I/O, all data loaded into the object in `__init__` (pickle-safe).

## Submit
- Class to pickle: `team_bot.TeamBot` (methods `act` / `decide` / `get_action` / `__call__` all work).
- Local check: `python make_pickle.py` -> writes `myteam.pkl`, reloads it, plays one move.
- Real submission: use the organisers' `python -m submit` with `TeamBot()`; include ALL .py files:
  `team_bot.py bot.py adapter.py cards.py equity.py preflop_data.py`.

## BEFORE SUBMITTING: check adapter.py against the official template
1. Key names for cards / pot / stacks / to_call / min & max raise (aliases in `KEYS`).
2. `RAISE_IS_TO` - True if raise amount = total bet ("raise to"), False if "raise by".
3. `ACTION_FORMAT` - what the engine expects back (dict / tuple / str).
If the engine calls a different method name, add an alias in team_bot.py.

## Strategy
Monte Carlo equity vs the opponent's estimated RANGE (narrowed by their actions), pot odds,
push/fold under 13bb, wide button opens, 3-bets, value bets / semi-bluffs, and in-match
opponent modelling (fold-to-bet, aggression, defend & 3-bet rates) that adjusts bluffing,
calling and opening. Fail-safe: any error -> check/fold; thinking capped at 0.6 s (limit 10 s).

## Local results (duplicate, seats swapped, bb/100)
equity-bot +183, TAG +139, nit +117, station +514, maniac +868, random +411, self-play ~0. Zero faults.
Test locally: local_engine.py (engine + duplicate_match), sparring.py (opponents).
