"""The class you pickle and submit (myteam.pkl).

Exposes the method names engines commonly call (act / decide / get_action / __call__), all routed
through the adapter to the engine-independent PokerBot. Adjust in one place once the official
template is known (e.g. rename `act` to whatever the engine calls).
"""
from __future__ import annotations

import adapter
from bot import Action, PokerBot


class TeamBot:
    def __init__(self, seed: int | None = None, think_time: float = 0.6):
        # everything the bot needs is created here and stored on the object (pickle-safe, no files)
        self.core = PokerBot(seed=seed, think_time=think_time)
        self.name = "MyTeamBot"

    def act(self, state):
        obs = adapter.to_obs(state)
        action = self.core.decide(obs)
        return adapter.from_action(action, obs)

    # aliases for other engine conventions
    decide = act
    get_action = act
    play = act
    run = act

    def __call__(self, state):
        return self.act(state)

    # optional hooks some engines call; safe no-ops
    def new_hand(self, *args, **kwargs):
        pass

    def hand_over(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass
