"""Task-level metrics: wrong actions, conflict spells, recovery times."""

from __future__ import annotations

from typing import Optional

from splitmind.environment.room_world import step_toward


def optimal_seek_action(rooms: list, agent_room: str, key_room: str) -> tuple:
    if agent_room == key_room:
        return ("collect",)
    return ("move", step_toward(rooms, agent_room, key_room))


def wrong_action_cause(action: tuple, optimal: tuple, belief: Optional[str],
                       belief_version: Optional[int], true_key: str, world_version: int) -> Optional[str]:
    """None if the action is optimal, else the reason the planner got it wrong.

    stale    planner believed a location that was true at an earlier world version
    noise    planner's belief is current-version evidence that was simply wrong
    unknown  planner had no belief and was exploring
    other    belief was correct but the action was still suboptimal
    """
    if tuple(action) == tuple(optimal):
        return None
    if belief is None:
        return "unknown"
    if belief != true_key:
        if belief_version is not None and belief_version < world_version:
            return "stale"
        return "noise"
    return "other"


def spells(flags: list) -> list:
    """Lengths of contiguous runs of True."""
    out, run = [], 0
    for f in flags:
        if f:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def recovery_times(event_times: list, correct_by_t: dict, end: int) -> list:
    """Time from each world change to the first step at which the condition holds.

    ``correct_by_t`` maps step -> bool. An event that is not recovered before
    the next event (or ``end``) is censored and reported as None.
    """
    out = []
    for i, te in enumerate(event_times):
        stop = event_times[i + 1] if i + 1 < len(event_times) else end
        rec = None
        for t in range(te, stop):
            if correct_by_t.get(t, False):
                rec = t - te
                break
        out.append(rec)
    return out
