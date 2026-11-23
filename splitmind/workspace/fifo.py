"""FIFO: messages are processed in arrival order; each one overwrites the content.

The content after a tick is the last message delivered in that tick, or the
previous content if nothing arrived. Evidence time and confidence are ignored.
"""

from __future__ import annotations

from splitmind.workspace.base import ArbitrationStrategy


class FIFOStrategy(ArbitrationStrategy):
    name = "fifo"

    def select(self, candidates, arrivals, incumbent, now):
        return arrivals[-1] if arrivals else incumbent
