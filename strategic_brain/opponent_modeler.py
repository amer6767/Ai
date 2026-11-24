"""Opponent behavioral profiler tracking aggression, full-send vulnerabilities, and expansion tempo."""

from __future__ import annotations

from dataclasses import dataclass

from core.types import TerritoryEntity


@dataclass
class OpponentProfile:
    color_key: tuple[int, int, int]
    observations: int = 0
    aggression_score: float = 0.0        # high if frequently expanding/attacking us
    vulnerability_score: float = 0.0     # high if troops plummeted suddenly (full-sent / multi-fronted)
    avg_density: float = 10.0
    last_troops: float = 0.0
    last_area: int = 1


class OpponentModeler:
    """Maintains rolling behavior estimates for visible kingdoms solely from visual observations."""

    def __init__(self) -> None:
        self._profiles: dict[tuple[int, int, int], OpponentProfile] = {}

    def reset(self) -> None:
        self._profiles.clear()

    @staticmethod
    def _key(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
        return (rgb[0] // 16, rgb[1] // 16, rgb[2] // 16)

    def update(self, entities: list[TerritoryEntity], self_losing_troops: bool) -> dict[int, OpponentProfile]:
        result: dict[int, OpponentProfile] = {}
        for e in entities:
            if e.is_self:
                continue
            k = self._key(e.mean_rgb)
            prof = self._profiles.get(k)
            density = e.estimated_troops / max(1.0, float(e.area_px))
            if prof is None:
                prof = OpponentProfile(
                    color_key=k,
                    observations=1,
                    avg_density=density,
                    last_troops=e.estimated_troops,
                    last_area=e.area_px,
                )
                self._profiles[k] = prof
            else:
                prof.observations += 1
                troop_drop_ratio = (prof.last_troops - e.estimated_troops) / max(50.0, prof.last_troops)
                # Sudden >22% drop signals a heavy attack or full-send -> huge vulture vulnerability!
                if troop_drop_ratio > 0.22:
                    prof.vulnerability_score = min(1.0, prof.vulnerability_score * 0.5 + min(1.0, troop_drop_ratio * 2.0))
                else:
                    prof.vulnerability_score *= 0.80

                if e.border_contact_with_self_px > 0 and self_losing_troops and e.delta_area_px > 0:
                    prof.aggression_score = min(1.0, prof.aggression_score + 0.35)
                elif e.delta_area_px > 3:
                    prof.aggression_score = min(1.0, prof.aggression_score * 0.92 + 0.08)
                else:
                    prof.aggression_score *= 0.95

                prof.avg_density = 0.8 * prof.avg_density + 0.2 * density
                prof.last_troops = e.estimated_troops
                prof.last_area = e.area_px
            result[e.entity_id] = prof
        return result
