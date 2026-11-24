"""Territory adjacency graph G=(V,E), border lengths, naval lanes, and GATv2 tensor builder."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from core.types import PerceivedFrame, TerritoryEntity


@dataclass
class TerritoryGraphData:
    node_feats: np.ndarray        # [max_nodes, 16] float32
    adj: np.ndarray               # [max_nodes, max_nodes] float32 (land contact + naval reachability)
    node_mask: np.ndarray         # [max_nodes] bool
    entities: list[TerritoryEntity] = field(default_factory=list)
    self_border_neighbors: list[TerritoryEntity] = field(default_factory=list)
    naval_targets: list[TerritoryEntity] = field(default_factory=list)
    encirclement_ratio: float = 0.0
    multi_front_count: int = 0
    max_neighbor_threat: float = 0.0
    self_density: float = 0.0


class TerritoryGraphBuilder:
    """Constructs relational territory graph G=(V,E) purely from CV label maps and entity stats."""

    def __init__(self, max_nodes: int = 32) -> None:
        self.max_nodes = max_nodes

    def build(self, pf: PerceivedFrame) -> TerritoryGraphData:
        n = self.max_nodes
        node_feats = np.zeros((n, 16), dtype=np.float32)
        adj = np.eye(n, dtype=np.float32)
        node_mask = np.zeros(n, dtype=bool)

        entities = pf.entities[:n]
        if not entities:
            return TerritoryGraphData(node_feats=node_feats, adj=adj, node_mask=node_mask)

        # Ensure self entity is always placed at slot 0 when visible
        self_idx = next((i for i, e in enumerate(entities) if e.is_self), None)
        if self_idx is not None and self_idx != 0:
            entities = [entities[self_idx]] + entities[:self_idx] + entities[self_idx + 1 :]

        id_to_slot = {e.entity_id: idx for idx, e in enumerate(entities)}
        h, w = pf.label_map.shape
        total_cells = float(max(1, h * w))

        # Vectorized neighbor contact detection across 4-connected pixels (horizontal + vertical shifts)
        lm = pf.label_map
        pairs_h = np.stack([lm[:, :-1].ravel(), lm[:, 1:].ravel()], axis=1)
        pairs_v = np.stack([lm[:-1, :].ravel(), lm[1:, :].ravel()], axis=1)
        pairs = np.concatenate([pairs_h, pairs_v], axis=0)
        diff_mask = (pairs[:, 0] != pairs[:, 1]) & (pairs[:, 0] > 0) & (pairs[:, 1] > 0)
        border_pairs = pairs[diff_mask]
        if len(border_pairs) > 0:
            uniq, counts = np.unique(border_pairs, axis=0, return_counts=True)
            for (a_id, b_id), cnt in zip(uniq, counts):
                sa = id_to_slot.get(int(a_id))
                sb = id_to_slot.get(int(b_id))
                if sa is not None and sb is not None:
                    w_edge = float(np.clip(np.log1p(cnt) / 5.0, 0.2, 1.0))
                    adj[sa, sb] = max(adj[sa, sb], w_edge)
                    adj[sb, sa] = max(adj[sb, sa], w_edge)

        # Also add weak naval edges between coastal entities when water separates them
        coastal_slots = [idx for idx, e in enumerate(entities) if e.coastal_px >= 2]
        for i in coastal_slots:
            for j in coastal_slots:
                if i != j and adj[i, j] == 0.0:
                    adj[i, j] = 0.15

        self_ent = entities[0] if entities[0].is_self else None
        self_troops = max(1.0, pf.self_troops if pf.self_troops > 0 else (self_ent.estimated_troops if self_ent else 1.0))
        self_area = max(1.0, float(pf.self_land_px if pf.self_land_px > 0 else (self_ent.area_px if self_ent else 1)))
        self_density = self_troops / self_area

        # Compute encirclement (fraction of own perimeter touching enemy territories vs neutral/water)
        encirclement = 0.0
        if pf.self_mask.any():
            dil = cv2.dilate(pf.self_mask, np.ones((3, 3), np.uint8)) > 0
            perim = dil & (pf.self_mask == 0)
            perim_total = int(perim.sum())
            if perim_total > 0:
                enemy_perim = int((perim & (lm > 0)).sum())
                encirclement = enemy_perim / float(perim_total)

        self_neighbors: list[TerritoryEntity] = []
        naval_targets: list[TerritoryEntity] = []
        max_threat = 0.0
        multi_front = 0

        for idx, e in enumerate(entities):
            node_mask[idx] = True
            e_area = max(1.0, float(e.area_px))
            e_density = e.estimated_troops / e_area
            rel_troops = float(np.clip(np.log1p(e.estimated_troops) - np.log1p(self_troops), -5.0, 5.0)) / 5.0
            rel_density = float(np.clip(e_density / max(0.5, self_density), 0.0, 4.0)) / 4.0
            is_border_neighbor = (not e.is_self) and (e.border_contact_with_self_px > 0 or (self_ent is not None and adj[0, idx] >= 0.2))

            if is_border_neighbor:
                self_neighbors.append(e)
                threat = (e.estimated_troops / max(10.0, self_troops)) * (1.0 + e.border_contact_with_self_px / max(5.0, self_area ** 0.5))
                if threat > max_threat:
                    max_threat = threat
                if e.estimated_troops > 0.35 * self_troops:
                    multi_front += 1
            elif (not e.is_self) and e.coastal_px >= 2 and (self_ent is not None and self_ent.coastal_px >= 2):
                naval_targets.append(e)

            node_feats[idx] = np.array(
                [
                    1.0 if e.is_self else 0.0,
                    1.0 if is_border_neighbor else 0.0,
                    float(np.log1p(e.area_px) / np.log1p(total_cells)),
                    float(np.clip(np.log1p(e.estimated_troops) / 16.0, 0.0, 1.0)),
                    float(np.clip(np.log1p(e_density) / 6.0, 0.0, 1.0)),
                    rel_troops,
                    rel_density,
                    e.centroid_norm[0],
                    e.centroid_norm[1],
                    float(np.clip(e.border_contact_with_self_px / 64.0, 0.0, 1.0)),
                    float(np.clip(e.coastal_px / 64.0, 0.0, 1.0)),
                    float(np.clip(e.delta_area_px / 25.0, -1.0, 1.0)),
                    float(np.clip(e.delta_troops / max(50.0, e.estimated_troops), -1.0, 1.0)),
                    1.0 if e.is_under_attack else 0.0,
                    1.0 if e.has_crown else 0.0,
                    e.troop_confidence,
                ],
                dtype=np.float32,
            )

        return TerritoryGraphData(
            node_feats=node_feats,
            adj=adj,
            node_mask=node_mask,
            entities=entities,
            self_border_neighbors=self_neighbors,
            naval_targets=naval_targets,
            encirclement_ratio=encirclement,
            multi_front_count=multi_front,
            max_neighbor_threat=float(max_threat),
            self_density=float(self_density),
        )
