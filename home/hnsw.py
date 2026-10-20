"""A small hierarchical navigable small-world index.

Search is cosine similarity. Vectors are stored as unit vectors, so each
comparison is a dot product and the norm is computed once.
"""

import heapq
import math
import random

import numpy as np

class HNSW:
    def __init__(self, m=12, ef_construction=64, ef_search=64):
        self.m = m
        self.m0 = m * 2
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = 1.0 / math.log(m)
        self.vec = {}
        self.level = {}
        self.neighbors = []
        self.dead = set()
        self.entry = None

    def _level(self):
        unit = random.random() or 1e-12
        return min(16, int(-math.log(unit) * self.ml))

    def _ensure(self, level):
        while len(self.neighbors) <= level:
            self.neighbors.append({})

    def _dot(self, query, key):
        return float(np.dot(query, self.vec[key]))

    def _search_layer(self, query, entries, ef, layer):
        graph = self.neighbors[layer] if layer < len(self.neighbors) else {}
        visited = set()
        best = []
        cands = []
        for eid in entries:
            if eid in visited or eid in self.dead or eid not in self.vec:
                continue
            visited.add(eid)
            sim = self._dot(query, eid)
            heapq.heappush(best, (sim, eid))
            heapq.heappush(cands, (-sim, eid))
        while cands:
            sim, current = heapq.heappop(cands)
            sim = -sim
            if len(best) >= ef and sim < best[0][0]:
                break
            for neighbor in graph.get(current, ()):
                if neighbor in visited or neighbor in self.dead or neighbor not in self.vec:
                    continue
                visited.add(neighbor)
                nsim = self._dot(query, neighbor)
                if len(best) < ef or nsim > best[0][0]:
                    heapq.heappush(best, (nsim, neighbor))
                    heapq.heappush(cands, (-nsim, neighbor))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted(best, reverse=True)

    def _closest(self, query, keys, limit):
        scored = [(self._dot(query, key), key) for key in keys if key in self.vec and key not in self.dead]
        scored.sort(reverse=True)
        return [key for _sim, key in scored[:limit]]

    def _link(self, layer, left, right, limit):
        graph = self.neighbors[layer]
        graph.setdefault(left, [])
        graph.setdefault(right, [])
        if right not in graph[left]:
            graph[left].append(right)
        if left not in graph[right]:
            graph[right].append(left)
        for node in (left, right):
            if len(graph[node]) > limit:
                graph[node] = self._closest(self.vec[node], graph[node], limit)
        if right not in graph[left] and left in graph[right]:
            graph[right] = [key for key in graph[right] if key != left]
        if left not in graph[right] and right in graph[left]:
            graph[left] = [key for key in graph[left] if key != right]

    def add(self, key, vector):
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError(f"empty embedding for {key}")
        unit = vector / norm
        self.vec[key] = unit
        self.dead.discard(key)
        level = self._level()
        self.level[key] = level
        self._ensure(level)
        for layer in range(level + 1):
            self.neighbors[layer].setdefault(key, [])
        if self.entry is None:
            self.entry = key
            return norm
        top = self.level.get(self.entry, 0)
        ep = [self.entry]
        for layer in range(top, level, -1):
            found = self._search_layer(unit, ep, 1, layer)
            if found:
                ep = [found[0][1]]
        for layer in range(min(level, top), -1, -1):
            found = self._search_layer(unit, ep, self.ef_construction, layer)
            limit = self.m0 if layer == 0 else self.m
            neighbors = self._closest(unit, [item for _sim, item in found if item != key], limit)
            for neighbor in neighbors:
                self._link(layer, key, neighbor, limit)
            if found:
                ep = [found[0][1]]
        if level > top:
            self.entry = key
        return norm

    def discard(self, key):
        self.dead.add(key)
        if self.entry == key:
            live = [item for item in self.vec if item not in self.dead]
            self.entry = max(live, key=lambda item: self.level.get(item, 0)) if live else None

    def search(self, vector, k, ef=None):
        if self.entry is None or k <= 0:
            return []
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return []
        unit = vector / norm
        ep = [self.entry]
        top = self.level.get(self.entry, 0)
        for layer in range(top, 0, -1):
            found = self._search_layer(unit, ep, 1, layer)
            if found:
                ep = [found[0][1]]
        width = ef or max(self.ef_search, k)
        found = self._search_layer(unit, ep, width, 0)
        return [(key, sim) for sim, key in found if key not in self.dead][:k]
