from shapely.geometry import Point, LineString
from shapely.ops import split
from players.player import Player
from src.cake import Cake
import src.constants as c
from shapely.geometry import Polygon


class Player8(Player):
    """
    Player8 - Adaptive boundary search strategy.

    Inspired by the CrustOptimizingPlayer implementation by Player9,
    particularly the idea of iterating over boundary point pairs and scoring cuts
    by area differences. The specific search heuristics, candidate density strategy,
    and early-break optimizations are original to our group.

    Modifications:
    - First-cut special handling (no early break, higher resolution)
    - Adaptive candidate counts based on perimeter length
    - Balanced runtime heuristics for complex shapes (e.g., Koch snowflake)
    """

    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.total_area = self.cake.exterior_shape.area
        self.target_area = self.total_area / self.children
        self.target_areas = [self.target_area for _ in range(self.children - 1)]
        self.target_ratio = self.cake.interior_shape.area / self.cake.exterior_shape.area


    def get_cuts(self, method='beam_search') -> list[tuple[Point, Point]]:
        if method == 'beam_search':
            return self.get_cuts_beam_search()
        else:
            return self.get_cuts_bruteforce()


    def divide_boundary_into_arcs(self, boundary: LineString, num_arcs: int) -> list[list[Point]]:
        samples_per_arc = 10
        arc_size = boundary.length / num_arcs

        arcs = [
            [boundary.interpolate(i * arc_size / samples_per_arc + j * (arc_size / samples_per_arc))
            for j in range(samples_per_arc)]
            for i in range(num_arcs)
        ]

        return arcs
    

    def get_opposite_arc_pairs(self, num_arcs: int) -> list[tuple[int, int]]:
        arc_pairs = []

        for i in range(num_arcs):
            for offset in range(2, num_arcs - 2):  # skip neighbors, add opposite arcs
                j = (i + offset) % num_arcs
                arc_pairs.append((i, j))

        return arc_pairs
    

    def convexity(polygon: Polygon) -> float:
        hull_area = polygon.convex_hull.area
        return polygon.area / hull_area  # 1.0 if convex, < 1.0 if concave


    def get_cuts_beam_search(self) -> list[tuple[Point, Point]]:

        beam_width = 10
        refinement_radius = 0.05

        moves = []

        for cut_index, t_area in enumerate(self.target_areas):
            print(f"\n=== Beam Search Finding cut {cut_index + 1}/{self.children - 1} ===")
            piece = max(self.cake.get_pieces(), key=lambda p: p.area)
            boundary = piece.boundary
            boundary_len = boundary.length

            # Initial candidate sampling
            num_candidates = max(150, min(220, int(boundary_len / 1.2)))  # sampling btwn 150 and 220 pts, depending on the perimeter
            # play around with this, could be reduced

            step = boundary_len / num_candidates
            points = [boundary.interpolate(i * step) for i in range(num_candidates)]

            # we can refine to sample more strategically - ie start with more dense regions,
            # sample more in thin or convex areas, avoid small angled cuts. this could be separated out into a different
            # function

            candidate_scores = []

            # Score all pairs as in brute force, but keep top k only

            # num_arcs = 8
            # arcs = self.divide_boundary_into_arcs(boundary, num_arcs)
            # arc_pairs = self.get_opposite_arc_pairs(num_arcs)

            # for i, j in arc_pairs:
            #     for p1 in arcs[i]:
            #         for p2 in arcs[j]:
            #             result = self.score_cut(piece, p1, p2, t_area, self.target_ratio, cut_index)
            #             if result is None:
            #                 continue
            #             score, candidate = result
            #             candidate_scores.append((score, candidate))

            for i in range(num_candidates):
                for j in range(i + 1, num_candidates):
                    result = self.score_cut(piece, points[i], points[j], t_area, self.target_ratio, cut_index)
                    if result is None:
                        continue
                    score, candidate = result
                    candidate_scores.append((score, candidate))

            # Keep top-k candidates for refinement
            candidate_scores.sort(key=lambda x: x[0])
            top_candidates = candidate_scores[:beam_width]

            # Local refinement around top candidates
            refined_best_score = float("inf")
            refined_best_cut = None

            for score, (p1, p2) in top_candidates:
                # Sample points near p1 and p2 within refinement_radius
                p1_candidates = self.sample_nearby_points(boundary, p1, refinement_radius, num_samples=5)
                p2_candidates = self.sample_nearby_points(boundary, p2, refinement_radius, num_samples=5)

                for rp1 in p1_candidates:
                    for rp2 in p2_candidates:
                        if rp1.equals(rp2):
                            continue
                        result = self.score_cut(piece, rp1, rp2, t_area, self.target_ratio, cut_index)
                        if result is None:
                            continue
                        score, candidate = result
                        if score < refined_best_score:
                            refined_best_score = score
                            refined_best_cut = candidate

            if refined_best_cut is None:
                print(f"[WARN] No valid cut found at step {cut_index + 1}, retrying with brute-force for this cut")
                fallback_cut = self.get_single_bruteforce_cut(piece, t_area, cut_index)
                if fallback_cut is None:
                    print("[ERROR] Fallback also failed.")
                    break
                refined_best_cut = fallback_cut

            print(
                f"Chosen cut {cut_index + 1}: best_score={refined_best_score:.4f}, "
                f"target_area={t_area:.2f}, p1={refined_best_cut[0]}, p2={refined_best_cut[1]}"
            )

            self.cake.cut(refined_best_cut[0], refined_best_cut[1])
            moves.append(refined_best_cut)

        return moves
    

    def get_single_bruteforce_cut(self, piece, t_area, cut_index):
        boundary = piece.boundary
        boundary_len = boundary.length

        num_candidates = max(150, min(220, int(boundary_len / 1.2)))
        step = boundary.length / num_candidates
        points = [boundary.interpolate(i * step) for i in range(num_candidates)]

        best_score = float("inf")
        best_cut = None

        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                result = self.score_cut(piece, points[i], points[j], t_area, self.target_ratio, cut_index)
                if result is None:
                    continue
                score, candidate = result

                if cut_index == 0 and min(piece.area, t_area) > t_area:
                    score *= 1.25

                if score < best_score:
                    best_score = score
                    best_cut = candidate

        return best_cut
    
    
    def sample_nearby_points(self, boundary, point, radius, num_samples=5):
        """
        Given a boundary and a point on it, sample `num_samples` points near `point` within `radius` along the boundary.
        This is used for local refinement around candidate cuts.
        """
        boundary_len = boundary.length
        pos = boundary.project(point)
        candidates = []
        step = radius / (num_samples - 1) if num_samples > 1 else 0

        for i in range(num_samples):
            offset = pos - radius/2 + i * step
            offset = offset % boundary_len
            candidate = boundary.interpolate(offset)
            candidates.append(candidate)

        return candidates
    

    def score_cut(self, piece, p1, p2, target_area, target_ratio, cut_index):
        valid, _ = self.cake.cut_is_valid(p1, p2)
        if not valid:
            return None  # invalid cut

        line = LineString([p1, p2])
        parts = split(piece, line)
        if len(parts.geoms) != 2:
            return None

        a1, a2 = parts.geoms[0].area, parts.geoms[1].area
        c1 = parts.geoms[0].boundary.intersection(self.cake.exterior_shape.boundary).length
        c2 = parts.geoms[1].boundary.intersection(self.cake.exterior_shape.boundary).length

        if a1 < c.MIN_PIECE_AREA or a2 < c.MIN_PIECE_AREA:
            return None

        smallest_area = min(a1, a2)
        area_score = abs(smallest_area - target_area)

        ratio_1 = c1 / a1
        ratio_2 = c2 / a2

        ratio_score = max(abs(ratio_1 - target_ratio), abs(ratio_2 - target_ratio))

        ratio_weight = min(0.05 + 0.01 * (self.children - 1), 0.5)
        area_weight = 1 - ratio_weight

        score = area_weight * area_score + ratio_weight * ratio_score

        if cut_index == 0 and smallest_area > target_area:
            score *= 1.25

        return score, (p1, p2)


    def get_cuts_bruteforce(self) -> list[tuple[Point, Point]]:

        moves: list[tuple[Point, Point]] = []

        for cut_index, t_area in enumerate(self.target_areas):
            print(f"\n=== Finding cut {cut_index + 1}/{self.children - 1} ===")
            piece = max(self.cake.get_pieces(), key=lambda p: p.area)
            boundary = piece.boundary
            boundary_len = boundary.length

            # Adaptive candidate density
            if cut_index == 0:
                # First cut: more resolution, no early break
                num_candidates = min(500, max(200, int(boundary_len / 0.4)))
            else:
                # Subsequent cuts: slightly higher than before, but not full brute force
                num_candidates = max(150, min(220, int(boundary_len / 1.2)))

            step = boundary.length / num_candidates
            points = [boundary.interpolate(i * step) for i in range(num_candidates)]

            best_score = float("inf")
            best_cut = None

            # Early exit thresholds (tuned for typical cake sizes)
            perfect_thresh = t_area * 0.001  # within 0.1% of target area

            for i in range(num_candidates):
                for j in range(i + 1, num_candidates):
                    p1 = points[i]
                    p2 = points[j]

                    valid, _ = self.cake.cut_is_valid(p1, p2)
                    if not valid:
                        continue

                    line = LineString([p1, p2])
                    parts = split(piece, line)
                    if len(parts.geoms) != 2:
                        continue

                    a1, a2 = parts.geoms[0].area, parts.geoms[1].area
                    c1 = parts.geoms[0].boundary.intersection(self.cake.exterior_shape.boundary).length
                    c2 = parts.geoms[1].boundary.intersection(self.cake.exterior_shape.boundary).length

                    # Respect minimum piece area
                    if a1 < c.MIN_PIECE_AREA or a2 < c.MIN_PIECE_AREA:
                        continue

                    smallest_area = min(a1, a2)
                    area_score = abs(smallest_area - t_area)

                    # Crust to area ratio for each piece
                    ratio_1 = c1 / a1
                    ratio_2 = c2 / a2

                    # if cut_index == 0:  # discard really bad ratios in the first cut
                    #     if max(abs(ratio_1 - target_ratio), abs(ratio_2 - target_ratio)) > 0.3:
                    #         continue

                    ratio_score = max(abs(ratio_1 - self.target_ratio), abs(ratio_2 - self.target_ratio))

                    # ratio_weight, area_weight = self.compute_weights(cut_index, self.children - 1, self.children)

                    ratio_weight = min(0.05 + 0.01 * (self.children - 1), 0.5)
                    area_weight = 1 - ratio_weight

                    score = area_weight * area_score + ratio_weight * ratio_score 

                    if cut_index == 0 and smallest_area > t_area:
                        score *= 1.25

                    if score < best_score:
                        best_score = score
                        best_cut = (p1, p2)

                        # Early break if we find a "perfect enough" cut
                        if best_score < perfect_thresh:
                            break
                if best_score < perfect_thresh:
                    break

            if best_cut is None:
                print(f"[WARN] No valid cut found at step {cut_index + 1}")
                break

            print(
                f"Chosen cut {cut_index + 1}: best_score={best_score:.4f}, "
                f"target_area={t_area:.2f}, p1={best_cut[0]}, p2={best_cut[1]}"
            )

            # Apply the cut
            self.cake.cut(best_cut[0], best_cut[1])
            moves.append(best_cut)

        return moves


    def compute_weights(self, cut_index: int, total_cuts: int, num_children: int) -> tuple[float, float]:
        base_ratio_weight = min(0.05 + 0.02 * (num_children - 1), 0.8)

        min_phase_weight = min(0.5 + 0.03 * (num_children - 1), 0.8)

        if total_cuts == 1:
            phase_weight = 1.0
        else:
            decay_factor = 0.7
            phase_weight = max(decay_factor ** (cut_index / (total_cuts - 1)), min_phase_weight)

        ratio_weight = base_ratio_weight * phase_weight
        area_weight = 1 - ratio_weight
        print("RATIO WEIGHT: ", ratio_weight, "AREA WEIGHT: ", area_weight)
        return ratio_weight, area_weight

# special geometric handling: thin parts, more sampling
# convex bump: don't need as many samples. longer cuts across the center better
