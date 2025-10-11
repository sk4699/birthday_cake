from shapely.geometry import Point, LineString
from shapely.ops import split
from players.player import Player
from src.cake import Cake
import src.constants as c


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

    def get_cuts(self) -> list[tuple[Point, Point]]:
        total_area = self.cake.exterior_shape.area
        total_crust = self.cake.exterior_shape.length

        target_area = total_area / self.children
        target_areas = [target_area for _ in range(self.children - 1)]
        target_ratio = total_crust / total_area

        moves: list[tuple[Point, Point]] = []

        for cut_index, t_area in enumerate(target_areas):
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
                    ratio_score = max(abs(ratio_1 - target_ratio), abs(ratio_2 - target_ratio))

                    # Weights - set ratio weight to vary within a range of 0.05 to 0.2
                    # As number of children increase, the ratio weight increases
                    # Area weight is proportional to ratio weight
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

            # Debug print for chosen cut
            print(
                f"Chosen cut {cut_index + 1}: best_score={best_score:.4f}, "
                f"target_area={t_area:.2f}, p1={best_cut[0]}, p2={best_cut[1]}"
            )

            # Apply the cut
            self.cake.cut(best_cut[0], best_cut[1])
            moves.append(best_cut)

        return moves
