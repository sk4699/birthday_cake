from shapely.geometry import Point, LineString
from shapely.ops import split
from players.player import Player
from src.cake import Cake
import src.constants as c
import math


class Player8(Player):
    """
    Player8 - Adaptive boundary search strategy with global crust balance feedback.
    Inspired by Player9 logic.

    This strategy adaptively selects cuts around the cake’s boundary to balance
    both area and crust ratio across pieces. It evaluates possible cuts based
    on a weighted combination of area accuracy, crust difference, and global
    balance to achieve fair divisions.
    """

    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)
        self.area_weight = 0.7
        self.crust_weight = 0.23


    # -------------------------------------------------------------
    # Local refinement of a chosen cut
    # -------------------------------------------------------------
    def _refine_cut(self, piece, p1, p2, t_area, delta=0.01, steps=4):
        """
        Fine-tunes a proposed cut by making small local adjustments to its endpoints
        to minimize the difference between the smaller piece’s area and the target.
        """
        boundary = piece.boundary

        def move(pt, step):
            s = boundary.project(pt)
            s_new = min(max(0.0, s + step * delta * boundary.length), boundary.length)
            return boundary.interpolate(s_new)

        best = (
            p1,
            p2,
            abs(min(split(piece, LineString([p1, p2])).geoms, key=lambda g: g.area).area - t_area),
        )

        for _ in range(steps):
            improved = False
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    q1, q2 = move(best[0], s1), move(best[1], s2)
                    parts = split(piece, LineString([q1, q2]))
                    if len(parts.geoms) != 2:
                        continue
                    small = min(parts.geoms[0].area, parts.geoms[1].area)
                    err = abs(small - t_area)
                    if err < best[2]:
                        best = (q1, q2, err)
                        improved = True
            if not improved:
                break
        return best[0], best[1]

    # -------------------------------------------------------------
    # Main cutting logic
    # -------------------------------------------------------------
    def get_cuts(self) -> list[tuple[Point, Point]]:
        """
        Iteratively selects cuts that minimize a composite score
        based on area fairness and crust ratio deviation from global mean,
        with emphasis on the larger piece earlier in the process.
        """
        total_area = self.cake.exterior_shape.area
        target_area = total_area / self.children
        target_areas = [target_area for _ in range(self.children - 1)]
        moves: list[tuple[Point, Point]] = []

        for cut_index, t_area in enumerate(target_areas):
            piece = max(self.cake.get_pieces(), key=lambda p: p.area)
            boundary = piece.boundary
            boundary_len = boundary.length

            # Adjust candidate density by stage
            if cut_index == 0:
                num_candidates = min(1200, max(300, int(boundary_len / 0.4)))
            else:
                num_candidates = max(400, min(220, int(boundary_len / 1.2)))

            step = boundary.length / num_candidates
            points = [boundary.interpolate(i * step) for i in range(num_candidates)]

            best_score = float("inf")
            best_cut = None
            perfect_thresh = 0.001
            area_prune_thresh = 0.3

            # Evaluate global crust balance
            ratios = [self.cake.get_piece_ratio(p) for p in self.cake.get_pieces()]
            global_mean = sum(ratios) / len(ratios)

            candidate_count = 0
            fallback_best = (None, float("inf"))

            for i in range(num_candidates):
                for j in range(i + 1, num_candidates):
                    p1, p2 = points[i], points[j]
                    valid, _ = self.cake.cut_is_valid(p1, p2)
                    if not valid:
                        continue

                    line = LineString([p1, p2])
                    parts = split(piece, line)
                    if len(parts.geoms) != 2:
                        continue

                    a1, a2 = parts.geoms
                    a1_area, a2_area = a1.area, a2.area
                    small = min(a1_area, a2_area)
                    area_diff = abs(small - t_area)

                    # Strict near-target filter ±0.25 cm²
                    if area_diff <= 0.25:
                        candidate_count += 1
                    else:
                        # Track fallback (closest cut if no near-targets exist)
                        if area_diff < fallback_best[1]:
                            fallback_best = ((p1, p2), area_diff)
                        continue

                    if abs(small - t_area) > t_area * area_prune_thresh:
                        continue
                    if a1_area < c.MIN_PIECE_AREA or a2_area < c.MIN_PIECE_AREA:
                        continue

                    # Area error
                    area_error = abs(small - t_area) / t_area

                    # --- Crust ratio scoring relative to global mean ---
                    r1 = self.cake.get_piece_ratio(a1)
                    r2 = self.cake.get_piece_ratio(a2)

                    total = a1_area + a2_area
                    a1_frac = a1_area / total
                    a2_frac = a2_area / total

                    if a1_area > a2_area:
                        larger_r_diff = abs(r1 - global_mean)
                        smaller_r_diff = abs(r2 - global_mean)
                        larger_frac = a1_frac
                        smaller_frac = a2_frac
                    else:
                        larger_r_diff = abs(r2 - global_mean)
                        smaller_r_diff = abs(r1 - global_mean)
                        larger_frac = a2_frac
                        smaller_frac = a1_frac

                    stage_progress = cut_index / max(1, self.children - 1)
                    larger_weight = 0.65 - 0.15 * stage_progress # earlier on, care more about whether leftover piece has decent crust ratio
                    smaller_weight = 1.0 - larger_weight

                    weighted_crust_diff = (
                        larger_r_diff * larger_frac * larger_weight +
                        smaller_r_diff * smaller_frac * smaller_weight
                    )

                    self.last_area_error = area_error
                    self.last_crust_error = weighted_crust_diff

                    # Dynamic weighting
                    alpha = cut_index / max(1, self.children - 1)
                    base_area_w = getattr(self, "area_weight_override", self.area_weight)
                    base_crust_w = getattr(self, "crust_weight_override", self.crust_weight)
                    area_weight = base_area_w + (1 - alpha) * 0.1
                    crust_weight = base_crust_w * (1 + 0.5 * alpha)

                    # First-cut bias
                    if cut_index == 0:
                        area_weight *= 0.8
                        crust_weight *= 1.6

                    score = (
                        area_weight * area_error +
                        crust_weight * weighted_crust_diff
                    )

                    if cut_index == 0 and small > t_area:
                        score *= 1.25

                    if area_error > 0.3:
                        continue

                    if score < best_score:
                        best_score = score
                        best_cut = (p1, p2)
                        if area_error < perfect_thresh and weighted_crust_diff < 0.01:
                            break
                if best_score < float("inf") and best_score < perfect_thresh:
                    break


            # Fallback if no near-target candidates were found
            if best_cut is None:
                if candidate_count == 0 and fallback_best[0] is not None:
                    best_cut = fallback_best[0]
                    print(f"No near-target candidates; using closest match ({fallback_best[1]:.3f})")
                else:
                    print(f"[WARN] No valid cut found at step {cut_index + 1}")
                    break

            # Final refinement
            best_cut = self._refine_cut(piece, best_cut[0], best_cut[1], t_area)

            # Apply chosen cut
            self.cake.cut(best_cut[0], best_cut[1])
            moves.append(best_cut)

            # Debug print
            # After applying the cut
            split_geoms = split(piece, LineString(best_cut)).geoms

            # Compute ratio error for the larger piece
            a1, a2 = split_geoms
            a1_area, a2_area = a1.area, a2.area
            r1 = self.cake.get_piece_ratio(a1)
            r2 = self.cake.get_piece_ratio(a2)

            # Print result
            def pct_diff(a, b):
                return abs(a - b) / max(abs(a), abs(b), 1e-6)  # avoid division by zero

            if pct_diff(r1, global_mean) <= 0.05 or pct_diff(r2, global_mean) <= 0.05:
                print(
                    f"Candidate near crust ratio at cut {cut_index + 1}: "
                    f"Points ({p1.x:.3f},{p1.y:.3f})-({p2.x:.3f},{p2.y:.3f}), "
                    f"Larger_r_diff={larger_r_diff:.3f}, Smaller_r_diff={smaller_r_diff:.3f}, "
                    f"Area error={area_error:.4f}"
                )

        # Save cuts and run post-processing wiggle
        self.cuts = moves
        print("Applying wiggle refinement...")
        self._wiggle_cuts(iterations=3, delta=0.02)
        print("Performing global rebalance sweep...")
        self._wiggle_cuts(iterations=2, delta=0.04)

        return moves


    # -------------------------------------------------------------
    # Post-processing: crust ratio smoothing
    # -------------------------------------------------------------
    def _wiggle_cuts(self, iterations=3, delta=0.02):
        """
        Gently adjusts cut endpoints to improve crust ratio balance
        WITHOUT altering overall area fairness.
        Works on temporary cake copies and only applies changes
        if both ratio std improves and area span stays stable.
        """
        if not hasattr(self, "cuts") or not self.cuts:
            print("No cuts to wiggle.")
            return

        boundary = self.cake.exterior_shape.boundary
        pieces = self.cake.get_pieces()
        before_ratios = [self.cake.get_piece_ratio(p) for p in pieces]
        before_mean = sum(before_ratios) / len(before_ratios)
        before_std = (sum((r - before_mean) ** 2 for r in before_ratios) / len(before_ratios)) ** 0.5
        before_areas = [p.area for p in pieces]
        before_span = max(before_areas) - min(before_areas)
        print(f"[WIGGLE] Start: ratio std={before_std:.4f}, area span={before_span:.4f}")

        def move(pt, step, step_size):
            s = boundary.project(pt)
            s_new = min(max(0.0, s + step * step_size * boundary.length), boundary.length)
            return boundary.interpolate(s_new)

        best_config = list(self.cuts)
        best_std = before_std
        best_span = before_span

        for it in range(iterations):
            step_size = delta * (1.0 + 0.5 * it)
            improved = False

            for i, (p1, p2) in enumerate(best_config):
                for s1 in (-1, 1):
                    for s2 in (-1, 1):
                        q1, q2 = move(p1, s1, step_size), move(p2, s2, step_size)
                        test_cake = self.cake.copy()
                        try:
                            test_cake.cut(q1, q2)
                        except Exception:
                            continue

                        test_pieces = test_cake.get_pieces()
                        ratios = [test_cake.get_piece_ratio(p) for p in test_pieces]
                        mean_r = sum(ratios) / len(ratios)
                        std_new = (sum((r - mean_r) ** 2 for r in ratios) / len(ratios)) ** 0.5

                        areas = [p.area for p in test_pieces]
                        span_new = max(areas) - min(areas)

                        # accept only if ratio improved and area span didn't blow up
                        if std_new < best_std and span_new <= best_span + 0.1:
                            print(f" Wiggle improved std {best_std:.4f}→{std_new:.4f}, span {best_span:.4f}→{span_new:.4f}")
                            best_config[i] = (q1, q2)
                            best_std, best_span = std_new, span_new
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break

        # Apply only the final best configuration
        if best_std < before_std:
            self.cake = self.cake.copy()
            for p1, p2 in best_config:
                try:
                    self.cake.cut(p1, p2)
                except Exception:
                    continue
            self.cuts = best_config
            print(f"[WIGGLE] Final: ratio std={best_std:.4f}, area span={best_span:.4f}")
        else:
            print("[WIGGLE] No improvement found; keeping original cuts.")
