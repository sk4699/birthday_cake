from shapely.geometry import Point, Polygon
from typing import Optional
from math import cos, sin, radians

from players.player import Player
from src.cake import Cake


class Player8(Player):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        super().__init__(children, cake, cake_path)

    def get_cuts(self) -> list[tuple[Point, Point]]:
        working_cake = self.cake.copy()
        all_cuts = []

        def divide(pieces_needed: int, target_piece: Polygon):
            if pieces_needed == 1:  # base case
                return

            cut = self.find_good_cut(target_piece, working_cake)  # find good cut
            if not cut:
                return

            from_p, to_p = cut
            new_pieces = working_cake.cut_piece(target_piece, from_p, to_p)  # get the new pieces post-cut
            all_cuts.append((from_p, to_p))

            divide(pieces_needed // 2, new_pieces[0])
            divide(pieces_needed - pieces_needed // 2, new_pieces[1])  # recurse on each of those two pieces, to cut each into ~ half the number of slices we need

        piece = working_cake.get_pieces()[0]
        divide(self.children, piece)

        return all_cuts
    

    def find_good_cut(self, piece: Polygon, working_cake: Cake) -> Optional[tuple[Point, Point]]:
        centroid = piece.centroid
        radius = max(centroid.distance(Point(p)) for p in piece.exterior.coords) * 2
        #  make sure cut line extends beyond polygon boundary  (double check this)

        def get_cut_points(angle_deg: float) -> tuple[Point, Point]:
            # given an angle, give back points for a cut that goes through the centroid at that angle
            angle_rad = radians(angle_deg)  # easier to work in radians
            dx = cos(angle_rad)  # x component of unit vector of angle
            dy = sin(angle_rad)  # y component of unit vector of angle
            #  get the start and end points of the cut, using centroid as base
            from_p = Point(centroid.x - dx * radius, centroid.y - dy * radius)
            to_p = Point(centroid.x + dx * radius, centroid.y + dy * radius)
            return from_p, to_p

        def score_cut(from_p: Point, to_p: Point) -> tuple[int, float]:
            #  prioritizing equal area first and foremost
            #  second, equal crust ratio among cuts with equal area
            #  returns two part score
            #  (0, ratio_diff) means good area equality, now think about crust
            #  (1, area_diff) means choose best area equality, don't worry about crust
            #  (2, float("inf")) means bad cut
            #  this will help for sorting cuts to choose the best!

            bound = piece.boundary
            a = bound.interpolate(bound.project(from_p))  # making sure cut is right length
            b = bound.interpolate(bound.project(to_p))

            from_p = a
            to_p = b

            is_valid, reason = working_cake.cut_is_valid(from_p, to_p)  # first just check whether cut is valid
            if not is_valid:
                print("invalid: ", reason)
                return (2, float("inf"))

            try:
                cut_pieces = working_cake.cut_piece(piece, from_p, to_p)

                if len(cut_pieces) != 2:  # should be cut into 2 pieces, something's wrong!
                    return (2, float("inf"))

                areas = [p.area for p in cut_pieces]
                ratios = [working_cake.get_piece_ratio(p) for p in cut_pieces]

                area_diff = abs(areas[0] - areas[1])
                ratio_diff = abs(ratios[0] - ratios[1])

                if area_diff < 0.5:  # window of tolerance
                    return (0, ratio_diff)
                else:
                    return (1, area_diff)
            except Exception as e:
                print("Exception in cut_piece or scoring:", e)
                return (2, float("inf"))

        best_score = (2, float("inf"))
        best_cut = None

        for angle in range(0, 180, 2):  # check every 2 angles from 0 to 180 - brute forcing for now
            from_p, to_p = get_cut_points(angle)
            score = score_cut(from_p, to_p)

            if score < best_score:  # tuple comparison works right here
                best_score = score
                best_cut = (from_p, to_p)

        if best_cut is not None:
            bound = piece.boundary
            from_p = bound.interpolate(bound.project(best_cut[0]))  # making sure cut is right length
            to_p = bound.interpolate(bound.project(best_cut[1]))
            best_cut = (from_p, to_p)
        else:
            print("No valid cut found for this piece")

        return best_cut
    