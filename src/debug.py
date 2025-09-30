import tkinter as tk

import constants as c
from shapely import Polygon, LineString


def DEBUG_show_polygons(
    polygons: list[Polygon],
    line: LineString | None = None,
    size=(700, 600),
):
    scaler = 20

    root = tk.Tk()
    root.title("Polygon Viewer")
    W, H = size

    canvas = tk.Canvas(root, width=W, height=H, bg="white", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    polygons.sort(key=lambda x: x.area, reverse=True)
    for polygon in polygons:
        xys = list(polygon.exterior.coords[:-1])
        ext_points = [c * scaler for xy in xys for c in xy]
        canvas.create_polygon(ext_points, outline="black", fill=c.CAKE_CRUST, width=2)

    for idx, polygon in enumerate(polygons):
        centroid = polygon.centroid
        canvas.create_text(
            (centroid.x + idx) * scaler, (centroid.y + idx) * scaler, text=f"{idx}"
        )

    if line:
        line_coords = [num * scaler for (x, y) in line.coords for num in (x, y)]
        canvas.create_line(line_coords, fill="blue", width=2)

    root.mainloop()
