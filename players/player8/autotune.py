import numpy as np
from players.player8 import Player

def evaluate_weights(area_w, crust_w, cakes, children):
    """
    Run the cut algorithm on multiple cakes with given weights,
    return average combined error (area + crust deviation).
    """
    total_error = 0
    for cake in cakes:
        cutter = Player(cake, children)
        cutter.area_weight = area_w
        cutter.crust_weight = crust_w
        moves = cutter.get_cuts()
        
        # Example metric: average area error + crust ratio error over all cuts
        error_sum = 0
        for cut_index, cut in enumerate(moves):
            # You need to compute or access area error and crust ratio error per cut
            # For example, you can modify your code to save those errors per cut and access them here.
            error_sum += cutter.last_area_error + cutter.last_crust_error  
        total_error += error_sum / len(moves)
    return total_error / len(cakes)


def autotune_weights(cakes, children, area_range, crust_range, steps=10):
    best_score = float('inf')
    best_params = (None, None)
    
    area_ws = np.linspace(area_range[0], area_range[1], steps)
    crust_ws = np.linspace(crust_range[0], crust_range[1], steps)
    
    for aw in area_ws:
        for cw in crust_ws:
            score = evaluate_weights(aw, cw, cakes, children)
            print(f"Testing area_w={aw:.3f}, crust_w={cw:.3f} => score={score:.4f}")
            if score < best_score:
                best_score = score
                best_params = (aw, cw)
    print(f"Best weights found: area_weight={best_params[0]:.3f}, crust_weight={best_params[1]:.3f}")
    return best_params

sample_cakes = []

best_area_w, best_crust_w = autotune_weights(
    cakes=sample_cakes,
    children=4,
    area_range=(0.1, 1.0),
    crust_range=(0.1, 2.0),
    steps=10
)