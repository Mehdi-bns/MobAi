import numpy as np
from scipy.optimize import linear_sum_assignment

COMPLEXITY_THRESHOLD = 500

PLACE_WEIGHTS = {
    'A': 1, 'B': 1, 'C': 2, 'D': 2, 'E': 3, 'F': 3,
    'G': 4, 'H': 4, 'I': 5, 'K': 5, 'M': 6, 'N': 6,
    'P': 7, 'Q': 7, 'R': 8, 'S': 8, 'T': 9, 'V': 9, 'W': 10
}

def slot_cost(sku, slot):
    place = slot['place']
    try:
        level = int(slot['level'])
    except:
        level = 0
    weight = sku.get('weight_kg', 0)
    
    if weight > 20 and level > 1:
        return 1e9
        
    demand = sku.get('forecast_7d', 0)
    entropy = sku.get('entropy', 2.0)
    priority = demand / (entropy + 0.1)
    dist = PLACE_WEIGHTS.get(place, 10)
    
    return (level * 25) + dist - priority * 0.01

def hungarian_assign(skus, slots):
    n = len(skus)
    m = len(slots)
    cost = np.full((n, m), 1e12)
    for i, sku in enumerate(skus):
        for j, slot in enumerate(slots):
            cost[i][j] = slot_cost(sku, slot)
    
    row_ind, col_ind = linear_sum_assignment(cost)
    placements = {}
    for r, c in zip(row_ind, col_ind):
        if cost[r][c] >= 1e9:
            placements[str(skus[r]['id'])] = "OVERFLOW"
        else:
            s = slots[c]
            placements[str(skus[r]['id'])] = f"{s['place']}{s['level']}-{s['side']}-{s['block']}"
    return placements

def greedy_assign(skus, slots):
    scored = []
    for sku in skus:
        demand = sku.get('forecast_7d', 0)
        entropy = sku.get('entropy', 2.0)
        sku['_priority'] = demand / (entropy + 0.1)
        scored.append(sku)
    scored.sort(key=lambda x: x['_priority'], reverse=True)
    
    used = set()
    aisle_load = {p: 0 for p in PLACE_WEIGHTS}
    placements = {}
    
    for sku in scored:
        best_score = 1e18
        best_idx = -1
        
        for j, slot in enumerate(slots):
            if j in used: continue
            
            c = slot_cost(sku, slot)
            c += aisle_load.get(slot['place'], 0) * 5
            
            if c < best_score:
                best_score = c
                best_idx = j
                
        if best_idx == -1 or best_score >= 1e9:
            placements[str(sku['id'])] = "OVERFLOW"
        else:
            s = slots[best_idx]
            placements[str(sku['id'])] = f"{s['place']}{s['level']}-{s['side']}-{s['block']}"
            used.add(best_idx)
            if sku['_priority'] > 50:
                aisle_load[s['place']] = aisle_load.get(s['place'], 0) + 1
                
    return placements

def optimize(skus, slots):
    complexity = len(skus) * len(slots)
    if complexity <= COMPLEXITY_THRESHOLD:
        return hungarian_assign(skus, slots), "hungarian"
    return greedy_assign(skus, slots), "greedy"
