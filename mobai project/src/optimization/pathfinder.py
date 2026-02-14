import heapq
import requests

CUOPT_URL = "https://optimize.api.nvidia.com/v1/cuopt"
CUOPT_API_KEY = None

ROW_ORDER = list("ABCDEFGHIKMNPQRSTVWX")
ROW_MAP = {c: i for i, c in enumerate(ROW_ORDER)}

def loc_to_coord(loc_code):
    parts = loc_code.split("-")
    letter = parts[0][0]
    level = int(parts[0][1])
    side = int(parts[1])
    block = int(parts[2])
    row = ROW_MAP.get(letter, 0) * 4 + level
    col = (side - 1) * 10 + block
    return (row, col)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal, obstacles=None):
    if obstacles is None: obstacles = set()
    open_set = [(0, start)]
    came_from = {}
    g = {start: 0}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
            
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nb = (current[0] + dx, current[1] + dy)
            if nb[0] < 0 or nb[1] < 0 or nb in obstacles: continue
            
            ng = g[current] + 1
            if ng < g.get(nb, float('inf')):
                g[nb] = ng
                f = ng + heuristic(nb, goal)
                heapq.heappush(open_set, (f, nb))
                came_from[nb] = current
    return [start, goal]

def try_cuopt(locations):
    if not CUOPT_API_KEY: return None
    # Skips logic if no key
    return None

def find_path(locations, dock=(0, 0)):
    cuopt_result = try_cuopt(locations)
    if cuopt_result:
        return {"method": "cuopt", "route": cuopt_result}
        
    coords = [loc_to_coord(loc) for loc in locations]
    indexed = sorted(enumerate(coords), key=lambda x: (x[1][0], x[1][1]))
    ordered_locs = [locations[i] for i, _ in indexed]
    ordered_coords = [c for _, c in indexed]
    
    full_path = []
    current = dock
    total_dist = 0
    for coord in ordered_coords:
        segment = astar(current, coord)
        total_dist += len(segment) - 1
        full_path.extend(segment[:-1])
        current = coord
        
    back = astar(current, dock)
    total_dist += len(back) - 1
    full_path.extend(back[1:])
    
    return {
        "method": "astar",
        "route": ordered_locs,
        "total_distance": total_dist,
        "path_points": len(full_path)
    }
