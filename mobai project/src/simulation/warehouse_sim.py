import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.optimization.optimizer import PLACE_WEIGHTS
from src.optimization.pathfinder import loc_to_coord, heuristic

DOCK = (0, 0)
CHARIOTS = [
    {"id": "C1", "capacity": 3, "position": DOCK, "busy_until": None},
    {"id": "C2", "capacity": 1, "position": DOCK, "busy_until": None},
    {"id": "C3", "capacity": 1, "position": DOCK, "busy_until": None},
]

class WarehouseSimulation:
    def __init__(self, excel_path, eval_loc_path):
        self.excel_path = excel_path
        self.eval_loc_path = eval_loc_path
        self.warehouse = {}
        self.operations = []
        self.stats = {'receipts': 0, 'deliveries': 0, 'overflows': 0, 'total_dist': 0}
        self.load_locations()
        
    def load_locations(self):
        df = pd.read_csv(self.eval_loc_path)
        for _, r in df.iterrows():
            code = str(r['code_emplacement'])
            self.warehouse[code] = {
                'code': code,
                'id': int(r['id_emplacement']),
                'type': r['type_emplacement'],
                'zone': r['zone'],
                'occupied': (r['actif'] == True),
                'product_id': None
            }

    def parse_slot(self, code):
        parts = code.split('-')
        if len(parts) < 3: return None
        lp = parts[0]
        if len(lp) < 2 or not lp[0].isdigit(): return None
        return {'level': lp[0], 'place': lp[1], 'side': parts[1], 'block': parts[2]}

    def validate_override(self, pid, slot_code, product_weight=0, product_entropy=1.0):
        """
        Returns (is_possible, message, status_type)
        status_type: 'OK', 'WARNING', 'IMPOSSIBLE'
        """
        if slot_code not in self.warehouse:
            return False, f"Slot {slot_code} does not exist.", "IMPOSSIBLE"
            
        slot = self.warehouse[slot_code]
        
        # 1. Check Occupancy
        if slot['occupied']:
            return False, f"Slot {slot_code} is already occupied.", "IMPOSSIBLE"
            
        # 2. Check Type
        if slot['type'] != 'PICKING':
             # Maybe possible but warning? Usually picking flow wants picking slots.
             pass 

        # 3. Check Constraints (Weight)
        parsed = self.parse_slot(slot_code)
        try:
            level = int(parsed['level'])
        except:
            level = 0
            
        if product_weight > 20 and level > 1:
            return False, f"Impossible: Product too heavy ({product_weight}kg) for level {level}.", "IMPOSSIBLE"
            
        # 4. Warnings (Entropy/Zone)
        warnings = []
        
        # Entropy check: High entropy (frequently ordered) should stay close/low
        # If entropy high (>2) and place is 'far' (e.g. weight > 5 in PLACE_WEIGHTS)
        place_cost = PLACE_WEIGHTS.get(parsed['place'], 5)
        if product_entropy > 1.5 and place_cost > 6:
            warnings.append(f"High demand product in far location (Zone {parsed['place']}).")
            
        # If warnings exist
        if warnings:
            return True, "Warning: " + " ".join(warnings), "WARNING"
            
        return True, "Placement is valid.", "OK"

    def find_best_slot(self, pid, weight=0.0, demand=0.0, entropy=1.0):
        candidates = []
        for code, slot in self.warehouse.items():
            if slot['occupied'] or slot['type'] != 'PICKING': continue
            
            parsed = self.parse_slot(code)
            if not parsed: continue
            
            try: lvl = int(parsed['level'])
            except: lvl = 0
            
            if weight > 20 and lvl > 1: continue
            
            dist = PLACE_WEIGHTS.get(parsed['place'], 10)
            prio = demand / (entropy + 0.1)
            score = (lvl * 25) + dist - prio * 0.01
            candidates.append((score, code))
            
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
            
        reserve = [c for c, s in self.warehouse.items() if not s['occupied'] and s['type'] == 'RESERVE']
        if reserve: return reserve[0]
        
        return None

    def find_product_loc(self, pid):
        for code, slot in self.warehouse.items():
            if slot['product_id'] == pid and slot['occupied']:
                return code
        return None

    def get_chariot(self, current_time):
        available = []
        for c in CHARIOTS:
            if c['busy_until'] is None or c['busy_until'] <= current_time:
                available.append(c)
        
        if available:
            available.sort(key=lambda x: x['capacity'], reverse=True)
            return available[0]
        
        return min(CHARIOTS, key=lambda x: x['busy_until'] if x['busy_until'] else current_time)

    def calc_dist(self, a, b):
        try:
            ca = loc_to_coord(a) if isinstance(a, str) else a
            cb = loc_to_coord(b) if isinstance(b, str) else b
            return heuristic(ca, cb)
        except:
            return 10

    def run(self, tx_df, lines_df, products_df, forecast_map, entropy_map, manual_overrides=None):
        print("Running Simulation...")
        if manual_overrides is None: manual_overrides = {}
        
        if manual_overrides:
            print(f"Applying {len(manual_overrides)} manual overrides.")
        
        p_weights = {}
        for _, r in products_df.iterrows():
            try: p_weights[int(r['id_produit'])] = float(r['weight_kg'])
            except: pass
            
        # Ensure chronological order
        tx_df = tx_df.sort_values('cree_le')

        for _, tx in tx_df.iterrows():
            tid = tx['id_transaction']
            ttype = tx['type_transaction']
            try:
                ttime = pd.to_datetime(tx['cree_le'])
            except:
                continue
            
            items = lines_df[lines_df['id_transaction'] == tid]
            
            for _, line in items.iterrows():
                if pd.isna(line['id_produit']): continue
                pid = int(line['id_produit'])
                try: qty = int(line['quantite'])
                except: qty = 1
                
                w = p_weights.get(pid, 0)
                d = forecast_map.get(pid, 0)
                e = entropy_map.get(pid, 1.0)
                
                chariot = self.get_chariot(ttime)
                
                dist = 0
                flow_type = ""
                
                if ttype == 'RECEIPT':
                    # Check override first
                    target = None
                    if pid in manual_overrides:
                        desired_slot = manual_overrides[pid]
                        # We force it if it's strictly valid/possible (we assume user confirmed constraints, but we check occupancy again to be safe during runtime as state changes)
                        if self.warehouse.get(desired_slot, {}).get('occupied', True) == False:
                            target = desired_slot
                            print(f"Override applied: {pid} -> {desired_slot}")
                        else:
                            print(f"Override failed: {desired_slot} occupied during runtime.")
                    
                    if not target:
                        target = self.find_best_slot(pid, w, d, e)
                    
                    flow_type = "Ingoing"
                    
                    if not target:
                        self.log_op(ttime, pid, flow_type, qty)
                        self.stats['overflows'] += 1
                        continue
                        
                    self.warehouse[target]['occupied'] = True
                    self.warehouse[target]['product_id'] = pid
                    
                    dist = self.calc_dist(DOCK, target)
                    self.stats['receipts'] += 1
                    
                elif ttype == 'DELIVERY':
                    flow_type = "Outgoing"
                    src = self.find_product_loc(pid)
                    if not src:
                        src = str(line['src'])
                        
                    if src in self.warehouse:
                        self.warehouse[src]['occupied'] = False
                        self.warehouse[src]['product_id'] = None
                        
                    dist = self.calc_dist(src, DOCK)
                    self.stats['deliveries'] += 1

                # Update Chariot Busy Time
                travel_time = timedelta(minutes=dist)
                start_time = max(ttime, chariot['busy_until']) if chariot['busy_until'] else ttime
                chariot['busy_until'] = start_time + travel_time
                
                self.stats['total_dist'] += dist
                
                # Log strictly according to requested format
                self.log_op(ttime, pid, flow_type, qty)

    def log_op(self, time, pid, flow_type, quantity):
        self.operations.append({
            'Date & Time': time.strftime('%d-%m-%Y %H:%M'),
            'Product ID': pid,
            'Flow Type': flow_type,
            'Quantity': quantity
        })

    def get_results(self):
        return {
            'operations': self.operations,
            'summary': self.stats,
            'final_warehouse': {
                'occupied': sum(1 for s in self.warehouse.values() if s['occupied']),
                'free': sum(1 for s in self.warehouse.values() if not s['occupied'])
            }
        }
