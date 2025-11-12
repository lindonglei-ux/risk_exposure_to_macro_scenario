from typing import List, Dict, Any
def reorder_two_columns(blocks: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    if not blocks: return blocks
    centers = [(b["x0"]+b["x1"])/2 for b in blocks]
    med = sorted(centers)[len(centers)//2]
    left  = [b for b in blocks if (b["x0"]+b["x1"])/2 <= med]
    right = [b for b in blocks if (b["x0"]+b["x1"])/2 >  med]
    left.sort(key=lambda b: (round(b["y0"],1), round(b["x0"],1)))
    right.sort(key=lambda b: (round(b["y0"],1), round(b["x0"],1)))
    return left + right
