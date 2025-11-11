import re
from typing import List

def mask_text_table_lines(lines: List[str], min_cols: int = 3, min_block: int = 3) -> List[bool]:
    def col_gaps(s: str) -> int:
        return len(re.findall(r"(\t|\s{2,})", s))
    mask = [False]*len(lines)
    i = 0
    while i < len(lines):
        j = i
        block = []
        while j < len(lines):
            s = lines[j].strip()
            if s and col_gaps(s) >= (min_cols - 1):
                block.append(j)
                j += 1
            else:
                break
        if len(block) >= min_block:
            for k in block:
                mask[k] = True
            i = j
        else:
            i += 1
    return mask
