from typing import Tuple, Optional

def compute_exhibit_region(caption_bbox: Tuple[float,float,float,float],
                           page_height: float,
                           next_caption_bbox: Optional[Tuple[float,float,float,float]] = None,
                           min_height: float = 120.0,
                           page_width: Optional[float] = None,
                           min_width: float = 140.0,
                           margin: float = 12.0) -> Tuple[float,float,float,float]:
    x0,y0,x1,y1 = caption_bbox
    top = y1 + 6
    bottom = next_caption_bbox[1] - 6 if next_caption_bbox else page_height - 6
    if bottom - top < min_height:
        bottom = min(page_height - 6, top + min_height)
    width = x1 - x0
    center = (x0 + x1) / 2.0
    half_width = max(width / 2.0, min_width / 2.0)
    left = center - half_width
    right = center + half_width
    if page_width is not None:
        left = max(0.0, left)
        right = min(page_width, right)
    left = max(0.0, left - margin)
    right = right + margin if page_width is None else min(page_width, right + margin)
    return (left, top, right, bottom)
