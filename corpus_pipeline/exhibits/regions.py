from typing import Tuple, Optional

def compute_exhibit_region(caption_bbox: Tuple[float,float,float,float],
                           page_height: float,
                           next_caption_bbox: Optional[Tuple[float,float,float,float]] = None,
                           min_height: float = 120.0) -> Tuple[float,float,float,float]:
    x0,y0,x1,y1 = caption_bbox
    top = y1 + 6
    bottom = next_caption_bbox[1] - 6 if next_caption_bbox else page_height - 6
    if bottom - top < min_height:
        bottom = min(page_height - 6, top + min_height)
    return (x0, top, x1, bottom)
