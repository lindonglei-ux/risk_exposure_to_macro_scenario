import re
from typing import Optional
tokenizer = None
have_tf = False
try:
    from transformers import AutoTokenizer
    have_tf = True
except Exception:
    pass
def count_tokens(s: str, tokenizer_name: Optional[str] = None) -> int:
    global tokenizer
    if have_tf and tokenizer_name:
        if tokenizer is None or getattr(tokenizer, "name_or_path", None) != tokenizer_name:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception:
                tokenizer = None
    if tokenizer:
        return len(tokenizer.encode(s, add_special_tokens=False))
    words = re.findall(r"\w+(\.\w+)?", s)
    return int(len(words) * 1.3)
