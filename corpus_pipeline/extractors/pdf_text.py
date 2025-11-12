from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

have_fitz = False
try:  # pragma: no cover - optional dependency
    import fitz  # PyMuPDF type: ignore

    have_fitz = True
except Exception:  # pragma: no cover - handled by fallback
    fitz = None


@dataclass
class _PDFObject:
    obj_id: int
    generation: int
    dictionary: Optional[Dict[str, Any]]
    stream: Optional[bytes]
    value: Any = None


class _PDFString(str):
    """Marker class to distinguish literal strings from operators."""


class _PDFName(str):
    """Marker class for PDF name objects."""


_WHITESPACE = b"\x00\t\n\x0c\r "
_DELIMITERS = b"[]()<>/{}%"
_OPERATOR_TOKENS = {
    "BT",
    "ET",
    "Tf",
    "Td",
    "TD",
    "Tm",
    "Tj",
    "TJ",
    "T*",
    "'",
    '"',
    "Tw",
    "Tc",
    "TL",
}


class _TokenStream:
    def __init__(self, tokens: List[Any]):
        self._tokens = tokens
        self._idx = 0

    def next(self) -> Any:
        tok = self._tokens[self._idx]
        self._idx += 1
        return tok

    def peek(self) -> Any:
        if self._idx >= len(self._tokens):
            return None
        return self._tokens[self._idx]

    def has_more(self) -> bool:
        return self._idx < len(self._tokens)


def _decode_name(raw: bytes) -> str:
    out = bytearray()
    i = 0
    while i < len(raw):
        if raw[i:i + 1] == b"#" and i + 2 < len(raw):
            try:
                out.append(int(raw[i + 1:i + 3], 16))
                i += 3
                continue
            except ValueError:
                pass
        out.append(raw[i])
        i += 1
    return out.decode("latin1", errors="ignore")


def _read_literal_string(data: bytes, idx: int) -> Tuple[_PDFString, int]:
    depth = 1
    buf = bytearray()
    i = idx + 1
    while i < len(data) and depth > 0:
        ch = data[i:i + 1]
        if ch == b"\\":
            i += 1
            if i >= len(data):
                break
            esc = data[i:i + 1]
            if esc in b"nrtbf" + b"nrtbf":
                mapping = {
                    b"n": b"\n",
                    b"r": b"\r",
                    b"t": b"\t",
                    b"b": b"\b",
                    b"f": b"\f",
                }
                buf.extend(mapping.get(esc, esc))
            elif esc in b"()\\":
                buf.extend(esc)
            elif esc.isdigit():
                oct_digits = [esc]
                for _ in range(2):
                    if i + 1 < len(data) and data[i + 1:i + 2].isdigit():
                        i += 1
                        oct_digits.append(data[i:i + 1])
                    else:
                        break
                try:
                    buf.append(int(b"".join(oct_digits), 8) & 0xFF)
                except ValueError:
                    pass
            else:
                buf.extend(esc)
        elif ch == b"(":
            depth += 1
            buf.extend(ch)
        elif ch == b")":
            depth -= 1
            if depth == 0:
                i += 1
                break
            buf.extend(ch)
        else:
            buf.extend(ch)
        i += 1
    return _PDFString(buf.decode("utf-8", errors="ignore")), i


def _read_hex_string(data: bytes, idx: int) -> Tuple[_PDFString, int]:
    i = idx + 1
    buf = bytearray()
    hex_digits = bytearray()
    while i < len(data):
        ch = data[i:i + 1]
        if ch in b"\r\n\t \x00":
            i += 1
            continue
        if ch == b">":
            if len(hex_digits) % 2 == 1:
                hex_digits.append(ord("0"))
            if hex_digits:
                try:
                    buf.extend(bytes.fromhex(hex_digits.decode("ascii")))
                except ValueError:
                    pass
            i += 1
            break
        hex_digits.append(ch[0])
        if len(hex_digits) == 2:
            try:
                buf.append(int(hex_digits.decode("ascii"), 16))
            except ValueError:
                pass
            hex_digits.clear()
        i += 1
    return _PDFString(buf.decode("utf-8", errors="ignore")), i


def _tokenize(data: bytes) -> List[Any]:
    tokens: List[Any] = []
    i = 0
    n = len(data)
    while i < n:
        ch = data[i:i + 1]
        if not ch:
            break
        if ch in _WHITESPACE:
            i += 1
            continue
        if ch == b"%":
            while i < n and data[i:i + 1] not in (b"\n", b"\r"):
                i += 1
            continue
        if ch == b"(":
            val, i = _read_literal_string(data, i)
            tokens.append(val)
            continue
        if ch == b"<":
            if i + 1 < n and data[i + 1:i + 2] == b"<":
                tokens.append("<<")
                i += 2
            else:
                val, i = _read_hex_string(data, i)
                tokens.append(val)
            continue
        if ch == b">":
            if i + 1 < n and data[i + 1:i + 2] == b">":
                tokens.append(">>")
                i += 2
            else:
                i += 1
            continue
        if ch in (b"[", b"]"):
            tokens.append(ch.decode("latin1"))
            i += 1
            continue
        if ch == b"/":
            j = i + 1
            while j < n and data[j:j + 1] not in _WHITESPACE + _DELIMITERS:
                j += 1
            tokens.append(_PDFName(_decode_name(data[i + 1:j])))
            i = j
            continue
        if ch in b"+-0123456789.":
            j = i + 1
            while j < n and data[j:j + 1] not in _WHITESPACE + _DELIMITERS:
                j += 1
            num = data[i:j].decode("latin1")
            try:
                if any(c in num for c in (".", "e", "E")):
                    tokens.append(float(num))
                else:
                    tokens.append(int(num))
            except ValueError:
                tokens.append(num)
            i = j
            continue
        j = i + 1
        while j < n and data[j:j + 1] not in _WHITESPACE + _DELIMITERS:
            j += 1
        word = data[i:j].decode("latin1", errors="ignore")
        tokens.append(word)
        i = j
    return tokens


def _parse_array(stream: _TokenStream) -> List[Any]:
    items: List[Any] = []
    while stream.has_more():
        tok = stream.next()
        if tok == "]":
            break
        if tok == "[":
            items.append(_parse_array(stream))
        elif tok == "<<":
            items.append(_parse_dictionary(stream))
        else:
            if isinstance(tok, (int, float)) and isinstance(stream.peek(), (int, float)) and stream.peek() is not None:
                nxt = stream.peek()
                nxt2 = stream._tokens[stream._idx + 1] if stream._idx + 1 < len(stream._tokens) else None
                if nxt2 == "R":
                    stream.next()
                    stream.next()
                    items.append((int(tok), int(nxt)))
                    continue
            items.append(tok)
    return items


def _parse_value(stream: _TokenStream) -> Any:
    tok = stream.next()
    if tok == "[":
        return _parse_array(stream)
    if tok == "<<":
        return _parse_dictionary(stream)
    if isinstance(tok, (int, float)):
        nxt = stream.peek()
        nxt2 = stream._tokens[stream._idx + 1] if stream._idx + 1 < len(stream._tokens) else None
        if isinstance(nxt, (int, float)) and nxt2 == "R":
            stream.next()
            stream.next()
            return (int(tok), int(nxt))
        return tok
    if tok == "null":
        return None
    if tok == "true":
        return True
    if tok == "false":
        return False
    return tok


def _parse_dictionary(stream: _TokenStream) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    while stream.has_more():
        tok = stream.next()
        if tok == ">>":
            break
        if isinstance(tok, _PDFName):
            key = str(tok)
            if not stream.has_more():
                break
            result[key] = _parse_value(stream)
    return result


def _parse_pdf_dictionary(data: bytes) -> Optional[Dict[str, Any]]:
    data = data.strip()
    if not data.startswith(b"<<"):
        return None
    tokens = _tokenize(data)
    stream = _TokenStream(tokens)
    if stream.next() != "<<":
        return None
    return _parse_dictionary(stream)


def _decode_ascii_hex(data: bytes) -> bytes:
    cleaned = re.sub(rb"\s+", b"", data)
    cleaned = cleaned.rstrip(b">")
    if len(cleaned) % 2 == 1:
        cleaned += b"0"
    try:
        return bytes.fromhex(cleaned.decode("ascii", errors="ignore"))
    except ValueError:
        return data


def _decode_ascii85(data: bytes) -> bytes:
    cleaned = re.sub(rb"\s+", b"", data)
    if cleaned.endswith(b"~>"):
        cleaned = cleaned[:-2]
    try:
        import binascii

        return binascii.a85decode(cleaned, adobe=True)
    except Exception:
        return data


def _apply_filters(raw: bytes, filters: Any) -> bytes:
    if not raw:
        return raw
    filters_list: List[Any]
    if isinstance(filters, list):
        filters_list = filters
    elif filters is None:
        filters_list = []
    else:
        filters_list = [filters]
    data = raw
    for filt in filters_list:
        name = None
        if isinstance(filt, dict):
            name = filt.get("Name") or filt.get("name")
        if isinstance(filt, _PDFName):
            name = str(filt)
        if isinstance(filt, str):
            name = filt
        if not name:
            continue
        if name.endswith("Decode"):
            name = name[:-6]
        if name == "Flate":
            try:
                data = zlib.decompress(data)
            except Exception:
                pass
        elif name == "ASCIIHex":
            data = _decode_ascii_hex(data)
        elif name == "ASCII85":
            data = _decode_ascii85(data)
        else:
            # unsupported filter; leave as-is
            pass
    return data


class _SimplePDFDocument:
    def __init__(self, pdf_path: Path):
        self.path = pdf_path
        self._raw = pdf_path.read_bytes()
        self._objects: Dict[int, _PDFObject] = {}
        self.pages: List[Dict[str, Any]] = []
        self._parse_objects()
        self._build_pages()

    def _parse_objects(self) -> None:
        data = self._raw
        pattern = re.compile(rb"(\d+)\s+(\d+)\s+obj")
        pos = 0
        length = len(data)
        while True:
            m = pattern.search(data, pos)
            if not m:
                break
            obj_id = int(m.group(1))
            gen = int(m.group(2))
            start = m.end()
            end = data.find(b"endobj", start)
            if end == -1:
                break
            obj_content = data[start:end]
            pos = end + len(b"endobj")
            stream_pos = obj_content.find(b"stream")
            dictionary_bytes = obj_content
            stream_bytes = None
            if stream_pos != -1:
                dictionary_bytes = obj_content[:stream_pos]
                stream_data = obj_content[stream_pos + len(b"stream"):]
                if stream_data.startswith(b"\r\n"):
                    stream_data = stream_data[2:]
                elif stream_data.startswith(b"\n"):
                    stream_data = stream_data[1:]
                stream_end = stream_data.find(b"endstream")
                if stream_end != -1:
                    stream_bytes = stream_data[:stream_end]
                else:
                    stream_bytes = stream_data
            dictionary = _parse_pdf_dictionary(dictionary_bytes) if dictionary_bytes else None
            decoded_stream = None
            parsed_value: Any = None
            if stream_bytes is not None:
                filters = dictionary.get("Filter") if dictionary else None
                decoded_stream = _apply_filters(stream_bytes, filters)
                parsed_value = dictionary
            else:
                if dictionary is not None:
                    parsed_value = dictionary
                stripped = obj_content.strip()
                if stripped and parsed_value is None:
                    try:
                        tokens = _tokenize(stripped)
                        token_stream = _TokenStream(tokens)
                        if token_stream.has_more():
                            parsed_value = _parse_value(token_stream)
                    except Exception:
                        parsed_value = None
            obj_record = _PDFObject(
                obj_id,
                gen,
                dictionary,
                decoded_stream,
                parsed_value,
            )
            self._objects[obj_id] = obj_record
            if (
                dictionary
                and dictionary.get("Type") == "ObjStm"
                and decoded_stream is not None
            ):
                self._parse_object_stream(decoded_stream, dictionary)

    def _parse_object_stream(self, stream: bytes, dictionary: Dict[str, Any]) -> None:
        try:
            count = int(dictionary.get("N", 0))
            header_len = int(dictionary.get("First", 0))
        except (TypeError, ValueError):
            return
        if count <= 0 or header_len < 0 or header_len > len(stream):
            return
        header_bytes = stream[:header_len]
        data_bytes = stream[header_len:]
        try:
            header_text = header_bytes.decode("latin1", errors="ignore")
        except Exception:
            return
        header_parts = header_text.strip().split()
        if len(header_parts) < count * 2:
            return
        entries: List[Tuple[int, int]] = []
        for i in range(0, count * 2, 2):
            try:
                obj_no = int(header_parts[i])
                offset = int(header_parts[i + 1])
            except (ValueError, IndexError):
                continue
            entries.append((obj_no, offset))
        if not entries:
            return
        entries.sort(key=lambda itm: itm[1])
        for idx, (obj_no, offset) in enumerate(entries):
            if offset < 0 or offset >= len(data_bytes):
                continue
            next_offset = entries[idx + 1][1] if idx + 1 < len(entries) else len(data_bytes)
            if next_offset < offset:
                continue
            obj_data = data_bytes[offset:next_offset].strip()
            if not obj_data:
                continue
            parsed_value: Any = None
            try:
                tokens = _tokenize(obj_data)
                token_stream = _TokenStream(tokens)
                if token_stream.has_more():
                    parsed_value = _parse_value(token_stream)
            except Exception:
                parsed_value = None
            dict_value = parsed_value if isinstance(parsed_value, dict) else None
            if obj_no not in self._objects:
                self._objects[obj_no] = _PDFObject(
                    obj_no,
                    0,
                    dict_value,
                    None,
                    parsed_value,
                )

    def _resolve_ref(self, value: Any) -> Optional[_PDFObject]:
        if isinstance(value, tuple) and len(value) == 2:
            return self._objects.get(int(value[0]))
        if isinstance(value, int):
            return self._objects.get(value)
        return None

    def _build_pages(self) -> None:
        catalog = None
        for obj in self._objects.values():
            if obj.dictionary and obj.dictionary.get("Type") == "Catalog":
                catalog = obj
                break
        if not catalog:
            # fall back to any page-like objects
            for obj in self._objects.values():
                if obj.dictionary and obj.dictionary.get("Type") == "Page":
                    page_dict = {
                        "object": obj,
                        "MediaBox": obj.dictionary.get("MediaBox"),
                        "Contents": obj.dictionary.get("Contents"),
                        "Resources": obj.dictionary.get("Resources"),
                    }
                    self.pages.append(page_dict)
            return
        pages_root = self._resolve_ref(catalog.dictionary.get("Pages")) if catalog.dictionary else None
        if not pages_root:
            return
        self._collect_pages(pages_root, {})

    def _collect_pages(self, node: _PDFObject, inherited: Dict[str, Any]) -> None:
        node_dict = node.dictionary or {}
        inherited_next = dict(inherited)
        for key in ("MediaBox", "CropBox", "Resources", "Rotate"):
            if key in node_dict:
                inherited_next[key] = node_dict[key]
        node_type = node_dict.get("Type")
        if node_type == "Pages":
            kids = node_dict.get("Kids") or []
            for kid in kids:
                kid_obj = self._resolve_ref(kid)
                if kid_obj:
                    self._collect_pages(kid_obj, inherited_next)
        elif node_type == "Page":
            page_dict = {
                "object": node,
                "MediaBox": node_dict.get("MediaBox") or inherited_next.get("MediaBox"),
                "CropBox": node_dict.get("CropBox") or inherited_next.get("CropBox"),
                "Resources": node_dict.get("Resources") or inherited_next.get("Resources"),
                "Contents": node_dict.get("Contents"),
            }
            self.pages.append(page_dict)

    def num_pages(self) -> int:
        return len(self.pages)

    def page_size(self, page_idx: int) -> Tuple[float, float]:
        if page_idx < 0 or page_idx >= len(self.pages):
            return (0.0, 0.0)
        page = self.pages[page_idx]
        mediabox = page.get("CropBox") or page.get("MediaBox") or [0, 0, 0, 0]
        mediabox = self._resolve_value(mediabox)
        if isinstance(mediabox, dict):
            mediabox = mediabox.get("MediaBox") or mediabox.get("CropBox") or mediabox
            mediabox = self._resolve_value(mediabox)
        if isinstance(mediabox, tuple):
            mediabox = list(mediabox)
        if isinstance(mediabox, list) and len(mediabox) >= 4:
            x0, y0, x1, y1 = [float(v) for v in mediabox[:4]]
            return (x1 - x0, y1 - y0)
        return (0.0, 0.0)

    def _resolve_value(self, value: Any, depth: int = 0) -> Any:
        if depth > 5:
            return value
        if isinstance(value, tuple) and len(value) == 2 and all(
            isinstance(v, int) for v in value
        ):
            obj = self._resolve_ref(value)
            if not obj:
                return value
            if obj.value is not None:
                return self._resolve_value(obj.value, depth + 1)
            if obj.dictionary is not None:
                return self._resolve_value(obj.dictionary, depth + 1)
            if obj.stream is not None:
                return obj.stream
        if isinstance(value, list):
            return [self._resolve_value(v, depth + 1) for v in value]
        return value

    def _page_contents(self, page_idx: int) -> List[bytes]:
        page = self.pages[page_idx]
        contents = page.get("Contents")
        out: List[bytes] = []
        if isinstance(contents, list):
            refs = contents
        elif contents is None:
            refs = []
        else:
            refs = [contents]
        for ref in refs:
            if isinstance(ref, (tuple, int)):
                obj = self._resolve_ref(ref)
                if obj and obj.stream is not None:
                    out.append(obj.stream)
            elif isinstance(ref, bytes):
                out.append(ref)
        obj = page.get("object")
        if obj and obj.stream is not None:
            out.append(obj.stream)
        return out

    def extract_page_blocks(self, page_idx: int) -> List[Dict[str, Any]]:
        if page_idx < 0 or page_idx >= len(self.pages):
            return []
        streams = self._page_contents(page_idx)
        if not streams:
            return []
        combined = b"\n".join(streams)
        tokens = _tokenize(combined)
        blocks = _interpret_text(tokens)
        return blocks


def _matrix_multiply(m1: List[float], m2: List[float]) -> List[float]:
    return [
        m1[0] * m2[0] + m1[2] * m2[1],
        m1[1] * m2[0] + m1[3] * m2[1],
        m1[0] * m2[2] + m1[2] * m2[3],
        m1[1] * m2[2] + m1[3] * m2[3],
        m1[0] * m2[4] + m1[2] * m2[5] + m1[4],
        m1[1] * m2[4] + m1[3] * m2[5] + m1[5],
    ]


def _translate(tx: float, ty: float) -> List[float]:
    return [1.0, 0.0, 0.0, 1.0, tx, ty]


def _estimate_width(text: str, font_size: float) -> float:
    if not text:
        return 0.0
    width_units = 0.0
    for ch in text:
        if ch.isspace():
            width_units += 0.33
        elif ch in "il.,:;" or ord(ch) < 128 and ch.islower():
            width_units += 0.45
        elif ch in "mwMW":
            width_units += 0.9
        else:
            width_units += 0.6
    return max(font_size * width_units, font_size * 0.4)


class _TextState:
    def __init__(self) -> None:
        self.font_size = 12.0
        self.leading = 0.0
        self.text_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.text_line_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    def begin(self) -> None:
        self.text_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.text_line_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    def set_font(self, size: float) -> None:
        self.font_size = float(size) if size else 12.0
        if not self.leading:
            self.leading = self.font_size * 1.2

    def set_leading(self, value: float) -> None:
        self.leading = float(value)

    def set_matrix(self, a: float, b: float, c: float, d: float, e: float, f: float) -> None:
        self.text_matrix = [a, b, c, d, e, f]
        self.text_line_matrix = [a, b, c, d, e, f]

    def move_text(self, tx: float, ty: float) -> None:
        self.text_line_matrix = _matrix_multiply(self.text_line_matrix, _translate(tx, ty))
        self.text_matrix = self.text_line_matrix.copy()

    def next_line(self) -> None:
        leading = self.leading if self.leading else self.font_size * 1.2
        self.move_text(0.0, -leading)

    def advance(self, dx: float) -> None:
        self.text_matrix = _matrix_multiply(self.text_matrix, _translate(dx, 0.0))

    def position(self) -> Tuple[float, float]:
        return (self.text_matrix[4], self.text_matrix[5])


def _emit_segment(segments: List[Dict[str, Any]], text: str, state: _TextState) -> None:
    filtered = [ch if ord(ch) >= 32 or ch in "\n\r\t" else " " for ch in text]
    cleaned = "".join(filtered).replace("\r", " ").replace("\n", " ").strip()
    if not cleaned:
        return
    x, y = state.position()
    text_width = _estimate_width(cleaned, state.font_size)
    scale_x = math.hypot(state.text_matrix[0], state.text_matrix[1])
    scale_y = math.hypot(state.text_matrix[2], state.text_matrix[3])
    width = text_width * (scale_x if scale_x else 1.0)
    height = state.font_size * (scale_y if scale_y else 1.0) * 1.1
    segment = {
        "text": cleaned,
        "x": x,
        "y": y,
        "font_size": state.font_size,
        "x0": x,
        "y0": y - height,
        "x1": x + width,
        "y1": y + height * 0.1,
    }
    segments.append(segment)
    state.advance(text_width)


def _interpret_text(tokens: List[Any]) -> List[Dict[str, Any]]:
    state = _TextState()
    segments: List[Dict[str, Any]] = []
    stack: List[Any] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if isinstance(tok, str) and tok in _OPERATOR_TOKENS:
            if tok == "BT":
                state.begin()
            elif tok == "ET":
                pass
            elif tok == "Tf" and len(stack) >= 2:
                size = stack.pop()
                font = stack.pop()  # noqa: F841 - font name unused in fallback
                if isinstance(size, (int, float)):
                    state.set_font(float(size))
            elif tok == "TL" and stack:
                val = stack.pop()
                if isinstance(val, (int, float)):
                    state.set_leading(float(val))
            elif tok == "Td" and len(stack) >= 2:
                ty = stack.pop()
                tx = stack.pop()
                if isinstance(tx, (int, float)) and isinstance(ty, (int, float)):
                    state.move_text(float(tx), float(ty))
            elif tok == "TD" and len(stack) >= 2:
                ty = stack.pop()
                tx = stack.pop()
                if isinstance(tx, (int, float)) and isinstance(ty, (int, float)):
                    state.set_leading(-float(ty))
                    state.move_text(float(tx), float(ty))
            elif tok == "Tm" and len(stack) >= 6:
                f = [stack.pop() for _ in range(6)][::-1]
                if all(isinstance(v, (int, float)) for v in f):
                    state.set_matrix(*[float(v) for v in f])
            elif tok == "T*":
                state.next_line()
            elif tok == "'" and stack:
                val = stack.pop()
                if isinstance(val, _PDFString):
                    state.next_line()
                    _emit_segment(segments, str(val), state)
            elif tok == '"' and len(stack) >= 3:
                string_val = stack.pop()
                if isinstance(string_val, _PDFString):
                    stack.pop()  # char spacing - ignored
                    stack.pop()  # word spacing - ignored
                    state.next_line()
                    _emit_segment(segments, str(string_val), state)
            elif tok == "Tj" and stack:
                val = stack.pop()
                if isinstance(val, _PDFString):
                    _emit_segment(segments, str(val), state)
            elif tok == "TJ" and stack:
                arr = stack.pop()
                if isinstance(arr, list):
                    text_parts: List[str] = []
                    total_advance = 0.0
                    for item in arr:
                        if isinstance(item, _PDFString):
                            text_parts.append(str(item))
                            total_advance += _estimate_width(str(item), state.font_size)
                        elif isinstance(item, (int, float)):
                            total_advance += -float(item) / 1000.0 * state.font_size
                    text = "".join(text_parts)
                    _emit_segment(segments, text, state)
                    if total_advance:
                        state.advance(max(total_advance, 0.0))
            elif tok in {"Tw", "Tc"} and stack:
                stack.pop()  # spacing operators ignored
            stack.clear()
        else:
            if tok == "[":
                stack.append(tok)
            elif tok == "]":
                items: List[Any] = []
                while stack:
                    itm = stack.pop()
                    if itm == "[":
                        break
                    items.append(itm)
                stack.append(list(reversed(items)))
            else:
                stack.append(tok)
        i += 1
    return _group_segments(segments)


def _group_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = [s for s in segments if s.get("text")]
    if not filtered:
        return []
    filtered.sort(key=lambda s: (-s["y"], s["x"]))
    lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_y: Optional[float] = None
    for seg in filtered:
        if not current:
            current = [seg]
            current_y = seg["y"]
            continue
        assert current_y is not None
        if abs(seg["y"] - current_y) <= max(seg["font_size"], current[0]["font_size"]) * 0.6:
            current.append(seg)
            current_y = (current_y * (len(current) - 1) + seg["y"]) / len(current)
        else:
            lines.append(current)
            current = [seg]
            current_y = seg["y"]
    if current:
        lines.append(current)
    blocks: List[Dict[str, Any]] = []
    for line in lines:
        line.sort(key=lambda s: s["x"])
        text_parts: List[str] = []
        prev = None
        for seg in line:
            if prev is not None:
                gap = seg["x"] - prev["x1"]
                if gap > max(prev["font_size"], seg["font_size"]) * 0.3:
                    text_parts.append(" ")
            text_parts.append(seg["text"])
            prev = seg
        x0 = min(s["x0"] for s in line)
        y0 = min(s["y0"] for s in line)
        x1 = max(s["x1"] for s in line)
        y1 = max(s["y1"] for s in line)
        blocks.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": "".join(text_parts)})
    return blocks


_simple_cache: Dict[Path, _SimplePDFDocument] = {}


def _get_simple_doc(pdf_path: Path) -> _SimplePDFDocument:
    doc = _simple_cache.get(pdf_path)
    if doc is None:
        doc = _SimplePDFDocument(pdf_path)
        _simple_cache[pdf_path] = doc
    return doc


def num_pages(pdf_path: Path) -> int:
    if have_fitz:
        with fitz.open(str(pdf_path)) as doc:  # type: ignore[attr-defined]
            return doc.page_count
    return _get_simple_doc(pdf_path).num_pages()


def get_page_pixmap(pdf_path: Path, page_idx: int, clip: Optional[Tuple[float, float, float, float]] = None,
                    dpi: int = 300):
    if have_fitz:
        doc = fitz.open(str(pdf_path))  # type: ignore[attr-defined]
        page = doc.load_page(page_idx)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        if clip:
            rect = fitz.Rect(*clip)
            pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
        else:
            pix = page.get_pixmap(matrix=mat, alpha=False)
        doc.close()
        return pix
    return None


def extract_page_text_blocks(pdf_path: Path, page_idx: int) -> List[Dict[str, Any]]:
    if have_fitz:
        blocks_out: List[Dict[str, Any]] = []
        with fitz.open(str(pdf_path)) as doc:  # type: ignore[attr-defined]
            page = doc.load_page(page_idx)
            for i, b in enumerate(page.get_text("blocks")):
                x0, y0, x1, y1, text, *_ = b
                blocks_out.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text or "", "block_no": i})
        return blocks_out
    return _get_simple_doc(pdf_path).extract_page_blocks(page_idx)


def page_size(pdf_path: Path, page_idx: int) -> Tuple[float, float]:
    if have_fitz:
        with fitz.open(str(pdf_path)) as doc:  # type: ignore[attr-defined]
            page = doc.load_page(page_idx)
            rect = page.rect
            return (rect.width, rect.height)
    return _get_simple_doc(pdf_path).page_size(page_idx)
