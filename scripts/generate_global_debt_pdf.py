import textwrap
from typing import List

TITLE_LINES = [
    "3Q 2025 BAC Global Debt Threat Scenario",
    "BAC-Global Debt Threat Scenario (Summary)",
    "",
    "- After a few quarters of expansion above baseline, scenario volatility resurfaces. Real US GDP (\"RGDP\") growth declines quarter-over-quarter (\"QoQ\") by approximately 4.0%. RGDP briefly exceeds the baseline level after the start of the horizon before falling below baseline later in the scenario. The unemployment rate (\"UR\") begins to weaken after staying below baseline, peaks at 7.8% before retracing, and remains below baseline levels thereafter.",
    "- The Federal Reserve begins reducing rates early in the scenario and keeps rates low until recovery gets underway. The Fed cuts 100 bps through 1Q 2026, when the average fed effective rate reaches 0.9%. Rates remain on hold until 2Q 2026.",
    "- Short-term rates decline from the start of the horizon, with the U.S. 3-month Treasury yield (\"US3\") reaching a 0.9% trough by 3Q 2026.",
    "- US Treasury 10Y (\"UST 10Y\") rates rise from the start of the horizon, moving from 4.4% in 2Q 2025 to 4.7% in 4Q 2025 before resuming their decline by 3Q 2026 and ending the horizon around baseline levels.",
    "- 10-year swap (\"USSW10\") rates remain near baseline, dipping 10 bps below baseline in 3Q 2026 before converging back with baseline by the end of the horizon.",
    "- Home (\"HP\") and Commercial (\"CREPI\") real estate price indexes fall below baseline by 1.6% and 2.6%, respectively, before recovering and finishing marginally under baseline levels.",
    "- The Corporate Composite Index (\"Corp CI\") troughs at 88 before recovering modestly and ending slightly below baseline.",
    "- The U.S. dollar weakens by roughly 12.5% by 4Q 2026 before stabilizing modestly higher by the end of the horizon.",
    "",
    "BAC-Global Debt Threat Scenario (Narrative)",
    "",
    "Introduction",
    "The Bank of America Corporation (\"BAC\") Global Debt Threat (BAG-GDT) scenario intends to test BAC against a backdrop of a more tightly governed monetary policy stance that tolerates higher inflation, accompanied by downgrades in U.S. debt, weaker U.S. dollar, slower international growth, and a sizable contraction in domestic housing starts.",
    "",
    "Economic growth and labor market",
    "The scenario is characterized by a recession starting after a brief early period in which the economy remains slightly above baseline. U.S. RGDP declines and remains near baseline until 2026, accompanied by increases in the unemployment rate that begin after the economy loses momentum. The UR peaks at 7.8% before gradually declining. By the end of the horizon, RGDP falls a cumulative 3.5% before a moderate recovery takes hold.",
    "",
    "Inflation, interest rates and central bank policies",
    "A significant turn for Federal Reserve policy, including subdued U.S. Treasury bond operations and regional program sales, sustains rates and enables the scenario's moderate bias to higher rates. Three-month Treasury yields trend toward baseline before falling sharply in the first half of the horizon, whereas longer-term rates and the federal funds rate fall to a 0.75% to 1% range by the end of 2026. The scenario includes a reduction of the Federal Reserve balance sheet from $9.0 trillion to $6.5 trillion. The Fed reintroduces asset purchases in 2Q 2026 in response to the weakening economy.",
    "The approach fuels a cycle of rising inflation expectations and a pronounced steepening of the yield curve. Concerns over the sustainability of U.S. deficits and debt levels leads to a two-notch downgrade of the country's long-term credit rating to AA+, although the short-term rating remains at P-1. The Fed pauses balance sheet reduction in late 2026 before gradually expanding its balance sheet by roughly $350 billion from 2Q 2026 to 2Q 2027. Asset purchases are matched by allocating 50% to Treasuries, 10% to currency, 0% to repo, and 40% to other liabilities, including the Treasury General Account.",
    "",
    "Household and business credit",
    "As financing costs surge, aggregate demand contracts, prompting firms to delay investment and households to reduce consumption. The benchmark S&P 500 sees declines culminating in 4Q 2025 before recovering gradually. Corporate profits deteriorate as investors, alarmed by the perceived instability in U.S. commitments, money markets, real estate quality, and sustained weakness of the dollar, step back from risk. These developments reduce capital formation, weigh heavily on the supply of labor, and diminish household formation. The scenario assumes that the federal government responds with a downgrade of the country's long-term debt rating to AA+, although the short-term rating remains at P-1.",
    "Housing investment is hit hardest, with real residential investment sliding roughly 1.5% below baseline by 4Q 2025. Nonresidential structures drop about 3% below baseline by late 2025, with a slow recovery beginning in 2026. Equipment spending falls more modestly, with levels 1.7% below baseline by 2Q 2026.",
    "",
    "Financial markets and asset prices",
    "Equity markets face sustained declines, with the S&P 500 settling 18% below baseline in 4Q 2025 before partially recovering by 3Q 2026. Volatility remains elevated as investors reassess corporate earnings and the global growth outlook. Corporate bond spreads widen substantially, pushing the Corporate Composite Index down to 88 at the trough before recovering toward 92 by the end of the horizon. Moody's Aaa and Baa 30-year corporate spreads peak at 270 bps and 399 bps, respectively, before gradually narrowing.",
    "Commodity prices are mixed: West Texas Intermediate (\"WTI\") crude oil drops from $97 per barrel in 2Q 2025 to $67 per barrel by early 2026 before gradually improving to $80 per barrel by 3Q 2026.",
    "Home price and commercial real estate indexes fall below baseline by 1.6% and 2.6%, respectively, before improving marginally. HPI and CREPI show a cumulative decline in the final quarters of the projection horizon, and CRE performance remains under pressure as higher vacancies and refinancing challenges persist through 2026.",
    "",
    "International developments",
    "Internationally, economies experience similar slowdowns. Outside of the United States, inflation and currencies remain under pressure with long-run spreads widening. Term-premium shocks and sudden-stop dynamics spur additional volatility across Asian sovereigns. Capital outflows from the U.S. push investors toward emerging market economies seeking higher returns; this produces a synchronized global slowdown as emerging regions absorb those adjustments. Several advanced economies face downgrades of sovereign credit ratings to AA, while others hold stable outlooks on substantial fiscal reforms.",
    "European GDP growth drops to an annualized 0.7% in 4Q 2026 (dropping cumulatively by 0.8%). U.K. real GDP drops 2.0%, reaching a maximum cumulative decline of 5.2% by 1Q 2027; for the Eurozone, the maximum real GDP cumulative drop is 3.9% by 4Q 2027. Japan also suffers a recession, dropping a cumulative 6.1% by 3Q 2028.",
    "",
    "International developments (continued)",
    "Chinese real GDP growth slips to 4.0% in 4Q 2026 (dropping cumulatively by 5.7%), before gradually recovering. Emerging markets with a heavy reliance on external financing face elevated funding costs, amplifying currency volatility. Latin American economies see currencies depreciate by between 6% and 12% over the horizon, with the exception of the Mexican peso. Emerging Asia experiences similar depreciation pressures, particularly across the Indonesian rupiah and the Philippine peso.",
    "As the value of the U.S. dollar has declined relative to other major currencies, it weakens against other currencies at the start of the second half of the horizon. By contrast, the Euro and the U.K. pound appreciate within a narrow range. Developed Asia and Latin America experience notable depreciations. Developing Asia currencies decline by 8.6% and against the Japanese yen by 15.7%. All currencies even out before levelling higher by the end of the horizon.",
    "",
    "Additional U.S. Economic Narrative - Ops Loss",
    "The scenario includes a narrative for operational risk losses linked to strained supply chains, increased geopolitical tension, and cyber events that create added operational turmoil and business continuity disruptions not included in narratives.",
]


def wrap_lines(paragraphs: List[str], width: int = 92) -> List[str]:
    wrapped: List[str] = []
    for paragraph in paragraphs:
        if paragraph == "":
            wrapped.append("")
            continue
        for line in textwrap.wrap(paragraph, width=width):
            wrapped.append(line)
    return wrapped


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_page_stream(lines: List[str]) -> bytes:
    parts = ["BT", "/F1 12 Tf", "14 TL", "72 760 Td"]
    for line in lines:
        if not line:
            parts.append("T*")
            continue
        parts.append(f"({escape_pdf_text(line)}) Tj")
        parts.append("T*")
    parts.append("ET")
    stream = "\n".join(parts).encode("utf-8")
    return stream


def build_pdf(lines: List[str], lines_per_page: int = 45) -> bytes:
    pages = [lines[i:i + lines_per_page] for i in range(0, len(lines), lines_per_page)]
    objects: List[bytes] = []

    def add_object(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    page_objects = []
    content_objects = []
    for page_lines in pages:
        stream = make_page_stream(page_lines)
        content_stream = b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        content_obj = add_object(content_stream)
        content_objects.append(content_obj)
        page_obj = add_object(
            b"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 0 0 R >> >> /Contents "
            + f"{content_obj} 0 R".encode("ascii") + b" >>"
        )
        page_objects.append(page_obj)

    pages_kids = " ".join(f"{num} 0 R" for num in page_objects)
    pages_obj = add_object(f"<< /Type /Pages /Kids [ {pages_kids} ] /Count {len(page_objects)} >>".encode("ascii"))

    catalog_obj = add_object(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode("ascii"))

    font_obj = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    # Fix up parent and font references
    fixed_objects: List[bytes] = []
    for index, obj in enumerate(objects, start=1):
        if index in page_objects:
            parent_ref = f"/Parent {pages_obj} 0 R"
            font_ref = f"/F1 {font_obj} 0 R"
            replaced = obj.replace(b"/Parent 0 0 R", parent_ref.encode("ascii"))
            replaced = replaced.replace(b"/F1 0 0 R", font_ref.encode("ascii"))
            fixed_objects.append(replaced)
        else:
            fixed_objects.append(obj)

    objects = fixed_objects

    xref_positions = []
    buffer = bytearray(b"%PDF-1.4\n")
    for number, obj in enumerate(objects, start=1):
        xref_positions.append(len(buffer))
        buffer.extend(f"{number} 0 obj\n".encode("ascii"))
        buffer.extend(obj)
        buffer.extend(b"\nendobj\n")

    xref_start = len(buffer)
    buffer.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    buffer.extend(b"0000000000 65535 f \n")
    for pos in xref_positions:
        buffer.extend(f"{pos:010d} 00000 n \n".encode("ascii"))

    buffer.extend(b"trailer\n")
    buffer.extend(f"<< /Size {len(objects) + 1} /Root {catalog_obj} 0 R >>\n".encode("ascii"))
    buffer.extend(b"startxref\n")
    buffer.extend(f"{xref_start}\n".encode("ascii"))
    buffer.extend(b"%%EOF\n")

    return bytes(buffer)


if __name__ == "__main__":
    wrapped_lines = wrap_lines(TITLE_LINES)
    pdf_bytes = build_pdf(wrapped_lines)
    output_path = "data/BAC_Global_Debt_Threat_Scenario.pdf"
    with open(output_path, "wb") as pdf_file:
        pdf_file.write(pdf_bytes)
    print(f"Wrote {output_path} with {len(wrapped_lines)} lines across {len(wrapped_lines) // 45 + 1} page(s).")
