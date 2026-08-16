"""Collect high-confidence Nigerian flood occurrence labels from live news discovery.

The collector has two modes:
1. Daily live discovery from national and major-outlet searches.
2. Automatic historical backfill while the evidence store is still small. It
   queries every state/FCT so the urban model does not learn only Abuja/Lagos.

News remains weak supervision: only headlines that describe flooding as already
occurring and name a Nigerian location become positive training evidence.
"""

from __future__ import annotations

import csv
import html
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urlparse

import requests

OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "urban_flood_events.csv")
BACKFILL_TARGET = 120

LOCATIONS = [
    ("FCT", "Abuja", 9.0765, 7.3986, ["abuja", "fct", "maitama", "asokoro", "garki", "wuse", "wuse 2", "gudu", "lokogoma", "gaduwa", "lugbe", "kubwa", "jabi", "gwarinpa", "apo", "guzape", "nyanya", "kuje", "gwagwalada", "bwari"]),
    ("Abia", "Umuahia", 5.5249, 7.4946, ["abia", "umuahia", "aba"]),
    ("Adamawa", "Yola", 9.2035, 12.4954, ["adamawa", "yola", "mubi"]),
    ("Akwa Ibom", "Uyo", 5.0389, 7.9098, ["akwa ibom", "uyo", "eket"]),
    ("Anambra", "Awka", 6.2101, 7.0741, ["anambra", "awka", "onitsha", "nnewi", "ogidi"]),
    ("Bauchi", "Bauchi", 10.3158, 9.8442, ["bauchi"]),
    ("Bayelsa", "Yenagoa", 4.9267, 6.2676, ["bayelsa", "yenagoa"]),
    ("Benue", "Makurdi", 7.7337, 8.5214, ["benue", "makurdi"]),
    ("Borno", "Maiduguri", 11.8469, 13.1571, ["borno", "maiduguri"]),
    ("Cross River", "Calabar", 4.9517, 8.3220, ["cross river", "calabar"]),
    ("Delta", "Asaba", 6.1985, 6.7274, ["delta state", "asaba", "warri", "ughelli"]),
    ("Ebonyi", "Abakaliki", 6.3249, 8.1137, ["ebonyi", "abakaliki"]),
    ("Edo", "Benin City", 6.3350, 5.6037, ["edo state", "benin city"]),
    ("Ekiti", "Ado-Ekiti", 7.6233, 5.2209, ["ekiti", "ado ekiti", "ado-ekiti"]),
    ("Enugu", "Enugu", 6.5244, 7.5100, ["enugu", "nsukka"]),
    ("Gombe", "Gombe", 10.2897, 11.1673, ["gombe"]),
    ("Imo", "Owerri", 5.4891, 7.0176, ["imo state", "owerri"]),
    ("Jigawa", "Dutse", 11.7594, 9.3392, ["jigawa", "dutse"]),
    ("Kaduna", "Kaduna", 10.5105, 7.4165, ["kaduna", "zaria", "kafanchan"]),
    ("Kano", "Kano", 12.0022, 8.5920, ["kano"]),
    ("Katsina", "Katsina", 12.9908, 7.6018, ["katsina"]),
    ("Kebbi", "Birnin Kebbi", 12.4539, 4.1975, ["kebbi", "birnin kebbi"]),
    ("Kogi", "Lokoja", 7.8023, 6.7333, ["kogi", "lokoja", "okene"]),
    ("Kwara", "Ilorin", 8.4966, 4.5421, ["kwara", "ilorin"]),
    ("Lagos", "Ikeja", 6.6018, 3.3515, ["lagos", "ikeja", "lekki", "victoria island", "ikorodu", "epe", "ajah"]),
    ("Nasarawa", "Lafia", 8.4966, 8.5153, ["nasarawa", "lafia", "keffi", "mararaba"]),
    ("Niger", "Minna", 9.6139, 6.5569, ["niger state", "minna", "suleja", "bida", "shiroro"]),
    ("Ogun", "Abeokuta", 7.1475, 3.3619, ["ogun", "abeokuta", "ijebu ode", "ota", "sagamu"]),
    ("Ondo", "Akure", 7.2571, 5.2058, ["ondo state", "akure"]),
    ("Osun", "Osogbo", 7.7827, 4.5418, ["osun", "osogbo", "ile-ife", "ile ife", "ilesa"]),
    ("Oyo", "Ibadan", 7.3775, 3.9470, ["oyo state", "ibadan", "ogbomoso"]),
    ("Plateau", "Jos", 9.8965, 8.8583, ["plateau", "jos"]),
    ("Rivers", "Port Harcourt", 4.8156, 7.0498, ["rivers state", "port harcourt"]),
    ("Sokoto", "Sokoto", 13.0059, 5.2476, ["sokoto"]),
    ("Taraba", "Jalingo", 8.8920, 11.3771, ["taraba", "jalingo"]),
    ("Yobe", "Damaturu", 11.7470, 11.9608, ["yobe", "damaturu", "potiskum"]),
    ("Zamfara", "Gusau", 12.1704, 6.6641, ["zamfara", "gusau"]),
]

OCCURRED = re.compile(r"\b(flooded|flooding|floods|flash flood|inundated|submerged|submerges|swept|sweeps|washed away|overflowed)\b", re.I)
WARNING_ONLY = re.compile(r"\b(warns?|warning|forecast|expected|may flood|risk of flooding|alert issued)\b", re.I)
OCCURRENCE_OVERRIDE = re.compile(r"\b(flooded|flooding|floods|submerged|submerges|inundated|swept|sweeps)\b", re.I)


def alias_match(text: str, alias: str) -> bool:
    pattern = r"(^|[^a-z0-9])" + re.escape(alias.lower()).replace(r"\ ", r"[-\s]+").replace(r"\-", r"[-\s]+") + r"([^a-z0-9]|$)"
    return bool(re.search(pattern, text.lower()))


def locate(text: str):
    matches = []
    for state, city, lat, lon, aliases in LOCATIONS:
        for alias in aliases:
            if alias_match(text, alias):
                matches.append((len(alias), state, city, lat, lon, alias))
    if not matches:
        return None
    matches.sort(reverse=True)
    _, state, city, lat, lon, hit = matches[0]
    return state, city, lat, lon, hit


def parse_gdelt_time(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def clean_title(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def key(row: dict[str, str]) -> str:
    title = re.sub(r"\W+", " ", row["title"].lower()).strip()
    return f"{row['event_time'][:10]}|{row['state']}|{title}"


def load_existing() -> dict[str, dict[str, str]]:
    if not os.path.exists(OUT_PATH):
        return {}
    with open(OUT_PATH, newline="", encoding="utf-8") as handle:
        return {key(row): row for row in csv.DictReader(handle)}


def fetch_gdelt_articles() -> list[dict]:
    try:
        response = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={"query": '(flood OR flooding OR "flash flood" OR inundation) Nigeria', "mode": "ArtList", "maxrecords": 250, "format": "json", "timespan": "3months", "sort": "DateDesc"},
            timeout=30,
        )
        response.raise_for_status()
        return [
            {"title": item.get("title", ""), "url": item.get("url", ""), "published": parse_gdelt_time(str(item.get("seendate") or "")), "source": str(item.get("domain") or "GDELT")}
            for item in response.json().get("articles", [])
        ]
    except Exception as exc:
        print(f"GDELT unavailable: {exc}")
        return []


def fetch_google_news(query: str) -> list[dict]:
    try:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-NG&gl=NG&ceid=NG:en"
        response = requests.get(url, timeout=30, headers={"User-Agent": "NaijaClimaGuard flood evidence collector"})
        response.raise_for_status()
        root = ET.fromstring(response.text)
        rows = []
        for item in root.findall("./channel/item"):
            title = clean_title(item.findtext("title") or "")
            link = (item.findtext("link") or "").strip()
            source_node = item.find("source")
            source = clean_title(source_node.text or "Google News") if source_node is not None else "Google News"
            try:
                published = parsedate_to_datetime(item.findtext("pubDate") or "").astimezone(timezone.utc)
            except Exception:
                published = None
            rows.append({"title": title, "url": link, "published": published, "source": source})
        return rows
    except Exception as exc:
        print(f"Google News query failed ({query}): {exc}")
        return []


def fetch_live_articles() -> list[dict]:
    rows = fetch_gdelt_articles()
    queries = [
        '(flood OR flooding OR "flash flood") Nigeria when:7d',
        'site:vanguardngr.com (flood OR flooding) Nigeria when:7d',
        'site:guardian.ng (flood OR flooding) Nigeria when:7d',
        'site:dailytrust.com (flood OR flooding) Nigeria when:7d',
        'site:thecable.ng (flood OR flooding) Nigeria when:7d',
    ]
    for query in queries:
        rows.extend(fetch_google_news(query))
    return rows


def fetch_historical_backfill(existing_rows: int) -> list[dict]:
    if existing_rows >= BACKFILL_TARGET:
        return []
    print(f"Historical backfill active: {existing_rows}/{BACKFILL_TARGET} evidence rows")
    rows: list[dict] = []
    # One state-specific query each substantially reduces Lagos/Abuja bias without
    # hammering any individual publisher. The collector stops historical expansion
    # once the evidence store is large enough for shadow research.
    for state, city, _lat, _lon, _aliases in LOCATIONS:
        query_place = "Abuja FCT" if state == "FCT" else f'"{state}" Nigeria'
        queries = [
            f'(flood OR flooding OR "flash flood") {query_place} when:365d',
            f'(flood OR flooding) "{city}" Nigeria when:365d',
        ]
        for query in queries:
            rows.extend(fetch_google_news(query))
            time.sleep(0.18)
    return rows


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    existing = load_existing()
    added = 0
    discovered = fetch_live_articles()
    discovered.extend(fetch_historical_backfill(len(existing)))

    for article in discovered:
        title = clean_title(str(article.get("title") or ""))
        if not title or not OCCURRED.search(title):
            continue
        if WARNING_ONLY.search(title) and not OCCURRENCE_OVERRIDE.search(title):
            continue
        location = locate(title)
        if not location:
            continue
        event_time = article.get("published")
        if not isinstance(event_time, datetime):
            continue
        state, city, lat, lon, matched = location
        source_url = str(article.get("url") or "")
        source_domain = str(article.get("source") or "").strip() or urlparse(source_url).netloc.replace("www.", "")
        row = {
            "event_time": event_time.astimezone(timezone.utc).isoformat(),
            "state": state,
            "location": city,
            "latitude": f"{lat:.5f}",
            "longitude": f"{lon:.5f}",
            "title": title,
            "source_url": source_url,
            "source_domain": source_domain,
            "matched_alias": matched,
            "label": "1",
            "confidence": "news_reported_occurrence",
        }
        k = key(row)
        if k not in existing:
            existing[k] = row
            added += 1

    fieldnames = ["event_time", "state", "location", "latitude", "longitude", "title", "source_url", "source_domain", "matched_alias", "label", "confidence"]
    rows = sorted(existing.values(), key=lambda row: row["event_time"])
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    state_count = len({row["state"] for row in rows})
    print(f"urban flood event store: {len(rows)} rows across {state_count} states/FCT ({added} new from {len(discovered)} discovered articles)")


if __name__ == "__main__":
    main()
