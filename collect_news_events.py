"""Collect high-confidence Nigerian flood occurrence labels from GDELT.

The output is weakly-supervised evidence for the urban flash-flood model. A news
article is never treated as a perfect sensor: we only keep headlines that both
name a Nigerian location and describe flooding as already occurring.
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "urban_flood_events.csv")

LOCATIONS = [
    ("FCT", "Abuja", 9.0765, 7.3986, ["abuja", "fct", "maitama", "asokoro", "garki", "wuse", "lugbe", "kubwa", "jabi", "gwarinpa", "apo", "guzape", "nyanya", "kuje", "gwagwalada", "bwari"]),
    ("Abia", "Umuahia", 5.5249, 7.4946, ["abia", "umuahia", "aba"]),
    ("Adamawa", "Yola", 9.2035, 12.4954, ["adamawa", "yola", "mubi"]),
    ("Akwa Ibom", "Uyo", 5.0389, 7.9098, ["akwa ibom", "uyo", "eket"]),
    ("Anambra", "Awka", 6.2101, 7.0741, ["anambra", "awka", "onitsha", "nnewi"]),
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
    ("Niger", "Minna", 9.6139, 6.5569, ["niger state", "minna", "suleja", "bida"]),
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

OCCURRED = re.compile(r"\b(flooded|flooding|floods|flash flood|inundated|submerged|swept|sweeps|washed away|overflowed)\b", re.I)
WARNING_ONLY = re.compile(r"\b(warns?|warning|forecast|expected|may flood|risk of flooding|alert issued)\b", re.I)


def parse_gdelt_time(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def locate(text: str):
    lower = text.lower()
    matches = []
    for state, city, lat, lon, aliases in LOCATIONS:
        hit = next((alias for alias in aliases if alias in lower), None)
        if hit:
            matches.append((len(hit), state, city, lat, lon, hit))
    if not matches:
        return None
    matches.sort(reverse=True)
    _, state, city, lat, lon, hit = matches[0]
    return state, city, lat, lon, hit


def key(row: dict[str, str]) -> str:
    title = re.sub(r"\W+", " ", row["title"].lower()).strip()
    return f"{row['event_time'][:10]}|{row['state']}|{title}"


def load_existing() -> dict[str, dict[str, str]]:
    if not os.path.exists(OUT_PATH):
        return {}
    with open(OUT_PATH, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {key(row): row for row in rows}


def fetch_articles():
    query = '(flood OR flooding OR "flash flood" OR inundation) Nigeria'
    response = requests.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={"query": query, "mode": "ArtList", "maxrecords": 250, "format": "json", "timespan": "3months", "sort": "DateDesc"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("articles", [])


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    existing = load_existing()
    added = 0

    for article in fetch_articles():
        title = str(article.get("title") or "").strip()
        if not title or not OCCURRED.search(title):
            continue
        # A headline that only forecasts risk should not become a positive label.
        if WARNING_ONLY.search(title) and not re.search(r"\b(flooded|flooding|floods|submerged|inundated|swept|sweeps)\b", title, re.I):
            continue
        location = locate(title)
        if not location:
            continue
        event_time = parse_gdelt_time(str(article.get("seendate") or ""))
        if not event_time:
            continue
        state, city, lat, lon, matched = location
        source_url = str(article.get("url") or "")
        row = {
            "event_time": event_time.isoformat(),
            "state": state,
            "location": city,
            "latitude": f"{lat:.5f}",
            "longitude": f"{lon:.5f}",
            "title": title,
            "source_url": source_url,
            "source_domain": urlparse(source_url).netloc.replace("www.", ""),
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

    print(f"urban flood event store: {len(rows)} rows ({added} new)")


if __name__ == "__main__":
    main()
