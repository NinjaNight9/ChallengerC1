#!/usr/bin/env python
"""Backfill public ATP Challenger moneyline odds from TennisExplorer result pages.

The daily historical results page exposes one row-pair per match and the H/A
moneyline prices shown for that completed match. We retain only men's Challenger
tournaments. A market-sanity flag is stored because archived web data can
occasionally contain malformed price pairs; invalid pairs are never allowed into
C1 value-model training.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

from bs4 import BeautifulSoup

BASE = "https://www.tennisexplorer.com/results/?type=atp-single&year={y}&month={m:02d}&day={d:02d}"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36"


def daterange(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def fetch(url: str, tries: int = 5) -> str:
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as exc:
            last = exc
            if attempt + 1 < tries:
                time.sleep(min(30.0, 2.0 ** attempt + random.random()))
    raise RuntimeError(f"failed {url}: {last!r}")


def _float(text: str):
    text = text.strip().replace(",", ".")
    try:
        x = float(text)
        return x if x > 1.0 else None
    except Exception:
        return None


def price_quality(o1, o2):
    if o1 is None or o2 is None:
        return None, False
    s = 1.0 / o1 + 1.0 / o2
    # Deliberately broad. Normal two-way tennis books usually sit just above 1;
    # this only rejects clearly broken archive rows rather than optimizing a
    # profitability filter after seeing results.
    return s, bool(0.94 <= s <= 1.20)


def parse_day(html: str, day: date) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict] = []
    for table in soup.find_all("table", class_="result"):
        current_tournament = ""
        rows = table.find_all("tr")
        i = 0
        while i < len(rows):
            row = rows[i]
            classes = row.get("class", [])
            if "head" in classes and "flags" in classes:
                td = row.find("td", class_="t-name")
                current_tournament = td.get_text(" ", strip=True) if td else ""
                i += 1
                continue

            rid = row.get("id", "")
            if rid and not rid.endswith("b") and ("one" in classes or "two" in classes):
                if "challenger" not in current_tournament.casefold():
                    i += 1
                    continue
                p1td = row.find("td", class_="t-name")
                p1 = p1td.get_text(" ", strip=True) if p1td else ""
                p2 = ""
                if i + 1 < len(rows):
                    r2 = rows[i + 1]
                    if r2.get("id") == rid + "b":
                        p2td = r2.find("td", class_="t-name")
                        p2 = p2td.get_text(" ", strip=True) if p2td else ""
                p1 = re.sub(r"\s*\(\d+\)\s*$", "", p1).strip()
                p2 = re.sub(r"\s*\(\d+\)\s*$", "", p2).strip()

                vals = []
                for td in row.find_all("td", class_=["course", "coursew"]):
                    v = _float(td.get_text(" ", strip=True))
                    if v is not None:
                        vals.append(v)
                odd1 = vals[0] if len(vals) >= 1 else None
                odd2 = vals[1] if len(vals) >= 2 else None
                implied_sum, valid_price_pair = price_quality(odd1, odd2)

                ttd = row.find("td", class_=lambda c: c and "time" in c.split())
                tm = ""
                if ttd:
                    m = re.search(r"\b\d{2}:\d{2}\b", ttd.get_text(" ", strip=True))
                    tm = m.group(0) if m else ""
                info = row.select_one("td:last-child a[href*='match-detail']") or row.select_one("a[href*='match-detail']")
                match_url = "https://www.tennisexplorer.com" + info.get("href") if info and info.get("href") else ""

                if p1 and p2:
                    out.append({
                        "date": day.isoformat(), "time": tm,
                        "tournament": current_tournament,
                        "player1": p1, "player2": p2,
                        "odd1": odd1, "odd2": odd2,
                        "implied_sum": implied_sum,
                        "valid_price_pair": int(valid_price_pair),
                        "match_url": match_url,
                        "source": "TennisExplorer",
                        "price_type": "historical_displayed_average",
                    })
                if i + 1 < len(rows) and rows[i + 1].get("id") == rid + "b":
                    i += 1
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delay-min", type=float, default=0.65)
    ap.add_argument("--delay-max", type=float, default=1.15)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["date","time","tournament","player1","player2","odd1","odd2","implied_sum","valid_price_pair","match_url","source","price_type"]
    existing_dates = set()
    if args.resume and path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            existing_dates = {r["date"] for r in csv.DictReader(f) if r.get("date")}
    mode = "a" if args.resume and path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == "w": w.writeheader()
        total = valid = 0
        for idx, d in enumerate(daterange(start, end), 1):
            if d.isoformat() in existing_dates:
                continue
            url = BASE.format(y=d.year, m=d.month, d=d.day)
            try:
                rows = parse_day(fetch(url), d)
            except Exception as exc:
                print(f"WARN {d}: {exc}", flush=True)
                time.sleep(5)
                continue
            for r in rows:
                w.writerow(r)
                valid += int(r["valid_price_pair"])
            f.flush(); total += len(rows)
            if idx % 20 == 0 or rows:
                print(f"{d}: {len(rows)} Challenger matches, total={total}, valid_prices={valid}", flush=True)
            time.sleep(random.uniform(args.delay_min, args.delay_max))
    print(f"DONE {start}..{end}: total={total}, valid_prices={valid}, wrote {path}")


if __name__ == "__main__":
    main()
