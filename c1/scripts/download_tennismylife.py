#!/usr/bin/env python
"""Download TennisMyLife ATP/Challenger/qualifying CSVs when run in a networked environment."""
from __future__ import annotations
import argparse, json, urllib.request
from pathlib import Path

API = "https://stats.tennismylife.org/api/data-files"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/tennismylife")
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    args = ap.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    payload = json.loads(urllib.request.urlopen(API, timeout=30).read())
    wanted = set()
    for y in args.years:
        wanted |= {f"{y}.csv", f"{y}_challenger.csv", f"{y}_atp_quali.csv"}
    for item in payload.get("files", []):
        name = item.get("name", "")
        if name not in wanted:
            continue
        target = root / ("atp_quali" if "atp_quali" in name else "") / name
        target.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading", name)
        urllib.request.urlretrieve(item["url"], target)

if __name__ == "__main__":
    main()
