#!/usr/bin/env python
"""Download TennisMyLife ATP/Challenger/qualifying CSVs when run in a networked environment."""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path, PurePosixPath

API = "https://stats.tennismylife.org/api/data-files"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/tennismylife")
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(API, headers={"User-Agent": "ChallengerC1/1.0"})
    payload = json.loads(urllib.request.urlopen(req, timeout=30).read())

    wanted = set()
    for y in args.years:
        wanted |= {
            f"{y}.csv",
            f"{y}_challenger.csv",
            f"atp_quali/{y}_atp_quali.csv",
        }

    downloaded = []
    for item in payload.get("files", []):
        raw_name = str(item.get("name", "")).replace("\\", "/").lstrip("/")
        if raw_name not in wanted:
            continue
        safe_parts = [p for p in PurePosixPath(raw_name).parts if p not in ("", ".", "..")]
        target = root.joinpath(*safe_parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading", raw_name)
        urllib.request.urlretrieve(item["url"], target)
        downloaded.append(raw_name)

    missing = sorted(wanted - set(downloaded))
    if missing:
        raise RuntimeError(f"TennisMyLife API did not expose expected files: {missing}")
    print(f"Downloaded {len(downloaded)} files")


if __name__ == "__main__":
    main()
