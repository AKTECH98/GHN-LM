#!/usr/bin/env python3
"""Generate GitHub raw URL manifest for Notion image embeds."""

import json
from pathlib import Path

REPO = "AKTECH98/GHN-LM"
BRANCH = "main"
BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "docs" / "assets"


def main() -> None:
    manifest = {"base_url": BASE, "assets": {}}
    for path in sorted(ASSETS_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".png", ".pdf"}:
            rel = path.relative_to(REPO_ROOT).as_posix()
            manifest["assets"][path.name] = f"{BASE}/{rel}"
    manifest["assets"]["report.pdf"] = f"{BASE}/docs/report.pdf"
    out = Path(__file__).resolve().parents[1] / "docs" / "assets" / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest['assets'])} URLs to {out}")


if __name__ == "__main__":
    main()
