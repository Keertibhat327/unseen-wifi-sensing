"""Build a minimal API deployment directory from an explicit source allowlist."""

import argparse
import json
import shutil
from pathlib import Path


def prepare(output: Path):
    root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=False)
    for package in ("backend", "contracts"):
        for path in (root / package).glob("*.py"):
            target = output / package / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    (output / "app.py").write_text("from backend.app import create_app\n\napp = create_app()\n")
    shutil.copyfile(root / "backend/requirements.txt", output / "requirements.txt")
    (output / ".python-version").write_text("3.12\n")
    (output / "vercel.json").write_text(
        json.dumps(
            {"$schema": "https://openapi.vercel.sh/vercel.json", "framework": "fastapi"}, indent=2
        )
    )
    print(f"Prepared API source at {output.resolve()}; secrets and raw data excluded.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".deploy/api"))
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()
