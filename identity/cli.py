"""Local-only experimental enrollment. Never use a synced folder for the identity database."""

import argparse
import json
from pathlib import Path

import numpy as np

from detection.adapters import AdapterConfig
from detection.recording import read_recording
from identity.store import CONSENT_TEXT, CONSENT_VERSION, IdentityStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-local-identity", action="store_true")
    parser.add_argument("--db", type=Path, default=Path(".identity/local.db"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("consent")
    sub.add_parser("list")
    delete = sub.add_parser("delete")
    delete.add_argument("--subject", required=True)
    for name in ("enroll", "match"):
        command = sub.add_parser(name)
        command.add_argument("--replay", type=Path, required=True)
        command.add_argument("--config", type=Path, default=Path("configs/synthetic.json"))
        if name == "enroll":
            command.add_argument("--subject", required=True)
            command.add_argument("--accept-consent", choices=[CONSENT_VERSION])
        else:
            command.add_argument("--max-distance", type=float, default=3)
            command.add_argument("--min-margin", type=float, default=0.5)
    args = parser.parse_args()
    if args.command == "consent":
        print(f"{CONSENT_VERSION}\n{CONSENT_TEXT}")
        return
    if not args.enable_local_identity:
        parser.error("identity is disabled; --enable-local-identity is required")
    if args.command == "enroll" and args.accept_consent != CONSENT_VERSION:
        parser.error(
            f"Read `python -m identity.cli consent` and explicitly accept {CONSENT_VERSION}"
        )
    with IdentityStore(args.db, enabled=True) as store:
        if args.command == "list":
            print(json.dumps(store.list_subjects()))
        elif args.command == "delete":
            print(json.dumps({"deleted": store.delete(args.subject)}))
        else:
            config = AdapterConfig.model_validate_json(args.config.read_text())
            windows = [w.values for w in read_recording(args.replay, config)]
            if not windows:
                raise ValueError("recording contains no complete windows")
            if args.command == "enroll":
                store.enroll(args.subject, np.stack(windows), config, consent=CONSENT_TEXT)
                print("Enrolled locally. No raw windows copied; no cloud upload.")
            else:
                for window in windows:
                    print(
                        json.dumps(
                            store.match(
                                window,
                                config,
                                config.sample_rate_hz,
                                max_distance=args.max_distance,
                                min_margin=args.min_margin,
                            )
                        )
                    )


if __name__ == "__main__":
    main()
