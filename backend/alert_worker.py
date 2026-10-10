"""External bounded alert worker; never run a background loop inside a serverless handler."""

import argparse
import json
import os
import time

from dotenv import load_dotenv
from sqlalchemy import update
from sqlalchemy.orm import Session

from backend.alerts import AlertSettings, deliver_one
from backend.db import AlertRow, make_engine


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="process one bounded batch then exit")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--requeue-dead", action="store_true", help="after fixing configuration")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 100:
        parser.error("batch size must be between 1 and 100")
    settings = AlertSettings.from_env()
    if settings.mode == "off":
        parser.error("set ALERT_MODE=mock or telegram explicitly")
    engine = make_engine(os.environ["DATABASE_URL"])
    try:
        if args.requeue_dead:
            with Session(engine) as db:
                db.execute(
                    update(AlertRow)
                    .where(
                        AlertRow.mode == settings.mode,
                        AlertRow.status == "dead",
                        AlertRow.expires_at > time.time(),
                    )
                    .values(status="pending", next_attempt=0, attempts=0, error=None)
                )
                db.commit()
        while True:
            counts = {}
            for _ in range(args.batch_size):
                status = deliver_one(engine, settings)
                counts[status] = counts.get(status, 0) + 1
                if status in {"idle", "busy"}:
                    break
                time.sleep(1)  # Avoid bursts into one Telegram chat.
            print(json.dumps(counts), flush=True)
            if args.once:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
