"""One-off script to grant admin privileges to a user.

There's no API endpoint for this by design — the very first admin can't
be created through an admin-only API (nothing would be able to authorize
it). Run this directly against the database instead:

    python scripts/promote_admin.py someone@example.com
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal
from app.db_models import User


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/promote_admin.py <email>", file=sys.stderr)
        sys.exit(1)

    email = sys.argv[1]
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).one_or_none()
        if user is None:
            print(f"No user found with email {email!r}", file=sys.stderr)
            sys.exit(1)

        user.is_admin = True
        db.commit()
        print(f"{email} is now an admin.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
