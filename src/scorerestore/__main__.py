"""Allow ``python -m scorerestore`` to use the public CLI."""

from scorerestore.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
