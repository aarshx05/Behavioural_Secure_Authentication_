"""Entry point for the Keystroke Authentication web dashboard.

Run with:

    python webapp.py

then open http://127.0.0.1:5000 in a browser.
"""

import argparse

from bauth.webapp import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keystroke Authentication dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(host=args.host, port=args.port, debug=args.debug)
