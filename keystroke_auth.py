"""Entry point for the Keystroke Authentication System.

The implementation lives in the ``bauth`` package; this keeps the documented
``python keystroke_auth.py`` command working.
"""

from bauth.cli import cli

if __name__ == "__main__":
    cli()
