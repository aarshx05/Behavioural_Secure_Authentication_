"""Salted password hashing.

Earlier versions stored the password in ``metadata.pkl`` in the clear. In a
system whose whole purpose is authentication -- and which collects real typing
data from real people -- that is indefensible, so passwords are now stored as a
salted hash and never written back in recoverable form.

scrypt is used where available because it is memory-hard, which makes bulk
offline guessing expensive rather than merely slow. PBKDF2-HMAC-SHA256 is the
fallback for builds whose OpenSSL lacks scrypt.

The password *length* is stored alongside the hash. That is not a leak: the
feature vector length is ``4n + 12``, so anyone holding a profile can already
read the length straight off the stored samples. The layout code needs it, so
it is recorded explicitly rather than recovered by arithmetic.
"""

import hashlib
import hmac
import os

SALT_BYTES = 16
KEY_BYTES = 32

# ~16 MB of memory per hash: enough to be costly in bulk, small enough that a
# login stays imperceptible.
SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1

PBKDF2_ROUNDS = 480_000


def _scrypt_available():
    try:
        hashlib.scrypt(b"x", salt=b"y", n=2, r=1, p=1, dklen=16)
        return True
    except (ValueError, AttributeError):
        return False


_HAS_SCRYPT = _scrypt_available()


def _derive(password, salt, record):
    """Derive the key for ``record``'s parameters."""
    raw = password.encode("utf-8")
    if record["scheme"] == "scrypt":
        return hashlib.scrypt(
            raw,
            salt=salt,
            n=record["n"],
            r=record["r"],
            p=record["p"],
            dklen=record["dklen"],
            maxmem=record["n"] * record["r"] * 256 + (1 << 20),
        )
    if record["scheme"] == "pbkdf2_sha256":
        return hashlib.pbkdf2_hmac("sha256", raw, salt, record["rounds"], record["dklen"])
    raise ValueError(f"unknown password scheme: {record['scheme']!r}")


def hash_password(password):
    """Hash ``password`` with a fresh random salt.

    Returns a self-describing record, so the parameters can be strengthened
    later without invalidating existing profiles.
    """
    salt = os.urandom(SALT_BYTES)
    if _HAS_SCRYPT:
        record = {
            "scheme": "scrypt",
            "n": SCRYPT_N,
            "r": SCRYPT_R,
            "p": SCRYPT_P,
            "dklen": KEY_BYTES,
        }
    else:
        record = {
            "scheme": "pbkdf2_sha256",
            "rounds": PBKDF2_ROUNDS,
            "dklen": KEY_BYTES,
        }
    record["salt"] = salt
    record["hash"] = _derive(password, salt, record)
    return record


def verify_password(password, record):
    """Constant-time check of ``password`` against a stored record."""
    if not record or "hash" not in record or "salt" not in record:
        return False
    try:
        candidate = _derive(password, record["salt"], record)
    except (ValueError, KeyError, TypeError):
        return False
    # compare_digest to avoid leaking how much of the hash matched via timing.
    return hmac.compare_digest(candidate, record["hash"])


def describe(record):
    """Human-readable summary of how a password is stored."""
    if not record:
        return "not set"
    if record.get("scheme") == "scrypt":
        return f"scrypt (N={record['n']}, r={record['r']}, p={record['p']})"
    if record.get("scheme") == "pbkdf2_sha256":
        return f"pbkdf2-sha256 ({record['rounds']:,} rounds)"
    return str(record.get("scheme", "unknown"))
