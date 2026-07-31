"""CMU keystroke dynamics benchmark (Killourhy & Maxion, DSN 2009).

Dataset: ``DSL-StrongPasswordData.csv`` from https://www.cs.cmu.edu/~keystroke/
51 subjects typing the password ``.tie5Roanl`` 400 times each, across 8
sessions of 50 repetitions, giving 20,400 rows.

Each row holds 31 timing columns for the 11 keystrokes (10 characters plus
Return):

* ``H.<key>``          -- hold (dwell) time, 11 columns
* ``DD.<k1>.<k2>``     -- down-down latency, 10 columns
* ``UD.<k1>.<k2>``     -- up-down latency (true flight), 10 columns

Up-up latency is not published but is exactly recoverable. With
``press[i+1] = press[i] + DD[i]`` and ``release[i] = press[i] + H[i]``::

    UU[i] = release[i+1] - release[i] = DD[i] + H[i+1] - H[i]

The same algebra gives ``UD[i] = DD[i] - H[i]``, which the file already
contains -- :func:`check_consistency` uses that redundancy to verify the
loader against the published columns rather than trusting the parse.
"""

import csv
import os

import numpy as np

from bauth import features

URL = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
DEFAULT_PATH = os.path.join("data", "DSL-StrongPasswordData.csv")

PASSWORD = ".tie5Roanl"
# 'R' is typed with Shift, which the dataset records as its own keystroke.
KEYS = ["period", "t", "i", "e", "five", "Shift.r", "o", "a", "n", "l", "Return"]
TRANSITIONS = list(zip(KEYS[:-1], KEYS[1:]))

N_KEYS = len(KEYS)                 # 11
N_TRANSITIONS = len(TRANSITIONS)   # 10
N_RAW = N_KEYS + 2 * N_TRANSITIONS  # 31, the published feature set


class CMUData:
    """Parsed dataset held as parallel arrays."""

    def __init__(self, subjects, sessions, reps, hold, dd, ud):
        self.subjects = np.asarray(subjects)
        self.sessions = np.asarray(sessions, dtype=int)
        self.reps = np.asarray(reps, dtype=int)
        self.hold = np.asarray(hold, dtype=float)
        self.dd = np.asarray(dd, dtype=float)
        self.ud = np.asarray(ud, dtype=float)

    def __len__(self):
        return len(self.subjects)

    @property
    def subject_ids(self):
        return sorted(set(self.subjects.tolist()))

    @property
    def uu(self):
        """Up-up latency, derived: UU[i] = DD[i] + H[i+1] - H[i]."""
        return self.dd + self.hold[:, 1:] - self.hold[:, :-1]

    def mask(self, subject):
        return self.subjects == subject

    def raw31(self, index=slice(None)):
        """The published 31-column feature set, as used by the paper."""
        return np.hstack([self.hold[index], self.dd[index], self.ud[index]])

    def raw21(self, index=slice(None)):
        """Hold and down-down only: the non-redundant 21 columns.

        A keystroke sequence has 2n-1 degrees of freedom relative to the first
        press: n hold times and n-1 down-down latencies. Every other timing is
        an exact linear function of those two --

            UD[i] = DD[i] - H[i]
            UU[i] = DD[i] + H[i+1] - H[i]

        -- so the published 31-column set carries 10 redundant columns, and
        this project's 56-column set carries 35. Redundancy is not necessarily
        harmful (it can make structure explicit for a tree), but it is not new
        information, and it costs dimensions.
        """
        return np.hstack([self.hold[index], self.dd[index]])

    def extended(self, index=slice(None)):
        """This project's feature vector: 4n + 12 = 56 columns for n = 11."""
        hold, dd, ud, uu = (
            self.hold[index], self.dd[index], self.ud[index], self.uu[index],
        )
        return np.array([
            features.assemble(hold[i], dd[i], ud[i], uu[i], extended=True)
            for i in range(len(hold))
        ])


def _column_names():
    hold_cols = [f"H.{k}" for k in KEYS]
    dd_cols = [f"DD.{a}.{b}" for a, b in TRANSITIONS]
    ud_cols = [f"UD.{a}.{b}" for a, b in TRANSITIONS]
    return hold_cols, dd_cols, ud_cols


def load(path=DEFAULT_PATH):
    """Read the CSV into a :class:`CMUData`.

    Columns are looked up by name rather than position, so a reordered export
    would still parse correctly.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download it with:\n"
            f"    python run_eval.py --download\n"
            f"or fetch {URL} manually."
        )

    hold_cols, dd_cols, ud_cols = _column_names()

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [
            c for c in hold_cols + dd_cols + ud_cols if c not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"missing expected columns: {missing[:5]}")

        subjects, sessions, reps, hold, dd, ud = [], [], [], [], [], []
        for row in reader:
            subjects.append(row["subject"])
            sessions.append(int(row["sessionIndex"]))
            reps.append(int(row["rep"]))
            hold.append([float(row[c]) for c in hold_cols])
            dd.append([float(row[c]) for c in dd_cols])
            ud.append([float(row[c]) for c in ud_cols])

    return CMUData(subjects, sessions, reps, hold, dd, ud)


def check_consistency(data, tolerance=1e-6):
    """Verify the parse using the file's own redundancy.

    ``UD = DD - H`` must hold for every transition. If the loader mis-assigned a
    column this fails immediately, which is a far better failure than silently
    producing wrong features.
    """
    predicted = data.dd - data.hold[:, :-1]
    error = np.abs(predicted - data.ud)
    worst = float(error.max())
    return worst <= tolerance, worst


def describe(data):
    counts = {s: int((data.subjects == s).sum()) for s in data.subject_ids}
    per_subject = sorted(set(counts.values()))
    return {
        "rows": len(data),
        "subjects": len(data.subject_ids),
        "reps_per_subject": per_subject[0] if len(per_subject) == 1 else per_subject,
        "sessions": sorted(set(data.sessions.tolist())),
        "keys": N_KEYS,
        "raw_features": N_RAW,
        "extended_features": features.feature_dim(N_KEYS, True),
    }


def download(path=DEFAULT_PATH):
    """Fetch the dataset from CMU."""
    from urllib.request import urlopen

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with urlopen(URL, timeout=120) as response, open(path, "wb") as out:
        out.write(response.read())
    return os.path.getsize(path)
