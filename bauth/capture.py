"""Keystroke capture.

The original implementation kept press/release timings in module-level globals,
which made capture non-reentrant and impossible to test. This replaces that with
a self-contained recorder that also keeps the release timestamps the old code
discarded -- those are what make the UD/UU features in features.py possible.
"""

import time

from pynput import keyboard

from .context import capture_context


class KeystrokeCapture:
    """Records press and release timestamps for an expected character sequence.

    Matching follows the original rule: a keypress is recorded only when it is
    the next expected character. Anything else (wrong key, modifier, arrow) is
    ignored for timing purposes, though backspaces are counted separately as a
    correction signal.
    """

    def __init__(self, expected):
        self.expected = list(expected)
        self.press_times = []
        self.release_times = []
        self.corrections = 0
        self.extra_keys = 0

    # -- pynput callbacks ----------------------------------------------------
    def on_press(self, key):
        now = time.time()

        if key == keyboard.Key.enter:
            return False
        if key == keyboard.Key.backspace:
            self.corrections += 1
            return None

        char = getattr(key, "char", None)
        if char is None:
            return None

        index = len(self.press_times)
        if index < len(self.expected) and char == self.expected[index]:
            self.press_times.append(now)
        else:
            self.extra_keys += 1
        return None

    def on_release(self, key):
        now = time.time()

        char = getattr(key, "char", None)
        if char is None:
            return None

        index = len(self.release_times)
        if (
            index < len(self.expected)
            and index < len(self.press_times)
            and char == self.expected[index]
        ):
            self.release_times.append(now)
            if len(self.release_times) == len(self.expected):
                return False  # Whole sequence typed; stop listening.
        return None

    # -- derived timings -----------------------------------------------------
    @property
    def complete(self):
        return len(self.release_times) == len(self.expected) and len(self.expected) > 0

    def hold_times(self):
        """Dwell time: how long each key stays down."""
        return [r - p for p, r in zip(self.press_times, self.release_times)]

    def dd_times(self):
        """Down-down latency between consecutive keys.

        This is what the original code labelled 'flight_times'.
        """
        p = self.press_times
        return [p[i] - p[i - 1] for i in range(1, len(p))]

    def ud_times(self):
        """True flight time: previous key released -> next key pressed.

        Negative values mean the keys overlapped, which is the rollover typing
        that fast touch-typists produce and hunt-and-peck typists never do.
        """
        p, r = self.press_times, self.release_times
        n = min(len(p), len(r))
        return [p[i] - r[i - 1] for i in range(1, n)]

    def uu_times(self):
        """Up-up latency between consecutive key releases."""
        r = self.release_times
        return [r[i] - r[i - 1] for i in range(1, len(r))]

    def timings(self):
        """All four timing vectors, in the order features.assemble expects."""
        return self.hold_times(), self.dd_times(), self.ud_times(), self.uu_times()


def collect_keystroke_data(password, prompt=True):
    """Capture one typing sample of ``password``.

    Returns ``(capture, typed_password, context)``. ``capture`` is None when the
    sequence was not typed cleanly. The context snapshot is taken immediately
    after typing so it reflects the machine the sample came from.
    """
    if prompt:
        print("\nPress Enter to start typing your password.")
        input("")
        print("Type your password and press Enter to stop.")

    recorder = KeystrokeCapture(password)
    with keyboard.Listener(
        on_press=recorder.on_press, on_release=recorder.on_release
    ) as listener:
        typed = input("")
        listener.join()

    context = capture_context()

    if not recorder.complete:
        print("\nPassword length mismatch. This attempt will be skipped.")
        return None, typed, context

    return recorder, typed, context
