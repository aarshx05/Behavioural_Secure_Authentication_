"""Keystroke capture.

Timing comes from a pynput listener, which is the only way to see key *release*
events -- the console gives us characters but never tells us how long a key was
held.

Two things this has to get right:

* **The listener must be live before the user types.** Creating a Listener does
  not mean its OS keyboard hook is installed yet; that takes a few milliseconds.
  Reading input during that window loses the opening keystrokes, which shows up
  as an intermittent "password length mismatch" that depends on how fast the
  user types. ``Listener.wait()`` blocks until the hook is actually active.

* **The console input buffer must be drained afterwards.** The characters are
  still queued for stdin even though nothing read them, and would otherwise be
  swallowed by the next prompt.
"""

import os
import sys
import time

from pynput import keyboard

from .context import capture_context

# Set BAUTH_DEBUG_KEYS=1 to print every key event as it is seen. Useful when a
# particular keyboard layout produces something unexpected.
_DEBUG = os.environ.get("BAUTH_DEBUG_KEYS") == "1"

_SHIFT_KEYS = frozenset(
    k for k in (
        getattr(keyboard.Key, name, None)
        for name in ("shift", "shift_l", "shift_r")
    ) if k is not None
)

# Characters produced by Shift + the given key on a US layout, used only as a
# fallback when pynput cannot resolve the character itself.
_SHIFTED_DIGITS = {
    "1": "!", "2": "@", "3": "#", "4": "$", "5": "%",
    "6": "^", "7": "&", "8": "*", "9": "(", "0": ")",
    "-": "_", "=": "+", "[": "{", "]": "}", "\\": "|",
    ";": ":", "'": '"', ",": "<", ".": ">", "/": "?", "`": "~",
}


def _key_id(key):
    """Stable identifier for a physical key, independent of modifier state.

    ``key.char`` changes with Shift -- the same physical key reports "D" on
    press and "d" on release if Shift is let go in between -- so it cannot be
    used to pair a release with its press. The virtual key code does not move.
    """
    vk = getattr(key, "vk", None)
    if vk is not None:
        return ("vk", vk)
    value = getattr(key, "value", None)
    vk = getattr(value, "vk", None)
    if vk is not None:
        return ("vk", vk)
    char = getattr(key, "char", None)
    return ("char", char.lower()) if char else ("key", str(key))


def _resolve_char(key, shift_held):
    """Best-effort character for a key press.

    pynput usually supplies ``key.char``, but on Windows it can return None for
    a shifted letter depending on how the layout resolves. Dropping the
    keystroke in that case silently shortens the sample -- which looked like
    "4 characters typed" for a 5-character password beginning with a capital.
    """
    char = getattr(key, "char", None)
    if char and len(char) == 1 and ord(char) >= 32:
        return char

    vk = getattr(key, "vk", None)
    if vk is None:
        return None

    if 0x41 <= vk <= 0x5A:  # A-Z
        letter = chr(vk)
        return letter if shift_held else letter.lower()
    if 0x30 <= vk <= 0x39:  # 0-9 across the top row
        digit = chr(vk)
        return _SHIFTED_DIGITS.get(digit, digit) if shift_held else digit
    if 0x60 <= vk <= 0x69:  # numpad 0-9
        return chr(vk - 0x60 + 0x30)
    return None


def _drain_stdin():
    """Discard anything the console buffered while we were listening."""
    try:
        import msvcrt

        while msvcrt.kbhit():
            msvcrt.getwch()
    except ImportError:
        try:
            import termios

            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except (ImportError, OSError):
            pass


class KeystrokeCapture:
    """Records what was typed and the press/release timing of every character.

    Unlike the earlier version this records *all* printable keys rather than
    only those matching the expected password, so the caller can compare what
    was actually typed and report a genuine mismatch instead of silently
    producing a short sample.
    """

    def __init__(self, expected):
        self.expected = list(expected)
        self.typed = []
        self.press_times = []
        # One slot per press, filled in on release. Kept as a parallel list so
        # overlapping keys (rollover typing) resolve to the right press.
        self.release_times = []
        # Physical key identity per press, used to pair releases.
        self.key_ids = []
        self.corrections = 0
        self.finished = False
        self._shift_held = False

    # -- pynput callbacks ----------------------------------------------------
    def on_press(self, key):
        now = time.time()

        if _DEBUG:
            print(
                f"\n    [down] {key!r} char={getattr(key, 'char', None)!r} "
                f"vk={getattr(key, 'vk', None)}",
                flush=True,
            )

        if key in _SHIFT_KEYS:
            self._shift_held = True
            return None

        if key == keyboard.Key.enter:
            self.finished = True
            return False

        if key == keyboard.Key.backspace:
            self.corrections += 1
            if self.typed:
                self.typed.pop()
                self.press_times.pop()
                self.release_times.pop()
                self.key_ids.pop()
            return None

        char = _resolve_char(key, self._shift_held)
        if char is None:
            return None  # modifier, function key, or control character

        self.typed.append(char)
        self.press_times.append(now)
        self.release_times.append(None)
        self.key_ids.append(_key_id(key))
        return None

    def on_release(self, key):
        now = time.time()

        if _DEBUG:
            print(
                f"    [ up ] {key!r} char={getattr(key, 'char', None)!r} "
                f"vk={getattr(key, 'vk', None)}",
                flush=True,
            )

        if key in _SHIFT_KEYS:
            self._shift_held = False
            return None

        # Pair by physical key rather than by character: releasing Shift before
        # the letter turns a "D" press into a "d" release, so matching on the
        # character would never find the press and the sample would be thrown
        # away for every capital letter or shifted symbol.
        target = _key_id(key)
        for i in range(len(self.key_ids) - 1, -1, -1):
            if self.key_ids[i] == target and self.release_times[i] is None:
                self.release_times[i] = now
                return None
        return None

    # -- results -------------------------------------------------------------
    @property
    def text(self):
        return "".join(self.typed)

    @property
    def complete(self):
        """True when the expected password was typed and fully released."""
        return (
            len(self.expected) > 0
            and self.text == "".join(self.expected)
            and all(t is not None for t in self.release_times)
        )

    def _released(self):
        """Release times, substituting the press time for anything still open.

        A key can still be down when Enter arrives; treating it as a zero-length
        hold is better than discarding an otherwise good sample.
        """
        return [
            r if r is not None else p
            for p, r in zip(self.press_times, self.release_times)
        ]

    def hold_times(self):
        """Dwell time: how long each key stays down."""
        return [r - p for p, r in zip(self.press_times, self._released())]

    def dd_times(self):
        """Down-down latency between consecutive keys."""
        p = self.press_times
        return [p[i] - p[i - 1] for i in range(1, len(p))]

    def ud_times(self):
        """True flight time: previous key released -> next key pressed.

        Negative values mean the keys overlapped, which is the rollover typing
        that fast touch-typists produce and hunt-and-peck typists never do.
        """
        p, r = self.press_times, self._released()
        return [p[i] - r[i - 1] for i in range(1, len(p))]

    def uu_times(self):
        """Up-up latency between consecutive key releases."""
        r = self._released()
        return [r[i] - r[i - 1] for i in range(1, len(r))]

    def timings(self):
        """All four timing vectors, in the order features.assemble expects."""
        return self.hold_times(), self.dd_times(), self.ud_times(), self.uu_times()


class EventCapture:
    """Timings supplied by a client rather than read from a local keyboard.

    The web UI records ``keydown``/``keyup`` in the browser, which avoids the
    OS-level keyboard hook entirely -- no accessibility permissions, no console
    echo, and the password can sit in a masked field. Events arrive as
    ``[{"char": "a", "down": <ms>, "up": <ms>}, ...]`` with millisecond
    timestamps from a monotonic clock.

    Exposes the same ``timings()`` surface as :class:`KeystrokeCapture` so
    everything downstream is shared between the CLI and the web UI.
    """

    def __init__(self, events, expected, corrections=0):
        self.expected = list(expected)
        self.corrections = int(corrections)

        cleaned = [
            e
            for e in events
            if isinstance(e.get("char"), str)
            and len(e["char"]) == 1
            and e.get("down") is not None
            and e.get("up") is not None
        ]
        cleaned.sort(key=lambda e: e["down"])

        self.typed = [e["char"] for e in cleaned]
        # Browser timestamps are milliseconds; the rest of the system is seconds.
        self.press_times = [float(e["down"]) / 1000.0 for e in cleaned]
        self.release_times = [float(e["up"]) / 1000.0 for e in cleaned]

    @property
    def text(self):
        return "".join(self.typed)

    @property
    def complete(self):
        return len(self.expected) > 0 and self.text == "".join(self.expected)

    def hold_times(self):
        return [max(0.0, r - p) for p, r in zip(self.press_times, self.release_times)]

    def dd_times(self):
        p = self.press_times
        return [p[i] - p[i - 1] for i in range(1, len(p))]

    def ud_times(self):
        p, r = self.press_times, self.release_times
        return [p[i] - r[i - 1] for i in range(1, len(p))]

    def uu_times(self):
        r = self.release_times
        return [r[i] - r[i - 1] for i in range(1, len(r))]

    def timings(self):
        return self.hold_times(), self.dd_times(), self.ud_times(), self.uu_times()


def collect_keystroke_data(password, prompt=True):
    """Capture one typing sample of ``password``.

    Returns ``(capture, typed_text, context)``. ``capture`` is None when the
    sequence was not typed cleanly, in which case ``typed_text`` still reports
    what was actually entered so the caller can explain the mismatch.
    """
    recorder = KeystrokeCapture(password)
    listener = keyboard.Listener(on_press=recorder.on_press, on_release=recorder.on_release)
    listener.start()
    try:
        listener.wait()  # block until the keyboard hook is genuinely active
        if prompt:
            print("  Type the password and press Enter: ", end="", flush=True)
        listener.join()
    finally:
        listener.stop()

    _drain_stdin()
    print()

    context = capture_context()
    typed = recorder.text

    if not recorder.complete:
        if typed != password:
            print(f"  That did not match the password ({len(typed)} characters typed).")
        else:
            print("  Capture was incomplete. Please try again.")
        return None, typed, context

    return recorder, typed, context
