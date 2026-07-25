"""Contextual attributes captured alongside each keystroke sample.

These describe *where and when* a sample was typed (network, device, clock)
rather than *how* it was typed. They are deliberately kept out of the machine
learning feature vector -- see risk.py for why and for how they are used.
"""

import getpass
import os
import platform
import socket
import sys
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime

from . import config


def _local_ip():
    """Best-effort primary LAN address.

    Opening a UDP socket and 'connecting' only performs a local routing table
    lookup -- no packets are transmitted. The target is an RFC 5737
    documentation address so nothing real is ever contacted.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 9))
        return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "unknown"
    finally:
        sock.close()


def _public_ip():
    """Look up the external IP. Off by default; contacts a third-party service."""
    if not config.ENABLE_PUBLIC_IP_LOOKUP:
        return None
    try:
        from urllib.request import urlopen

        with urlopen(config.PUBLIC_IP_URL, timeout=config.PUBLIC_IP_TIMEOUT) as resp:
            return resp.read().decode("utf-8").strip()[:64]
    except Exception:
        # Offline, blocked, or slow -- never let this break a login.
        return None


def _mac_address():
    """MAC as a colon-separated string, or None if the OS fabricated one.

    uuid.getnode() sets the multicast bit when it cannot read a real adapter
    address, in which case the value is random per process and useless as a
    device signal.
    """
    node = uuid.getnode()
    if (node >> 40) & 0x01:
        return None
    return ":".join(f"{(node >> shift) & 0xFF:02x}" for shift in range(40, -8, -8))


def _keyboard_layout():
    """Active keyboard layout identifier. Windows only; None elsewhere."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes

        user32 = ctypes.windll.user32
        thread_id = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
        return hex(user32.GetKeyboardLayout(thread_id) & 0xFFFF)
    except Exception:
        return None


def _timezone():
    try:
        local = datetime.now().astimezone()
        offset = local.utcoffset()
        minutes = int(offset.total_seconds() // 60) if offset else 0
        return local.tzname() or time.tzname[0], minutes
    except Exception:
        return "unknown", 0


@dataclass
class CaptureContext:
    """Environment snapshot taken at the moment a sample is typed."""

    timestamp: float = 0.0
    iso_time: str = ""
    hour_of_day: int = 0
    weekday: int = 0
    timezone_name: str = "unknown"
    utc_offset_minutes: int = 0

    hostname: str = "unknown"
    local_ip: str = "unknown"
    public_ip: str = None
    mac_address: str = None

    os_name: str = "unknown"
    os_release: str = ""
    os_version: str = ""
    machine: str = ""
    processor: str = ""
    python_version: str = ""

    username: str = "unknown"
    keyboard_layout: str = None

    def to_dict(self):
        return asdict(self)

    @property
    def device_fingerprint(self):
        """Stable-ish identifier for the machine a sample came from."""
        return f"{self.hostname}|{self.mac_address or 'nomac'}|{self.machine}"

    @property
    def subnet(self):
        """The /24 of the local address, which survives DHCP lease changes."""
        parts = self.local_ip.split(".")
        if len(parts) == 4:
            return ".".join(parts[:3]) + ".0/24"
        return self.local_ip

    def summary(self):
        bits = [
            f"host={self.hostname}",
            f"ip={self.local_ip}",
            f"os={self.os_name} {self.os_release}",
            f"tz={self.timezone_name}",
            f"time={self.iso_time}",
        ]
        if self.public_ip:
            bits.insert(2, f"public_ip={self.public_ip}")
        return "  ".join(bits)


def capture_context():
    """Collect the full contextual snapshot for the current moment."""
    now = time.time()
    local = datetime.fromtimestamp(now)
    tz_name, tz_offset = _timezone()

    try:
        username = getpass.getuser()
    except Exception:
        username = os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"

    try:
        hostname = socket.gethostname()
    except OSError:
        hostname = "unknown"

    return CaptureContext(
        timestamp=now,
        iso_time=local.isoformat(timespec="seconds"),
        hour_of_day=local.hour,
        weekday=local.weekday(),
        timezone_name=tz_name,
        utc_offset_minutes=tz_offset,
        hostname=hostname,
        local_ip=_local_ip(),
        public_ip=_public_ip(),
        mac_address=_mac_address(),
        os_name=platform.system(),
        os_release=platform.release(),
        os_version=platform.version(),
        machine=platform.machine(),
        processor=platform.processor(),
        python_version=platform.python_version(),
        username=username,
        keyboard_layout=_keyboard_layout(),
    )


def from_dict(data):
    """Rebuild a CaptureContext from a stored dict, ignoring unknown keys."""
    if not data:
        return CaptureContext()
    fields = CaptureContext.__dataclass_fields__
    return CaptureContext(**{k: v for k, v in data.items() if k in fields})
