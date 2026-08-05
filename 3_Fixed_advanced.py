## GUI Room Mapping - EAI/YDLIDAR X2 (corrected protocol implementation)
##
## Fixes applied vs. original script:
## 1. Packet parser was misaligned by 2 bytes: it never read the CS
##    (checksum) field between LSA and the sample data, so it treated the
##    checksum bytes as the first sample and shifted every subsequent
##    sample early by 2 bytes. Every distance/quality value in every
##    packet was effectively garbage.
## 2. Angle decode was raw/100; official formula is (raw >> 1) / 64.0.
## 3. Distance decode used the raw 16-bit value directly as mm; official
##    formula for triangulation lidars is raw/4 (no-intensity format) or
##    (S2<<8|S1)>>2 (intensity format).
## 4. Checksum is now computed and packets that fail validation are
##    dropped instead of being fed into the map.
## 5. Sample format (2-byte no-intensity vs 3-byte intensity) is
##    auto-detected per packet by testing which one satisfies the
##    checksum, since this is a "rakitan" unit of uncertain firmware.
## 6. Second-level angle correction (triangulation geometry) applied.
## 7. Zero/frequency packets (CT & 0x01 == 1) are no longer mapped as
##    real points.
## 8. Occupancy grid changed from permanent ternary (-1/0/1) to a
##    log-odds probabilistic grid so isolated noisy hits decay instead
##    of leaving permanent phantom walls.
## 9. Reader thread and GUI thread no longer share mutable lists without
##    synchronization; a lock guards the handoff.
## 10. Grid is rendered via a single QImage blit instead of ~100k
##     individual QPainter.drawPoint calls per frame.
##
## Wiring note: this device is an EAI/YDLIDAR X2 (protocol confirmed
## against YDLIDAR-SDK-Communication-Protocol.md), not a Slamtec RPLIDAR.
## X2 needs a clean 5V supply capable of >=1A at spin-up; per the official
## manual, USB port current alone is often insufficient and a dedicated
## 5V auxiliary supply is required or the lidar will misbehave.
##
## Requirements: pip install numpy pyserial PyQt5

import sys
import struct
import threading
import time
import math

import numpy as np
import serial

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QImage, QColor
from PyQt5.QtCore import Qt, QTimer, QRectF

## ============================= CONFIGURATION =============================
PORT = '/dev/ttyUSB0'          # change this
BAUD = 115200

MAP_SIZE_METERS = 16.0
MAP_RESOLUTION = 0.05          # meters/pixel
MAP_SIZE_PIXELS = int(MAP_SIZE_METERS / MAP_RESOLUTION)

# Per official X2 datasheet: usable range ~0.10 - 8.0 m. Reject anything
# outside this as physically implausible for this sensor.
MIN_VALID_RANGE_M = 0.10
MAX_VALID_RANGE_M = 8.0

# Minimum intensity to accept a point, only used if the connected unit
# turns out to report intensity (auto-detected at runtime). Tune this
# against your own unit; start permissive and tighten once you see how
# noisy your specific clone is.
MIN_INTENSITY = 0

# Log-odds occupancy grid parameters (standard SLAM log-odds update).
LOGODDS_OCC = 0.85
LOGODDS_FREE = -0.4
LOGODDS_CLAMP = 5.0
OCC_DISPLAY_THRESHOLD = 0.5
FREE_DISPLAY_THRESHOLD = -0.5
## =========================================================================

ROBOT_GX = int((MAP_SIZE_METERS / 2) / MAP_RESOLUTION)
ROBOT_GY = int((MAP_SIZE_METERS / 2) / MAP_RESOLUTION)


# ---------------------------------------------------------------------------
# Protocol parsing
# ---------------------------------------------------------------------------

def _checksum_no_intensity(fsa_raw, ct, lsn, lsa_raw, sample_words):
    cs = 0x55AA
    cs ^= fsa_raw
    for w in sample_words:
        cs ^= w
    cs ^= ((lsn << 8) | ct) & 0xFFFF
    cs ^= lsa_raw
    return cs & 0xFFFF


def _checksum_intensity(fsa_raw, ct, lsn, lsa_raw, triplets):
    cs = 0x55AA
    cs ^= fsa_raw
    for i_byte, d_word in triplets:
        cs ^= i_byte
        cs ^= d_word
    cs ^= ((lsn << 8) | ct) & 0xFFFF
    cs ^= lsa_raw
    return cs & 0xFFFF


def try_parse_packet(buffer):
    """
    Attempt to parse one scan packet starting at buffer[0].
    Returns (consumed_bytes, result_dict_or_None).
    result_dict has keys: angles_deg (list), distances_m (list)
    If the header doesn't look like a valid packet start, consumed_bytes=1
    (caller should drop one byte and retry, standard resync behaviour).
    """
    if len(buffer) < 10:
        return 0, None  # need more data

    if buffer[0] != 0xAA or buffer[1] != 0x55:
        return 1, None  # resync

    ct = buffer[2]
    lsn = buffer[3]

    # Sanity cap: a corrupted/false sync can produce a huge bogus LSN.
    # Real packets are small; refuse to "believe" absurd sizes and resync.
    if lsn == 0 or lsn > 200:
        return 1, None

    fsa_raw = struct.unpack('<H', buffer[4:6])[0]
    lsa_raw = struct.unpack('<H', buffer[6:8])[0]
    cs = struct.unpack('<H', buffer[8:10])[0]

    # --- Candidate 1: no-intensity format, Si = 2 bytes ---
    size_no_int = 10 + 2 * lsn
    if len(buffer) >= size_no_int:
        words = [struct.unpack('<H', buffer[10 + 2 * i:12 + 2 * i])[0] for i in range(lsn)]
        if _checksum_no_intensity(fsa_raw, ct, lsn, lsa_raw, words) == cs:
            distances_mm = [w / 4.0 for w in words]
            return size_no_int, _finish_packet(ct, lsn, fsa_raw, lsa_raw, distances_mm, None)

    # --- Candidate 2: intensity format, Si = 3 bytes (I + D) ---
    size_int = 10 + 3 * lsn
    if len(buffer) >= size_int:
        triplets = []
        distances_mm = []
        intensities = []
        for i in range(lsn):
            off = 10 + 3 * i
            b0, b1, b2 = buffer[off], buffer[off + 1], buffer[off + 2]
            d_word = (b2 << 8) | b1
            triplets.append((b0, d_word))
            distances_mm.append(d_word >> 2)
            intensities.append(((b1 & 0x03) << 8) | b0)
        if _checksum_intensity(fsa_raw, ct, lsn, lsa_raw, triplets) == cs:
            return size_int, _finish_packet(ct, lsn, fsa_raw, lsa_raw, distances_mm, intensities)

    # Neither candidate validated against the checksum -> not a real
    # header at this position, resync by one byte.
    return 1, None


def _finish_packet(ct, lsn, fsa_raw, lsa_raw, distances_mm, intensities):
    # CT & 0x01 == 1 -> zero/frequency packet, not real scan data.
    if ct & 0x01:
        return {'angles_deg': [], 'distances_m': []}

    angle_fsa = (fsa_raw >> 1) / 64.0
    angle_lsa = (lsa_raw >> 1) / 64.0
    diff = angle_lsa - angle_fsa
    if diff < 0:
        diff += 360.0

    angles_deg = []
    distances_m = []
    for i in range(lsn):
        d_mm = distances_mm[i]
        if intensities is not None and intensities[i] < MIN_INTENSITY:
            continue
        d_m = d_mm / 1000.0
        if not (MIN_VALID_RANGE_M <= d_m <= MAX_VALID_RANGE_M):
            continue

        if lsn > 1:
            angle = i * diff / (lsn - 1) + angle_fsa
        else:
            angle = angle_fsa

        # Second-level (triangulation geometry) angle correction.
        if d_mm > 0:
            ang_correct = math.atan(21.8 * (155.3 - d_mm) / (155.3 * d_mm))
            angle += ang_correct * 180.0 / math.pi
        angle %= 360.0

        angles_deg.append(angle)
        distances_m.append(d_m)

    return {'angles_deg': angles_deg, 'distances_m': distances_m}


# ---------------------------------------------------------------------------
# Shared state (thread-safe handoff between reader thread and GUI thread)
# ---------------------------------------------------------------------------

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.occupancy = np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS), dtype=np.float32)
        self.latest_angles = np.empty(0, dtype=np.float64)
        self.latest_distances = np.empty(0, dtype=np.float64)
        self.packets_ok = 0
        self.bytes_resynced = 0

    def integrate_scan(self, angles_deg, distances_m):
        if not angles_deg:
            return
        angles_arr = np.array(angles_deg, dtype=np.float64)
        dist_arr = np.array(distances_m, dtype=np.float64)
        theta = np.deg2rad(angles_arr)
        xs = (MAP_SIZE_METERS / 2) + dist_arr * np.cos(theta)
        ys = (MAP_SIZE_METERS / 2) + dist_arr * np.sin(theta)
        gxs = (xs / MAP_RESOLUTION).astype(np.int32)
        gys = (ys / MAP_RESOLUTION).astype(np.int32)

        with self.lock:
            grid = self.occupancy
            h, w = grid.shape
            for gx, gy in zip(gxs, gys):
                for x, y in _bresenham(ROBOT_GX, ROBOT_GY, gx, gy)[:-1]:
                    if 0 <= x < w and 0 <= y < h:
                        v = grid[y, x] + LOGODDS_FREE
                        grid[y, x] = v if v > -LOGODDS_CLAMP else -LOGODDS_CLAMP
                if 0 <= gx < w and 0 <= gy < h:
                    v = grid[gy, gx] + LOGODDS_OCC
                    grid[gy, gx] = v if v < LOGODDS_CLAMP else LOGODDS_CLAMP
            self.latest_angles = angles_arr
            self.latest_distances = dist_arr

    def snapshot(self):
        with self.lock:
            return self.occupancy.copy(), self.latest_angles, self.latest_distances


def _bresenham(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


# ---------------------------------------------------------------------------
# Serial reader thread
# ---------------------------------------------------------------------------

def read_lidar(state: SharedState, stop_event: threading.Event):
    buffer = bytearray()
    ser = None
    while not stop_event.is_set():
        try:
            if ser is None:
                ser = serial.Serial(PORT, BAUD, timeout=1)
                buffer.clear()
            chunk = ser.read(1024)
            if not chunk:
                continue
            buffer += chunk

            while True:
                consumed, result = try_parse_packet(bytes(buffer))
                if consumed == 0:
                    break  # need more bytes
                del buffer[:consumed]
                if result is not None:
                    state.packets_ok += 1
                    state.integrate_scan(result['angles_deg'], result['distances_m'])
                else:
                    state.bytes_resynced += 1
        except (serial.SerialException, OSError) as e:
            print(f"[serial error] {e}, reconnecting in 1s...")
            try:
                if ser:
                    ser.close()
            except Exception:
                pass
            ser = None
            time.sleep(1.0)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class RadarWidget(QWidget):
    def __init__(self, state: SharedState):
        super().__init__()
        self.state = state
        self.setWindowTitle('GUI Room Mapping - EAI/YDLIDAR X2 (fixed)')
        self.resize(700, 760)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)  # ~20 FPS is plenty for a static occupancy map

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.drawRadar(qp)
        qp.end()

    def drawRadar(self, qp):
        qp.fillRect(self.rect(), Qt.black)
        occ, angles, distances = self.state.snapshot()

        h, w = occ.shape
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[:] = (35, 35, 35)                       # unknown
        rgb[occ <= FREE_DISPLAY_THRESHOLD] = (95, 95, 95)   # free
        rgb[occ >= OCC_DISPLAY_THRESHOLD] = (220, 60, 60)   # occupied

        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        side = min(self.width(), self.height() - 20)
        target = QRectF(0, 0, side, side)
        qp.drawImage(target, img)

        # live scan overlay
        qp.setPen(QPen(QColor(0, 255, 0), 2))
        scale = side / (MAX_VALID_RANGE_M * 2)
        cx, cy = side / 2, side / 2
        for a_deg, d in zip(angles, distances):
            theta = math.radians(a_deg)
            x = cx + d * math.cos(theta) * scale
            y = cy + d * math.sin(theta) * scale
            qp.drawPoint(int(x), int(y))

        qp.setPen(QPen(Qt.white))
        qp.drawText(5, int(side) + 15,
                    f"packets ok: {self.state.packets_ok}  bytes resynced: {self.state.bytes_resynced}")


if __name__ == '__main__':
    shared = SharedState()
    stop_event = threading.Event()
    reader = threading.Thread(target=read_lidar, args=(shared, stop_event), daemon=True)
    reader.start()

    app = QApplication(sys.argv)
    radar = RadarWidget(shared)
    radar.show()
    ret = app.exec_()
    stop_event.set()
    sys.exit(ret)