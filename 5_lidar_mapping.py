## =========================================================================
## RPLIDAR 2D Mapping — BreezySlam / Static Mode (No ROS)
## Wasaka Robotic Team | 2026
##
## Mode Otomatis:
##   [SLAM]   jika "pip install breezyslam" sudah dilakukan
##            → robot bisa bergerak, scan matching estimasi pose
##   [STATIS] tanpa breezyslam
##            → robot diam/manual, scan menumpuk di titik asal
##
## Install dependencies:
##   pip install pyserial numpy PyQt5
##   pip install breezyslam          # opsional, aktifkan SLAM mode
##
## Jalankan:
##   python3 5_lidar_mapping.py
##
## Wiring LiDAR:
##   Merah  -> 5V    Kuning -> GND
##   Hitam  -> TX (RX Jetson)    Hijau -> NC
## =========================================================================

import sys, struct, serial, numpy as np, time, math, os
from PyQt5 import QtWidgets, QtCore, QtGui

try:
    from breezyslam.algorithms import RMHC_SLAM
    from breezyslam.sensors import Laser
    SLAM_MODE = True
except ImportError:
    SLAM_MODE = False

## =========================================================================
## KONFIGURASI
## =========================================================================
SERIAL_PORT  = "/dev/ttyTHS1"   # Ganti ke /dev/ttyUSB0 jika via USB
BAUD_RATE    = 115200
MIN_DIST_MM  = 100               # Jarak minimum valid = 10 cm
MAX_DIST_MM  = 8000              # Jarak maksimum valid = 8 m
MIN_QUALITY  = 1                 # Kualitas minimum sinyal
MAP_PIX      = 800               # Ukuran peta N x N pixel
MAP_METERS   = 16.0              # Ukuran area peta (meter x meter)
SCAN_BINS    = 360               # Jumlah bin sudut (1 bin = 1 derajat)
GUI_REFRESH  = 50                # Interval refresh GUI (ms)
SCAN_HZ      = 13                # Estimasi rotasi LiDAR per detik
## =========================================================================


# ---------------------------------------------------------------------------
# PARSER PAKET LIDAR (protokol 0xAA 0x55)
# ---------------------------------------------------------------------------

class LidarPacketParser:
    HEADER_A = 0xAA
    HEADER_B = 0x55
    PKT_SIZE = 47
    N_PTS    = 12

    def __init__(self):
        self.buf = bytearray()

    def feed(self, raw: bytes) -> list:
        """Masukkan data mentah, kembalikan list (angle_deg, dist_mm, quality)."""
        self.buf.extend(raw)
        out = []
        while len(self.buf) >= self.PKT_SIZE:
            if self.buf[0] == self.HEADER_A and self.buf[1] == self.HEADER_B:
                pkt = bytes(self.buf[:self.PKT_SIZE])
                self.buf = self.buf[self.PKT_SIZE:]
                out.extend(self._decode(pkt))
            else:
                self.buf.pop(0)
        return out

    def _decode(self, pkt: bytes) -> list:
        res = []
        try:
            a0 = struct.unpack_from('<H', pkt, 4)[0] / 100.0
            a1 = struct.unpack_from('<H', pkt, 6)[0] / 100.0
            da = (a1 - a0) % 360
            for i in range(self.N_PTS):
                off = 8 + i * 3
                if off + 2 >= len(pkt):
                    break
                d = struct.unpack_from('<H', pkt, off)[0]
                q = pkt[off + 2]
                ang = (a0 + da * i / max(self.N_PTS - 1, 1)) % 360
                if q >= MIN_QUALITY and MIN_DIST_MM <= d <= MAX_DIST_MM:
                    res.append((ang, d, q))
        except Exception:
            pass
        return res


# ---------------------------------------------------------------------------
# WORKER THREAD — baca serial port di background
# ---------------------------------------------------------------------------

class LidarWorker(QtCore.QThread):
    data_sig  = QtCore.pyqtSignal(list)   # list of (angle, dist, quality)
    err_sig   = QtCore.pyqtSignal(str)
    stats_sig = QtCore.pyqtSignal(dict)   # {'fps': float}

    def __init__(self):
        super().__init__()
        self.running = False
        self.parser  = LidarPacketParser()

    def run(self):
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.5)
            self.running = True
        except serial.SerialException as e:
            self.err_sig.emit(
                f"Gagal membuka {SERIAL_PORT}\n{e}\n"
                f"Cek: ls -l /dev/ttyTHS* atau /dev/ttyUSB*"
            )
            return

        frames, t0 = 0, time.time()
        while self.running:
            try:
                raw = ser.read(256)
                if raw:
                    pts = self.parser.feed(raw)
                    if pts:
                        self.data_sig.emit(pts)
                        frames += 1
                dt = time.time() - t0
                if dt >= 1.0:
                    self.stats_sig.emit({'fps': round(frames / dt, 1)})
                    frames, t0 = 0, time.time()
            except serial.SerialException as e:
                self.err_sig.emit(str(e))
                break
        try:
            ser.close()
        except Exception:
            pass

    def stop(self):
        self.running = False
        self.wait(2000)


# ---------------------------------------------------------------------------
# NORMALISASI SCAN → array 360 bin
# ---------------------------------------------------------------------------

def normalize_scan(points: list) -> list:
    """
    Konversi daftar (angle, dist, quality) yang tidak seragam
    menjadi array SCAN_BINS elemen. Setiap elemen = jarak rata-rata
    di bin sudut tersebut. Bin kosong = 0 (tidak terdeteksi).
    """
    total  = [0] * SCAN_BINS
    count  = [0] * SCAN_BINS
    for ang, d, q in points:
        idx = int(ang % SCAN_BINS)
        total[idx]  += d
        count[idx]  += 1
    return [int(total[i] / count[i]) if count[i] else 0
            for i in range(SCAN_BINS)]


# ---------------------------------------------------------------------------
# MESIN PEMETAAN
# ---------------------------------------------------------------------------

class MappingEngine:
    """
    Abstraksi mesin pemetaan dua mode:

    SLAM mode  (breezyslam ada):
        - Scan matching RMHC untuk estimasi pose robot
        - Robot bisa bergerak bebas, peta dibangun secara global

    Statis mode (tanpa breezyslam):
        - Robot diasumsikan diam di tengah peta
        - Raycasting sederhana: jalur sinar = bebas, ujung = objek
        - Cocok untuk mapping satu posisi / demo
    """

    def __init__(self):
        self.mode     = 'slam' if SLAM_MODE else 'static'
        self.mapbytes = bytearray(MAP_PIX * MAP_PIX)
        self.pose     = (0.0, 0.0, 0.0)   # x_mm, y_mm, theta_deg
        self.path_px  = []                 # [(px, py)]

        if self.mode == 'slam':
            laser = Laser(
                scan_size                = SCAN_BINS,
                scan_rate_hz             = SCAN_HZ,
                detection_angle_degrees  = 360,
                distance_no_detection_mm = 0,
                detection_margin         = 0,
                offset_mm                = 0
            )
            self._slam = RMHC_SLAM(
                laser,
                map_size_pixels = MAP_PIX,
                map_size_meters = MAP_METERS,
                random_seed     = 42
            )
        else:
            # Array float untuk akumulasi nilai sel (lebih presisi)
            self._grid  = np.full((MAP_PIX, MAP_PIX), 128.0, dtype=np.float32)
            self._res_mm = (MAP_METERS * 1000) / MAP_PIX  # mm per pixel

    # ── Update dengan scan baru ────────────────────────────────────────────

    def update(self, scan_mm: list):
        if self.mode == 'slam':
            self._update_slam(scan_mm)
        else:
            self._update_static(scan_mm)
        # Catat jalur robot
        px, py, _ = self.get_robot_pixel()
        if not self.path_px or self.path_px[-1] != (px, py):
            self.path_px.append((px, py))

    def _update_slam(self, scan_mm: list):
        self._slam.update(scan_mm)
        x, y, t = self._slam.getpos()
        self.pose = (x, y, t)
        self._slam.getmap(self.mapbytes)

    def _update_static(self, scan_mm: list):
        """
        Raycasting ke occupancy grid.
        - Sel di sepanjang sinar: nilai naik (lebih terang = bebas)
        - Sel di ujung sinar (objek): nilai turun (lebih gelap = objek)
        """
        cx = cy = MAP_PIX // 2
        res = self._res_mm

        for i, d in enumerate(scan_mm):
            if d == 0:
                continue
            ang = math.radians(i)
            cos_a, sin_a = math.cos(ang), -math.sin(ang)  # Y dibalik

            # Bebas: sepanjang sinar
            n_free = min(int(d / res) - 1, MAP_PIX)
            for s in range(1, max(n_free, 1)):
                fx = int(cx + s * cos_a)
                fy = int(cy + s * sin_a)
                if 0 <= fx < MAP_PIX and 0 <= fy < MAP_PIX:
                    self._grid[fy, fx] = min(255.0,
                                             self._grid[fy, fx] + 2.5)

            # Objek: titik ujung
            ox = int(cx + (d / res) * cos_a)
            oy = int(cy + (d / res) * sin_a)
            if 0 <= ox < MAP_PIX and 0 <= oy < MAP_PIX:
                self._grid[oy, ox] = max(0.0, self._grid[oy, ox] - 30.0)

        # Sinkronkan ke mapbytes
        arr = np.clip(self._grid, 0, 255).astype(np.uint8)
        self.mapbytes[:] = arr.tobytes()

    # ── Posisi robot dalam pixel ───────────────────────────────────────────

    def get_robot_pixel(self):
        """Kembalikan (px, py, theta_deg) posisi robot di peta pixel."""
        if self.mode == 'slam':
            x, y, t = self.pose
            scale = MAP_PIX / (MAP_METERS * 1000)
            px = int(MAP_PIX / 2 + x * scale)
            py = int(MAP_PIX / 2 - y * scale)
        else:
            px = py = MAP_PIX // 2
            t = 0.0
        px = max(0, min(MAP_PIX - 1, px))
        py = max(0, min(MAP_PIX - 1, py))
        return px, py, t

    # ── Render peta ke QImage ──────────────────────────────────────────────

    def render_map(self) -> QtGui.QImage:
        """
        Render mapbytes ke QImage dengan color coding:
          Sangat gelap (0)   → dinding/obstacle  → merah gelap
          Abu (128)          → tidak diketahui   → biru-abu gelap
          Terang (255)       → area bebas        → biru muda
        """
        arr = np.frombuffer(self.mapbytes, dtype=np.uint8).reshape(MAP_PIX, MAP_PIX)
        rgb = np.zeros((MAP_PIX, MAP_PIX, 3), dtype=np.uint8)

        if self.mode == 'slam':
            # BreezySlam: 0=unknown, rendah=obstacle, tinggi=bebas
            mask_unk = arr == 0
            mask_occ = (arr > 0)   & (arr < 100)
            mask_trn = (arr >= 100) & (arr < 200)
            mask_fre = arr >= 200

            rgb[mask_unk] = [15,  20,  35]   # gelap: tidak diketahui
            rgb[mask_occ] = [160, 30,  30]   # merah: obstacle

            # Transisi
            t_val = arr[mask_trn].astype(np.float32)
            ratio = (t_val - 100) / 100.0
            rgb[mask_trn, 0] = (160 * (1 - ratio)).astype(np.uint8)
            rgb[mask_trn, 1] = (80  * ratio).astype(np.uint8)
            rgb[mask_trn, 2] = (100 * ratio + 30).astype(np.uint8)

            rgb[mask_fre] = [40, 120, 200]   # biru: bebas
        else:
            # Statis: 128=unknown, rendah=obstacle, tinggi=bebas
            mask_unk = (arr >= 110) & (arr <= 140)
            mask_occ = arr < 110
            mask_fre = arr > 140

            rgb[mask_unk] = [15,  20,  35]   # gelap: tidak diketahui
            rgb[mask_occ] = [180, 35,  35]   # merah: obstacle

            # Skala bebas: semakin terang = semakin bebas
            fv = arr[mask_fre].astype(np.float32)
            ratio = (fv - 140) / 115.0
            rgb[mask_fre, 0] = (40  * (1 - ratio)).astype(np.uint8)
            rgb[mask_fre, 1] = (100 * ratio + 20).astype(np.uint8)
            rgb[mask_fre, 2] = (200 * ratio + 55).astype(np.uint8)

        return QtGui.QImage(
            rgb.tobytes(), MAP_PIX, MAP_PIX,
            MAP_PIX * 3, QtGui.QImage.Format_RGB888
        )

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(self):
        self.mapbytes = bytearray(MAP_PIX * MAP_PIX)
        self.pose     = (0.0, 0.0, 0.0)
        self.path_px.clear()
        if self.mode == 'slam':
            laser = Laser(SCAN_BINS, SCAN_HZ, 360, 0, 0, 0)
            self._slam = RMHC_SLAM(laser, MAP_PIX, MAP_METERS, random_seed=42)
        else:
            self._grid[:] = 128.0

    def save_png(self, path: str):
        """Simpan peta saat ini sebagai file PNG."""
        img = self.render_map()
        img.save(path)


# ---------------------------------------------------------------------------
# WIDGET PETA
# ---------------------------------------------------------------------------

class MapWidget(QtWidgets.QWidget):
    """
    Tampilan peta occupancy grid dengan overlay:
    - Jalur robot (garis putih semi-transparan)
    - Posisi robot (lingkaran hijau + panah arah)
    - Scan LiDAR saat ini (opsional, toggle)
    """

    def __init__(self, engine: MappingEngine, parent=None):
        super().__init__(parent)
        self.engine       = engine
        self._qimg        = None
        self._show_scan   = True
        self._last_scan   = []           # scan_mm terbaru untuk overlay
        self.setMinimumSize(620, 620)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    def set_scan(self, scan_mm: list):
        self._last_scan = scan_mm

    def toggle_scan(self):
        self._show_scan = not self._show_scan

    def refresh(self):
        """Render ulang peta dan jadwalkan repaint."""
        self._qimg = self.engine.render_map()
        self.update()

    # ── paintEvent ─────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        w, h = self.width(), self.height()

        # ── Gambar peta ────────────────────────────────────────────────────
        if self._qimg and not self._qimg.isNull():
            scaled = self._qimg.scaled(
                w, h,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            ox = (w - scaled.width())  // 2
            oy = (h - scaled.height()) // 2
            painter.drawImage(ox, oy, scaled)
            scale_x = scaled.width()  / MAP_PIX
            scale_y = scaled.height() / MAP_PIX
        else:
            # Belum ada peta
            painter.fillRect(self.rect(), QtGui.QColor(10, 14, 24))
            painter.setPen(QtGui.QColor(60, 60, 80))
            painter.setFont(QtGui.QFont("Courier New", 13))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter,
                             "Menunggu data LiDAR...")
            return
            ox = oy = 0
            scale_x = scale_y = w / MAP_PIX

        # Faktor konversi pixel peta → pixel layar
        def to_screen(px, py):
            return (ox + px * scale_x, oy + py * scale_y)

        # ── Jalur robot ────────────────────────────────────────────────────
        path = self.engine.path_px
        if len(path) >= 2:
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 120), 1.5)
            painter.setPen(pen)
            for i in range(1, len(path)):
                x0, y0 = to_screen(*path[i - 1])
                x1, y1 = to_screen(*path[i])
                painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))

        # ── Overlay sinar scan ─────────────────────────────────────────────
        rpx, rpy, theta = self.engine.get_robot_pixel()
        rx_s, ry_s = to_screen(rpx, rpy)

        if self._show_scan and self._last_scan:
            pen_ray = QtGui.QPen(QtGui.QColor(0, 220, 80, 25), 1)
            painter.setPen(pen_ray)
            scale_mm_to_px = (scale_x / self._mm_per_map_px(scale_x))
            for i, d in enumerate(self._last_scan):
                if d == 0:
                    continue
                ang = math.radians(i)
                dx  = d * math.cos(ang) * scale_mm_to_px
                dy  = -d * math.sin(ang) * scale_mm_to_px
                painter.drawLine(
                    QtCore.QPointF(rx_s, ry_s),
                    QtCore.QPointF(rx_s + dx, ry_s + dy)
                )

        # ── Robot: lingkaran + panah arah ─────────────────────────────────
        R = 10
        # Lingkaran hijau
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 100), 2))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 200, 80, 180)))
        painter.drawEllipse(QtCore.QPointF(rx_s, ry_s), R, R)

        # Panah arah (theta)
        arrow_len = R + 12
        ax = rx_s + arrow_len * math.cos(math.radians(-theta))
        ay = ry_s + arrow_len * math.sin(math.radians(-theta))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 100), 2))
        painter.drawLine(QtCore.QPointF(rx_s, ry_s), QtCore.QPointF(ax, ay))

    def _mm_per_map_px(self, scale_x):
        """Berapa mm per pixel layar (untuk scaling sinar scan)."""
        mm_per_map_px = (MAP_METERS * 1000) / MAP_PIX
        return mm_per_map_px / scale_x if scale_x else 1.0

    def sizeHint(self):
        return QtCore.QSize(700, 700)


# ---------------------------------------------------------------------------
# MAIN WINDOW
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    _STYLE = """
        QMainWindow, QWidget  { background: #080c14; color: #00e060; }
        QLabel                { font-family: 'Courier New'; font-size: 13px;
                                color: #00cc55; padding: 1px 0; }
        QLabel#title          { color: #00ff88; font-size: 16px;
                                font-weight: bold; }
        QLabel#mode           { font-size: 12px; padding: 3px 6px;
                                border-radius: 4px; }
        QLabel#value          { color: #ffffff; font-size: 14px;
                                font-weight: bold; }
        QLabel#warn           { color: #ff5050; font-size: 12px; }
        QPushButton           { background: #0f2a1a; color: #00ff80;
                                border: 1px solid #1a5030; border-radius: 6px;
                                padding: 7px 10px; font-family: 'Courier New';
                                font-size: 12px; }
        QPushButton:hover     { background: #164028; }
        QPushButton:pressed   { background: #0a1a10; }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RPLIDAR 2D Mapping  —  Wasaka Robotic")
        self.setMinimumSize(980, 720)
        self.setStyleSheet(self._STYLE)

        self.engine = MappingEngine()
        self._scan_buf = {}   # angle -> (dist, quality), buffer scan penuh

        self._build_ui()
        self._start_worker()

    # ── Build UI ───────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Peta (kiri)
        self.map_widget = MapWidget(self.engine)
        layout.addWidget(self.map_widget, stretch=4)

        # Panel info (kanan)
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(230)
        pl = QtWidgets.QVBoxLayout(panel)
        pl.setAlignment(QtCore.Qt.AlignTop)
        pl.setSpacing(8)

        def lbl(text, obj=""):
            w = QtWidgets.QLabel(text)
            if obj: w.setObjectName(obj)
            return w

        def sep():
            f = QtWidgets.QFrame()
            f.setFrameShape(QtWidgets.QFrame.HLine)
            f.setStyleSheet("border: 1px solid #0f3322;")
            return f

        # Judul
        pl.addWidget(lbl("RPLIDAR MAPPING", "title"))

        # Badge mode
        mode_text = "SLAM MODE" if SLAM_MODE else "STATIS MODE"
        mode_color = "#003a1a" if SLAM_MODE else "#2a1a00"
        mode_border = "#00cc55" if SLAM_MODE else "#cc8800"
        mode_fg = "#00ff88" if SLAM_MODE else "#ffaa00"
        self.lbl_mode = lbl(f"  {mode_text}  ", "mode")
        self.lbl_mode.setStyleSheet(
            f"background:{mode_color}; color:{mode_fg};"
            f"border:1px solid {mode_border}; border-radius:4px;"
            f"font-size:12px;"
        )
        pl.addWidget(self.lbl_mode)

        if not SLAM_MODE:
            note = lbl("pip install breezyslam\nuntuk SLAM mode", "warn")
            note.setStyleSheet("color:#cc8800; font-size:11px;")
            pl.addWidget(note)

        pl.addWidget(sep())

        # Status & stats
        pl.addWidget(lbl("Status:"))
        self.lbl_status = lbl("Menghubungkan...", "warn")
        pl.addWidget(self.lbl_status)

        pl.addWidget(lbl("Packet Rate:"))
        self.lbl_fps = lbl("-", "value")
        pl.addWidget(self.lbl_fps)

        pl.addWidget(lbl("Titik Scan:"))
        self.lbl_pts = lbl("0", "value")
        pl.addWidget(self.lbl_pts)

        pl.addWidget(sep())

        # Pose robot (hanya SLAM mode)
        pl.addWidget(lbl("Pose Robot:"))
        self.lbl_x     = lbl("X : -", "value")
        self.lbl_y     = lbl("Y : -", "value")
        self.lbl_theta = lbl("θ : -", "value")
        pl.addWidget(self.lbl_x)
        pl.addWidget(self.lbl_y)
        pl.addWidget(self.lbl_theta)

        if not SLAM_MODE:
            for w in [self.lbl_x, self.lbl_y, self.lbl_theta]:
                w.setStyleSheet("color:#444; font-size:12px;")

        pl.addWidget(sep())

        # Info peta
        pl.addWidget(lbl("Konfigurasi Peta:", ""))
        cfg = lbl(
            f"Ukuran : {MAP_PIX} x {MAP_PIX} px\n"
            f"Area   : {MAP_METERS} x {MAP_METERS} m\n"
            f"Res    : {MAP_METERS*100/MAP_PIX:.1f} cm/px"
        )
        cfg.setStyleSheet("color:#337744; font-size:11px; font-family:'Courier New';")
        pl.addWidget(cfg)

        pl.addWidget(sep())

        # Legenda
        pl.addWidget(lbl("Legenda:", ""))
        for txt, color in [
            ("Dinding / Obstacle", "#cc2222"),
            ("Tidak diketahui",    "#2a3045"),
            ("Area bebas",         "#2878c8"),
            ("Jalur robot",        "#ffffff"),
        ]:
            row = QtWidgets.QWidget()
            rl  = QtWidgets.QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(6)
            sq = QtWidgets.QLabel("■")
            sq.setStyleSheet(f"color:{color}; font-size:14px;")
            rl.addWidget(sq)
            rl.addWidget(lbl(txt))
            rl.addStretch()
            pl.addWidget(row)

        pl.addWidget(sep())

        # Tombol kontrol
        self.btn_scan = QtWidgets.QPushButton("👁  Sembunyikan Sinar Scan")
        self.btn_scan.clicked.connect(self._toggle_scan)
        pl.addWidget(self.btn_scan)

        self.btn_save = QtWidgets.QPushButton("💾  Simpan Peta (PNG)")
        self.btn_save.clicked.connect(self._save_map)
        pl.addWidget(self.btn_save)

        self.btn_reset = QtWidgets.QPushButton("🔄  Reset Peta")
        self.btn_reset.clicked.connect(self._reset_map)
        pl.addWidget(self.btn_reset)

        pl.addStretch()

        # Error
        self.lbl_err = lbl("", "warn")
        self.lbl_err.setWordWrap(True)
        pl.addWidget(self.lbl_err)

        layout.addWidget(panel, stretch=1)

    # ── Worker & Timer ─────────────────────────────────────────────────────

    def _start_worker(self):
        self.worker = LidarWorker()
        self.worker.data_sig.connect(self._on_data)
        self.worker.err_sig.connect(self._on_error)
        self.worker.stats_sig.connect(self._on_stats)
        self.worker.start()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(GUI_REFRESH)

    # ── Slot ───────────────────────────────────────────────────────────────

    def _on_data(self, points: list):
        # Akumulasi scan penuh (1 putaran)
        for ang, d, q in points:
            self._scan_buf[round(ang, 1)] = (d, q)
        self.lbl_pts.setText(str(len(self._scan_buf)))

        # Jika scan cukup penuh, update mesin pemetaan
        if len(self._scan_buf) >= 180:
            scan_mm = normalize_scan(
                [(a, d, q) for a, (d, q) in self._scan_buf.items()]
            )
            self.engine.update(scan_mm)
            self.map_widget.set_scan(scan_mm)
            # Update label pose (SLAM mode)
            x, y, t = self.engine.pose
            self.lbl_x.setText(f"X : {x/1000:.2f} m")
            self.lbl_y.setText(f"Y : {y/1000:.2f} m")
            self.lbl_theta.setText(f"θ : {t:.1f}°")

        # Status
        if "Menghubungkan" in self.lbl_status.text():
            self.lbl_status.setText("Terhubung & Berjalan")
            self.lbl_status.setStyleSheet("color:#00ff88; font-size:12px;")

    def _refresh(self):
        self.map_widget.refresh()

    def _on_stats(self, stats: dict):
        self.lbl_fps.setText(f"{stats['fps']} pkt/s")

    def _on_error(self, msg: str):
        self.lbl_status.setText("Error")
        self.lbl_status.setStyleSheet("color:#ff4040; font-size:12px;")
        self.lbl_err.setText(msg)

    def _toggle_scan(self):
        self.map_widget.toggle_scan()
        if self.map_widget._show_scan:
            self.btn_scan.setText("👁  Sembunyikan Sinar Scan")
        else:
            self.btn_scan.setText("👁  Tampilkan Sinar Scan")

    def _save_map(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Simpan Peta", "peta_lidar.png",
            "PNG Image (*.png)"
        )
        if path:
            self.engine.save_png(path)
            QtWidgets.QMessageBox.information(
                self, "Tersimpan", f"Peta disimpan ke:\n{path}"
            )

    def _reset_map(self):
        reply = QtWidgets.QMessageBox.question(
            self, "Reset Peta",
            "Yakin ingin menghapus peta dan jalur robot?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._scan_buf.clear()
            self.engine.reset()
            self.lbl_x.setText("X : -")
            self.lbl_y.setText("Y : -")
            self.lbl_theta.setText("θ : -")

    def closeEvent(self, event):
        self._timer.stop()
        self.worker.stop()
        event.accept()


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window,     QtGui.QColor(8, 12, 20))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(0, 220, 90))
    app.setPalette(pal)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
