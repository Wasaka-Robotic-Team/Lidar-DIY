## =========================================================================
## RPLIDAR 2D Mapping — RViz / ROS Style Occupancy Grid Display
## Wasaka Robotic Team | 2026
##
## Fitur RViz Style:
##   - Skema warna ROS RViz: Black (Wall), White (Free), Grey (Unknown)
##   - Grid plane 1-meter ala RViz
##   - Ikon Robot 3D / Isometric (Sasis metalik + Lidar Puck Orange)
##   - Toggle Tampilan: 2D Top-Down vs 2.5D Isometric Tilt View
##   - Mode SLAM (BreezySlam) & Statis
##
## Install dependencies:
##   pip install pyserial numpy PyQt5
##   pip install breezyslam          # opsional, aktifkan SLAM mode
##
## Jalankan:
##   python3 5_lidar_mapping.py
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
GUI_REFRESH  = 80                # Interval refresh GUI (ms) — jangan terlalu kecil di Jetson
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
# WORKER THREAD
# ---------------------------------------------------------------------------

class LidarWorker(QtCore.QThread):
    data_sig  = QtCore.pyqtSignal(list)
    err_sig   = QtCore.pyqtSignal(str)
    stats_sig = QtCore.pyqtSignal(dict)

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


def normalize_scan(points: list) -> list:
    total = [0] * SCAN_BINS
    count = [0] * SCAN_BINS
    for ang, d, q in points:
        idx = int(ang % SCAN_BINS)
        total[idx] += d
        count[idx] += 1
    return [int(total[i] / count[i]) if count[i] else 0 for i in range(SCAN_BINS)]


# ---------------------------------------------------------------------------
# MESIN PEMETAAN (RVIZ OCCUPANCY GRID STYLING)
# ---------------------------------------------------------------------------

class MappingEngine:
    def __init__(self):
        self.mode     = 'slam' if SLAM_MODE else 'static'
        self.mapbytes = bytearray(MAP_PIX * MAP_PIX)
        self.pose     = (0.0, 0.0, 0.0)   # x_mm, y_mm, theta_deg
        self.path_px  = []

        if self.mode == 'slam':
            laser = Laser(
                scan_size=SCAN_BINS, scan_rate_hz=SCAN_HZ,
                detection_angle_degrees=360, distance_no_detection_mm=0,
                detection_margin=0, offset_mm=0
            )
            self._slam = RMHC_SLAM(laser, MAP_PIX, MAP_METERS, random_seed=42)
        else:
            self._grid   = np.full((MAP_PIX, MAP_PIX), 128.0, dtype=np.float32)
            self._res_mm = (MAP_METERS * 1000) / MAP_PIX

    def update(self, scan_mm: list):
        if self.mode == 'slam':
            self._update_slam(scan_mm)
        else:
            self._update_static(scan_mm)
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
        Raycasting dengan numpy (100x lebih cepat dari Python loop).
        Menggunakan Bresenham-style vectorized line drawing via linspace.
        """
        cx = cy = MAP_PIX // 2
        res = self._res_mm
        angles = np.deg2rad(np.arange(SCAN_BINS))
        dists  = np.array(scan_mm, dtype=np.float32)

        # Hanya proses sinar yang valid
        valid = dists > 0
        if not np.any(valid):
            return

        v_ang  = angles[valid]
        v_dist = dists[valid]
        cos_a  = np.cos(v_ang)
        sin_a  = -np.sin(v_ang)   # Y dibalik (image coords)

        n_steps = 8   # sampling setiap N mm sepanjang sinar (trade-off kecepatan vs akurasi)

        for idx in range(len(v_ang)):
            d     = v_dist[idx]
            ca    = cos_a[idx]
            sa    = sin_a[idx]
            n_pts = max(int(d / (res * n_steps)), 1)

            # Titik bebas sepanjang sinar (vectorized)
            steps = np.arange(1, n_pts)
            fxs   = np.clip((cx + steps * n_steps * ca).astype(np.int32), 0, MAP_PIX - 1)
            fys   = np.clip((cy + steps * n_steps * sa).astype(np.int32), 0, MAP_PIX - 1)
            self._grid[fys, fxs] = np.minimum(255.0, self._grid[fys, fxs] + 3.0)

            # Titik obstacle di ujung sinar
            ox = int(np.clip(cx + (d / res) * ca, 0, MAP_PIX - 1))
            oy = int(np.clip(cy + (d / res) * sa, 0, MAP_PIX - 1))
            self._grid[oy, ox] = max(0.0, self._grid[oy, ox] - 40.0)

        arr = np.clip(self._grid, 0, 255).astype(np.uint8)
        self.mapbytes[:] = arr.tobytes()

    def get_robot_pixel(self):
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

    def render_map(self) -> QtGui.QImage:
        """
        Gaya Warna Asli ROS / RViz:
          - Hitam (0, 0, 0)         = Wall / Obstacle (Occupied)
          - Putih (245, 245, 245)   = Free Space
          - Abu-abu (127, 140, 141) = Unknown Space
        """
        arr = np.frombuffer(self.mapbytes, dtype=np.uint8).reshape(MAP_PIX, MAP_PIX)
        rgb = np.zeros((MAP_PIX, MAP_PIX, 3), dtype=np.uint8)

        if self.mode == 'slam':
            mask_unk = arr == 0
            mask_occ = (arr > 0) & (arr < 110)
            mask_fre = arr >= 110

            rgb[mask_unk] = [127, 140, 141]  # RViz Neutral Grey (Unknown)
            rgb[mask_occ] = [0,   0,   0]    # RViz Solid Black (Obstacle)
            rgb[mask_fre] = [245, 245, 245]  # RViz Off-White (Free)
        else:
            mask_unk = (arr >= 115) & (arr <= 140)
            mask_occ = arr < 115
            mask_fre = arr > 140

            rgb[mask_unk] = [127, 140, 141]  # RViz Neutral Grey (Unknown)
            rgb[mask_occ] = [0,   0,   0]    # RViz Solid Black (Obstacle)
            rgb[mask_fre] = [245, 245, 245]  # RViz Off-White (Free)

        return QtGui.QImage(
            rgb.tobytes(), MAP_PIX, MAP_PIX,
            MAP_PIX * 3, QtGui.QImage.Format_RGB888
        )

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
        img = self.render_map()
        img.save(path)


# ---------------------------------------------------------------------------
# WIDGET PETA STYLE RVIZ 2D / 2.5D
# ---------------------------------------------------------------------------

class MapWidget(QtWidgets.QWidget):
    def __init__(self, engine: MappingEngine, parent=None):
        super().__init__(parent)
        self.engine     = engine
        self._qimg      = None
        self._show_scan = True
        self._is_iso    = False     # Default: 2D Top-Down (lebih ringan). Toggle ke Iso via tombol
        self._last_scan = []
        self.setMinimumSize(620, 620)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    def set_scan(self, scan_mm: list):
        self._last_scan = scan_mm

    def toggle_scan(self):
        self._show_scan = not self._show_scan

    def toggle_view(self):
        self._is_iso = not self._is_iso
        self.update()

    def refresh(self):
        self._qimg = self.engine.render_map()
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QtGui.QColor(45, 55, 65)) # RViz Dark Background

        if not self._qimg or self._qimg.isNull():
            painter.setPen(QtGui.QColor(200, 200, 200))
            painter.setFont(QtGui.QFont("Courier New", 13))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "Menunggu data LiDAR...")
            return

        painter.save()

        # Transformasi Isometrik / 2.5D ala RViz
        if self._is_iso:
            painter.translate(w / 2.0, h / 2.0 + 30)
            painter.scale(1.0, 0.65)           # Kemiringan perspektif 3D
            painter.rotate(-45)                # Rotasi sudut pandang 3D
            painter.translate(-w / 2.0, -h / 2.0)

        # Draw Base Map
        target_rect = QtCore.QRectF(w*0.08, h*0.08, w*0.84, h*0.84)
        painter.drawImage(target_rect, self._qimg)

        scale_x = target_rect.width() / MAP_PIX
        scale_y = target_rect.height() / MAP_PIX
        ox, oy  = target_rect.left(), target_rect.top()

        def to_screen(px, py):
            return (ox + px * scale_x, oy + py * scale_y)

        # Draw RViz 1-Meter Grid Lines
        painter.setPen(QtGui.QPen(QtGui.QColor(100, 110, 120, 90), 1, QtCore.Qt.DashLine))
        grid_step_px = (1.0 / MAP_METERS) * MAP_PIX
        for i in range(1, int(MAP_METERS)):
            gx, gy = to_screen(i * grid_step_px, i * grid_step_px)
            painter.drawLine(QtCore.QPointF(gx, target_rect.top()), QtCore.QPointF(gx, target_rect.bottom()))
            painter.drawLine(QtCore.QPointF(target_rect.left(), gy), QtCore.QPointF(target_rect.right(), gy))

        # Jalur Robot
        path = self.engine.path_px
        if len(path) >= 2:
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 60, 60, 200), 2))
            for i in range(1, len(path)):
                x0, y0 = to_screen(*path[i - 1])
                x1, y1 = to_screen(*path[i])
                painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))

        # Scan Rays
        rpx, rpy, theta = self.engine.get_robot_pixel()
        rx_s, ry_s = to_screen(rpx, rpy)

        if self._show_scan and self._last_scan:
            # Hanya gambar setiap RAY_STEP-th ray untuk hemat render time
            RAY_STEP = 8
            pen_ray  = QtGui.QPen(QtGui.QColor(255, 160, 0, 90), 1)
            painter.setPen(pen_ray)
            mm_to_px = (scale_x * MAP_PIX) / (MAP_METERS * 1000.0)
            for i in range(0, len(self._last_scan), RAY_STEP):
                d = self._last_scan[i]
                if d == 0: continue
                ang = math.radians(i)
                dx  = d * math.cos(ang) * mm_to_px
                dy  = -d * math.sin(ang) * mm_to_px
                painter.drawLine(QtCore.QPointF(rx_s, ry_s), QtCore.QPointF(rx_s + dx, ry_s + dy))

        # Robot Icon 3D / Isometric (Sasis Metalik + Lidar Orange Puck)
        painter.save()
        painter.translate(rx_s, ry_s)
        painter.rotate(-theta)

        # Sasis Utama (Dark Metal Box)
        painter.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20), 2))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(60, 70, 80)))
        painter.drawRoundedRect(QtCore.QRectF(-16, -16, 32, 32), 4, 4)

        # Panah Depan (Wedge)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 200, 0)))
        polygon = QtGui.QPolygonF([
            QtCore.QPointF(16, 0),
            QtCore.QPointF(8, -7),
            QtCore.QPointF(8, 7)
        ])
        painter.drawPolygon(polygon)

        # LiDAR Cylinder (Orange Puck khas RPLIDAR)
        painter.setPen(QtGui.QPen(QtGui.QColor(180, 50, 0), 1.5))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 100, 0))) # Bright Orange
        painter.drawEllipse(QtCore.QPointF(0, 0), 9, 9)

        # Lensa Lidar (Kuning)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 230, 0)))
        painter.drawEllipse(QtCore.QPointF(3, 0), 3, 3)

        painter.restore()
        painter.restore()

    def sizeHint(self):
        return QtCore.QSize(700, 700)


# ---------------------------------------------------------------------------
# MAIN WINDOW
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    _STYLE = """
        QMainWindow, QWidget  { background: #1e252d; color: #e0e0e0; }
        QLabel                { font-family: 'Courier New'; font-size: 13px; color: #d0d0d0; }
        QLabel#title          { color: #ffaa00; font-size: 16px; font-weight: bold; }
        QLabel#mode           { font-size: 12px; padding: 3px 6px; border-radius: 4px; }
        QLabel#value          { color: #ffffff; font-size: 14px; font-weight: bold; }
        QLabel#warn           { color: #ff5050; font-size: 12px; }
        QPushButton           { background: #2c3844; color: #ffffff;
                                border: 1px solid #4a5a6a; border-radius: 6px;
                                padding: 7px 10px; font-family: 'Courier New'; font-size: 12px; }
        QPushButton:hover     { background: #3d4d5c; border-color: #ffaa00; }
        QPushButton:pressed   { background: #1c2530; }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RPLIDAR ROS RViz Style Mapping — Wasaka Robotic")
        self.setMinimumSize(1000, 740)
        self.setStyleSheet(self._STYLE)

        self.engine = MappingEngine()
        self._scan_buf = {}

        self._build_ui()
        self._start_worker()

    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.map_widget = MapWidget(self.engine)
        layout.addWidget(self.map_widget, stretch=4)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(240)
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
            f.setStyleSheet("border: 1px solid #3a4a5a;")
            return f

        pl.addWidget(lbl("RVIZ GRID MAPPER", "title"))

        mode_text = "SLAM MODE" if SLAM_MODE else "STATIS MODE"
        mode_color = "#1b4332" if SLAM_MODE else "#4a2c00"
        mode_border = "#40c057" if SLAM_MODE else "#ff922b"
        mode_fg = "#51cf66" if SLAM_MODE else "#ffd43b"
        self.lbl_mode = lbl(f"  {mode_text}  ", "mode")
        self.lbl_mode.setStyleSheet(
            f"background:{mode_color}; color:{mode_fg};"
            f"border:1px solid {mode_border}; border-radius:4px; font-size:12px;"
        )
        pl.addWidget(self.lbl_mode)
        pl.addWidget(sep())

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

        pl.addWidget(lbl("Pose Robot (x,y,θ):"))
        self.lbl_x     = lbl("X : -", "value")
        self.lbl_y     = lbl("Y : -", "value")
        self.lbl_theta = lbl("θ : -", "value")
        pl.addWidget(self.lbl_x)
        pl.addWidget(self.lbl_y)
        pl.addWidget(self.lbl_theta)

        pl.addWidget(sep())

        pl.addWidget(lbl("Legenda RViz Style:", ""))
        for txt, color in [
            ("Wall / Obstacle",   "#000000"),
            ("Unknown Area",      "#7f8c8d"),
            ("Free Space",        "#f5f5f5"),
            ("Robot LiDAR Puck",  "#ff6600"),
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

        self.btn_view = QtWidgets.QPushButton("🔄 Toggle 2D / 2.5D 3D View")
        self.btn_view.clicked.connect(self.map_widget.toggle_view)
        pl.addWidget(self.btn_view)

        self.btn_scan = QtWidgets.QPushButton("👁 Toggle Sinar Scan")
        self.btn_scan.clicked.connect(self.map_widget.toggle_scan)
        pl.addWidget(self.btn_scan)

        self.btn_save = QtWidgets.QPushButton("💾 Simpan Peta (PNG)")
        self.btn_save.clicked.connect(self._save_map)
        pl.addWidget(self.btn_save)

        self.btn_reset = QtWidgets.QPushButton("🗑 Reset Peta")
        self.btn_reset.clicked.connect(self._reset_map)
        pl.addWidget(self.btn_reset)

        pl.addStretch()

        self.lbl_err = lbl("", "warn")
        self.lbl_err.setWordWrap(True)
        pl.addWidget(self.lbl_err)

        layout.addWidget(panel, stretch=1)

    def _start_worker(self):
        self.worker = LidarWorker()
        self.worker.data_sig.connect(self._on_data)
        self.worker.err_sig.connect(self._on_error)
        self.worker.stats_sig.connect(self._on_stats)
        self.worker.start()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(GUI_REFRESH)

    def _on_data(self, points: list):
        for ang, d, q in points:
            self._scan_buf[round(ang, 1)] = (d, q)
        self.lbl_pts.setText(str(len(self._scan_buf)))

        if len(self._scan_buf) >= 180:
            scan_mm = normalize_scan(
                [(a, d, q) for a, (d, q) in self._scan_buf.items()]
            )
            self.engine.update(scan_mm)
            self.map_widget.set_scan(scan_mm)
            x, y, t = self.engine.pose
            self.lbl_x.setText(f"X : {x/1000:.2f} m")
            self.lbl_y.setText(f"Y : {y/1000:.2f} m")
            self.lbl_theta.setText(f"θ : {t:.1f}°")

        if "Menghubungkan" in self.lbl_status.text():
            self.lbl_status.setText("Terhubung & Berjalan")
            self.lbl_status.setStyleSheet("color:#51cf66; font-size:12px;")

    def _refresh(self):
        self.map_widget.refresh()

    def _on_stats(self, stats: dict):
        self.lbl_fps.setText(f"{stats['fps']} pkt/s")

    def _on_error(self, msg: str):
        self.lbl_status.setText("Error")
        self.lbl_status.setStyleSheet("color:#ff6b6b; font-size:12px;")
        self.lbl_err.setText(msg)

    def _save_map(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Simpan Peta RViz", "peta_rviz_lidar.png",
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
            "Yakin ingin menghapus peta?",
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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

