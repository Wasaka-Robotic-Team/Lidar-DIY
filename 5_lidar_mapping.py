## =========================================================================
## RPLIDAR 2D Mapping — RViz Block Style | Wasaka Robotic | 2026
## v4 — Coarse Grid + Continuous Ray Tracing = Tampilan Blok RViz
##
## Perubahan v4:
##   - Grid internal 50x50 sel (bukan 600x600 piksel)
##   - Ray tracing kontinu (trace SEMUA sel dilewati sinar)
##   - Upscale tajam (FastTransformation) → tampilan blok kotak RViz
##   - Komputasi di MapUpdateWorker (thread terpisah, tidak freeze)
##
## Install: pip install pyserial numpy PyQt5
## Run    : python3 5_lidar_mapping.py
## =========================================================================

import sys, struct, serial, numpy as np, time, math
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
SERIAL_PORT  = "/dev/ttyTHS1"
BAUD_RATE    = 115200
MIN_DIST_MM  = 100
MAX_DIST_MM  = 8000
MIN_QUALITY  = 1
MAP_METERS   = 16.0          # Luas area peta (meter x meter)
GRID_N       = 50            # Grid internal N x N sel
CELL_MM      = MAP_METERS * 1000 / GRID_N   # mm per sel = 320 mm
DISPLAY_PIX  = 600           # Ukuran tampilan layar (pixel)
SCAN_BINS    = 360
GUI_REFRESH  = 100           # ms
SCAN_HZ      = 13
## =========================================================================


class LidarPacketParser:
    HEADER_A = 0xAA; HEADER_B = 0x55
    PKT_SIZE = 47;   N_PTS    = 12

    def __init__(self):
        self.buf = bytearray()

    def feed(self, raw):
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

    def _decode(self, pkt):
        res = []
        try:
            a0 = struct.unpack_from('<H', pkt, 4)[0] / 100.0
            a1 = struct.unpack_from('<H', pkt, 6)[0] / 100.0
            da = (a1 - a0) % 360
            for i in range(self.N_PTS):
                off = 8 + i * 3
                if off + 2 >= len(pkt): break
                d = struct.unpack_from('<H', pkt, off)[0]
                q = pkt[off + 2]
                ang = (a0 + da * i / max(self.N_PTS - 1, 1)) % 360
                if q >= MIN_QUALITY and MIN_DIST_MM <= d <= MAX_DIST_MM:
                    res.append((ang, d, q))
        except Exception:
            pass
        return res


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
            self.err_sig.emit(f"Gagal membuka {SERIAL_PORT}\n{e}")
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
                self.err_sig.emit(str(e)); break
        try: ser.close()
        except: pass

    def stop(self):
        self.running = False
        self.wait(2000)


def normalize_scan(points):
    total = [0] * SCAN_BINS
    count = [0] * SCAN_BINS
    for ang, d, q in points:
        idx = int(ang % SCAN_BINS)
        total[idx] += d
        count[idx] += 1
    return [int(total[i] / count[i]) if count[i] else 0 for i in range(SCAN_BINS)]


# ---------------------------------------------------------------------------
# MESIN PEMETAAN — Grid Internal GRID_N x GRID_N
# ---------------------------------------------------------------------------

class MappingEngine:
    """
    Grid internal kecil (GRID_N x GRID_N = 50x50 sel).
    Setiap sel = CELL_MM x CELL_MM mm = 320mm x 320mm.

    Ray tracing KONTINU: setiap sel yang dilewati sinar ditandai.
    Ini mengisi ruang bebas sebagai blok solid (bukan titik-titik).

    Render: grid kecil di-upscale ke DISPLAY_PIX dengan FastTransformation
    → tampilan blok kotak ala RViz / Minecraft.
    """

    def __init__(self):
        self.mode    = 'slam' if SLAM_MODE else 'static'
        self.pose    = (0.0, 0.0, 0.0)    # x_mm, y_mm, theta_deg
        self.path_px = []
        self._lock   = QtCore.QMutex()

        if self.mode == 'slam':
            # BreezySlam menggunakan piksel grid sendiri
            self._map_pix  = 400
            self.mapbytes  = bytearray(self._map_pix * self._map_pix)
            laser = Laser(SCAN_BINS, SCAN_HZ, 360, 0, 0, 0)
            self._slam = RMHC_SLAM(laser, self._map_pix, MAP_METERS, random_seed=42)
        else:
            # Grid internal 50x50 float
            self.mapbytes = bytearray(GRID_N * GRID_N)  # tidak dipakai langsung
            self._grid    = np.full((GRID_N, GRID_N), 128.0, dtype=np.float32)

    def update(self, scan_mm):
        if self.mode == 'slam':
            self._update_slam(scan_mm)
        else:
            self._update_static(scan_mm)
        px, py, _ = self.get_robot_pixel()
        locker = QtCore.QMutexLocker(self._lock)
        if not self.path_px or self.path_px[-1] != (px, py):
            self.path_px.append((px, py))
            if len(self.path_px) > 5000:
                self.path_px = self.path_px[-5000:]

    def _update_slam(self, scan_mm):
        self._slam.update(scan_mm)
        x, y, t = self._slam.getpos()
        self.pose = (x, y, t)
        self._slam.getmap(self.mapbytes)

    def _update_static(self, scan_mm):
        """
        Ray tracing kontinu pada grid GRID_N x GRID_N.

        Untuk setiap sinar yang valid:
          1. Hitung berapa sel yang dilewati: n = dist_mm / CELL_MM
          2. Tandai SEMUA sel dari pusat hingga n-1 sebagai FREE (+35)
          3. Tandai sel ke-n sebagai OBSTACLE (-80)

        Hasilnya: jalur sinar = blok putih solid, ujung = blok hitam.
        """
        cx = cy = GRID_N // 2
        cell = CELL_MM

        angles = np.deg2rad(np.arange(SCAN_BINS))
        dists  = np.array(scan_mm, dtype=np.float32)
        valid  = dists > 0

        if not np.any(valid):
            return

        v_ang  = angles[valid]
        v_dist = dists[valid]
        cos_a  = np.cos(v_ang)
        sin_a  = -np.sin(v_ang)   # Y dibalik (image coords)

        locker = QtCore.QMutexLocker(self._lock)

        for i in range(len(v_ang)):
            d  = float(v_dist[i])
            ca = float(cos_a[i])
            sa = float(sin_a[i])

            # Jumlah sel yang dilewati sinar ini
            n_cells = min(int(d / cell), GRID_N - 1)

            if n_cells > 1:
                # Tandai SEMUA sel bebas di sepanjang sinar (continuous!)
                ts  = np.arange(1, n_cells)
                fxs = np.clip((cx + ts * ca).astype(np.int32), 0, GRID_N - 1)
                fys = np.clip((cy + ts * sa).astype(np.int32), 0, GRID_N - 1)
                self._grid[fys, fxs] = np.minimum(255.0,
                                                   self._grid[fys, fxs] + 35.0)

            # Tandai sel obstacle di ujung sinar
            ox = int(np.clip(cx + n_cells * ca, 0, GRID_N - 1))
            oy = int(np.clip(cy + n_cells * sa, 0, GRID_N - 1))
            self._grid[oy, ox] = max(0.0, self._grid[oy, ox] - 80.0)

        # Sinkronkan ke mapbytes (GRID_N x GRID_N)
        arr = np.clip(self._grid, 0, 255).astype(np.uint8)
        self.mapbytes[:] = arr.tobytes()

    def get_robot_pixel(self):
        """
        Kembalikan posisi robot dalam koordinat GRID_N x GRID_N.
        """
        if self.mode == 'slam':
            x, y, t = self.pose
            scale = self._map_pix / (MAP_METERS * 1000)
            px = int(self._map_pix / 2 + x * scale)
            py = int(self._map_pix / 2 - y * scale)
            N  = self._map_pix
        else:
            px = py = GRID_N // 2
            t  = 0.0
            N  = GRID_N
        return (max(0, min(N-1, px)), max(0, min(N-1, py)), t)

    def render_map(self):
        """
        Render grid internal → QImage GRID_N x GRID_N dengan warna RViz,
        lalu upscale ke DISPLAY_PIX x DISPLAY_PIX dengan FastTransformation
        → blok kotak tajam (tidak blur) ala tampilan referensi.
        """
        locker = QtCore.QMutexLocker(self._lock)

        if self.mode == 'slam':
            N   = self._map_pix
            arr = np.frombuffer(bytes(self.mapbytes), dtype=np.uint8).reshape(N, N)
            rgb = np.full((N, N, 3), 127, dtype=np.uint8)
            rgb[(arr > 0) & (arr < 110)] = [0, 0, 0]       # Hitam: obstacle
            rgb[arr >= 110]              = [236, 236, 236]  # Putih: free
        else:
            arr = np.clip(self._grid.copy(), 0, 255).astype(np.uint8)
            rgb = np.full((GRID_N, GRID_N, 3), 127, dtype=np.uint8)
            rgb[arr < 100]  = [0, 0, 0]        # Hitam: obstacle
            rgb[arr > 143]  = [236, 236, 236]  # Putih: free
            N = GRID_N

        # Buat QImage kecil
        img_small = QtGui.QImage(
            rgb.tobytes(), N, N, N * 3, QtGui.QImage.Format_RGB888
        )

        # Upscale TANPA interpolasi → blok kotak tajam
        return img_small.scaled(
            DISPLAY_PIX, DISPLAY_PIX,
            QtCore.Qt.IgnoreAspectRatio,
            QtCore.Qt.FastTransformation   # ← Kunci tampilan blok!
        )

    def reset(self):
        locker = QtCore.QMutexLocker(self._lock)
        self.pose = (0.0, 0.0, 0.0)
        self.path_px.clear()
        if self.mode == 'slam':
            self.mapbytes = bytearray(self._map_pix * self._map_pix)
            laser = Laser(SCAN_BINS, SCAN_HZ, 360, 0, 0, 0)
            self._slam = RMHC_SLAM(laser, self._map_pix, MAP_METERS, random_seed=42)
        else:
            self._grid[:] = 128.0
            self.mapbytes = bytearray(GRID_N * GRID_N)

    def save_png(self, path):
        self.render_map().save(path)


# ---------------------------------------------------------------------------
# MAP UPDATE WORKER — thread terpisah
# ---------------------------------------------------------------------------

class MapUpdateWorker(QtCore.QThread):
    map_updated = QtCore.pyqtSignal()

    def __init__(self, engine):
        super().__init__()
        self.engine  = engine
        self._queue  = []
        self._qmutex = QtCore.QMutex()
        self._cond   = QtCore.QWaitCondition()
        self.running = False

    def enqueue(self, scan_mm):
        locker = QtCore.QMutexLocker(self._qmutex)
        self._queue = [scan_mm]   # Hanya simpan scan terbaru
        self._cond.wakeOne()

    def run(self):
        self.running = True
        while self.running:
            self._qmutex.lock()
            if not self._queue:
                self._cond.wait(self._qmutex, 200)
            scan = self._queue.pop(0) if self._queue else None
            self._qmutex.unlock()
            if scan is not None:
                self.engine.update(scan)
                self.map_updated.emit()

    def stop(self):
        self.running = False
        self._cond.wakeAll()
        self.wait(2000)


# ---------------------------------------------------------------------------
# WIDGET PETA
# ---------------------------------------------------------------------------

class MapWidget(QtWidgets.QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine     = engine
        self._qimg      = None
        self._show_scan = False
        self._is_iso    = True     # Default: Isometric seperti referensi
        self._last_scan = []
        self.setMinimumSize(580, 580)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    def set_scan(self, s): self._last_scan = s
    def toggle_scan(self): self._show_scan = not self._show_scan
    def toggle_view(self):
        self._is_iso = not self._is_iso
        self.update()

    def refresh(self):
        self._qimg = self.engine.render_map()
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QtGui.QColor(44, 54, 63))

        if not self._qimg or self._qimg.isNull():
            painter.setPen(QtGui.QColor(200, 200, 200))
            painter.setFont(QtGui.QFont("Courier New", 13))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter,
                             "Menunggu data LiDAR...")
            return

        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        painter.save()

        if self._is_iso:
            painter.translate(w / 2.0, h / 2.0 + 10)
            painter.scale(1.0, 0.58)
            painter.rotate(-45)
            painter.translate(-w / 2.0, -h / 2.0)

        # Gambar peta (sudah dalam DISPLAY_PIX x DISPLAY_PIX)
        mg  = 0.05
        tr  = QtCore.QRectF(w*mg, h*mg, w*(1-2*mg), h*(1-2*mg))
        painter.drawImage(tr, self._qimg)

        sx = tr.width()  / DISPLAY_PIX
        sy = tr.height() / DISPLAY_PIX
        ox, oy = tr.left(), tr.top()

        # Faktor konversi grid-sel → pixel layar
        # 1 sel = (DISPLAY_PIX/GRID_N) pixel display → lalu sx ke layar
        def grid_to_screen(gx, gy):
            dpx = (gx + 0.5) * (DISPLAY_PIX / GRID_N)
            dpy = (gy + 0.5) * (DISPLAY_PIX / GRID_N)
            return (ox + dpx * sx, oy + dpy * sy)

        # Jalur robot
        path = self.engine.path_px
        if len(path) >= 2:
            N = self.engine._map_pix if self.engine.mode == 'slam' else GRID_N
            scale_d = DISPLAY_PIX / N
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 60, 60, 220), 2))
            for i in range(1, len(path)):
                x0s = ox + (path[i-1][0] + 0.5) * scale_d * sx
                y0s = oy + (path[i-1][1] + 0.5) * scale_d * sy
                x1s = ox + (path[i][0]   + 0.5) * scale_d * sx
                y1s = oy + (path[i][1]   + 0.5) * scale_d * sy
                painter.drawLine(QtCore.QPointF(x0s, y0s),
                                 QtCore.QPointF(x1s, y1s))

        # Posisi robot
        rpx, rpy, theta = self.engine.get_robot_pixel()
        N = self.engine._map_pix if self.engine.mode == 'slam' else GRID_N
        scale_d = DISPLAY_PIX / N
        rx_s = ox + (rpx + 0.5) * scale_d * sx
        ry_s = oy + (rpy + 0.5) * scale_d * sy

        # Sinar scan (setiap 15 derajat, sangat hemat)
        if self._show_scan and self._last_scan:
            mm_to_dpx = (DISPLAY_PIX / (MAP_METERS * 1000)) * sx
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 160, 0, 100), 1))
            for i in range(0, len(self._last_scan), 15):
                d = self._last_scan[i]
                if d == 0: continue
                ang = math.radians(i)
                painter.drawLine(
                    QtCore.QPointF(rx_s, ry_s),
                    QtCore.QPointF(rx_s + d*math.cos(ang)*mm_to_dpx,
                                   ry_s - d*math.sin(ang)*mm_to_dpx)
                )

        # Robot icon: sasis metalik + LiDAR puck orange
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.save()
        painter.translate(rx_s, ry_s)
        painter.rotate(-theta)

        # Sasis
        painter.setPen(QtGui.QPen(QtGui.QColor(15, 15, 15), 2))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(55, 65, 75)))
        painter.drawRoundedRect(QtCore.QRectF(-14, -14, 28, 28), 3, 3)

        # Panah depan
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 200, 0)))
        painter.drawPolygon(QtGui.QPolygonF([
            QtCore.QPointF(14, 0), QtCore.QPointF(7, -6), QtCore.QPointF(7, 6)
        ]))

        # LiDAR orange puck
        painter.setPen(QtGui.QPen(QtGui.QColor(160, 45, 0), 1.5))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 95, 0)))
        painter.drawEllipse(QtCore.QPointF(0, 0), 8, 8)

        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 220, 0)))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(QtCore.QPointF(3, 0), 3, 3)

        painter.restore()
        painter.restore()

    def sizeHint(self):
        return QtCore.QSize(660, 660)


# ---------------------------------------------------------------------------
# MAIN WINDOW
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    _STYLE = """
        QMainWindow, QWidget { background:#1c2430; color:#dde1e7; }
        QLabel               { font-family:'Courier New'; font-size:13px; color:#c8d0da; padding:1px 0; }
        QLabel#title         { color:#ffb347; font-size:15px; font-weight:bold; }
        QLabel#value         { color:#ffffff; font-size:14px; font-weight:bold; }
        QLabel#warn          { color:#ff6b6b; font-size:12px; }
        QPushButton          { background:#28333f; color:#e8ecf0;
                               border:1px solid #3d5060; border-radius:6px;
                               padding:7px 10px; font-family:'Courier New'; font-size:12px; }
        QPushButton:hover    { background:#354555; border-color:#ffb347; }
        QPushButton:pressed  { background:#1a2530; }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RPLIDAR RViz Block Mapping — Wasaka Robotic")
        self.setMinimumSize(960, 700)
        self.setStyleSheet(self._STYLE)
        self.engine    = MappingEngine()
        self._scan_buf = {}
        self._build_ui()
        self._start_workers()

    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        hl = QtWidgets.QHBoxLayout(root)
        hl.setContentsMargins(10, 10, 10, 10)
        hl.setSpacing(10)

        self.map_widget = MapWidget(self.engine)
        hl.addWidget(self.map_widget, stretch=4)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(230)
        pl = QtWidgets.QVBoxLayout(panel)
        pl.setAlignment(QtCore.Qt.AlignTop)
        pl.setSpacing(7)

        def lbl(t, obj=""):
            w = QtWidgets.QLabel(t)
            if obj: w.setObjectName(obj)
            return w

        def sep():
            f = QtWidgets.QFrame()
            f.setFrameShape(QtWidgets.QFrame.HLine)
            f.setStyleSheet("border:1px solid #2e3d4d;")
            return f

        pl.addWidget(lbl("RVIZ BLOCK MAPPER", "title"))

        mode_text = "SLAM MODE"  if SLAM_MODE else "STATIS MODE"
        mode_fg   = "#51cf66"    if SLAM_MODE else "#ffd43b"
        mode_bg   = "#1b4332"    if SLAM_MODE else "#3d2c00"
        mode_brd  = "#40c057"    if SLAM_MODE else "#e67700"
        ml = QtWidgets.QLabel(f"  {mode_text}  ")
        ml.setStyleSheet(f"color:{mode_fg};background:{mode_bg};"
                         f"border:1px solid {mode_brd};border-radius:4px;"
                         f"padding:3px 6px;font-size:12px;font-family:'Courier New';")
        pl.addWidget(ml)
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

        pl.addWidget(lbl("Pose Robot (x, y, θ):"))
        self.lbl_x     = lbl("X : -", "value")
        self.lbl_y     = lbl("Y : -", "value")
        self.lbl_theta = lbl("θ : -", "value")
        pl.addWidget(self.lbl_x)
        pl.addWidget(self.lbl_y)
        pl.addWidget(self.lbl_theta)
        pl.addWidget(sep())

        pl.addWidget(lbl("Legenda RViz:"))
        for txt, col, extra in [
            ("Wall / Obstacle",  "#111111", "border:1px solid #666;"),
            ("Unknown Area",     "#7f8c8d", ""),
            ("Free Space",       "#ececec", "border:1px solid #aaa;"),
            ("Robot LiDAR Puck", "#ff6000", ""),
        ]:
            row = QtWidgets.QWidget()
            rl  = QtWidgets.QHBoxLayout(row)
            rl.setContentsMargins(0,0,0,0); rl.setSpacing(6)
            sq = QtWidgets.QLabel("■")
            sq.setStyleSheet(f"color:{col};font-size:15px;{extra}")
            rl.addWidget(sq); rl.addWidget(lbl(txt)); rl.addStretch()
            pl.addWidget(row)
        pl.addWidget(sep())

        cfg = QtWidgets.QLabel(
            f"Grid : {GRID_N}×{GRID_N} sel\n"
            f"Area : {MAP_METERS}×{MAP_METERS} m\n"
            f"Res  : {CELL_MM/10:.0f} cm/sel"
        )
        cfg.setStyleSheet("color:#4a6070;font-size:11px;font-family:'Courier New';")
        pl.addWidget(cfg)
        pl.addWidget(sep())

        self.btn_view = QtWidgets.QPushButton("🔄  2D / 2.5D Iso View")
        self.btn_view.clicked.connect(self.map_widget.toggle_view)
        pl.addWidget(self.btn_view)

        self.btn_scan = QtWidgets.QPushButton("👁  Tampilkan Sinar Scan")
        self.btn_scan.clicked.connect(self._toggle_scan)
        pl.addWidget(self.btn_scan)

        self.btn_save = QtWidgets.QPushButton("💾  Simpan Peta (PNG)")
        self.btn_save.clicked.connect(self._save_map)
        pl.addWidget(self.btn_save)

        self.btn_reset = QtWidgets.QPushButton("🗑  Reset Peta")
        self.btn_reset.clicked.connect(self._reset_map)
        pl.addWidget(self.btn_reset)

        pl.addStretch()
        self.lbl_err = lbl("", "warn")
        self.lbl_err.setWordWrap(True)
        pl.addWidget(self.lbl_err)
        hl.addWidget(panel, stretch=1)

    def _start_workers(self):
        self.map_worker = MapUpdateWorker(self.engine)
        self.map_worker.map_updated.connect(self._on_map_updated)
        self.map_worker.start()

        self.lidar_worker = LidarWorker()
        self.lidar_worker.data_sig.connect(self._on_data)
        self.lidar_worker.err_sig.connect(self._on_error)
        self.lidar_worker.stats_sig.connect(self._on_stats)
        self.lidar_worker.start()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.map_widget.refresh)
        self._timer.start(GUI_REFRESH)

    def _on_data(self, points):
        for ang, d, q in points:
            self._scan_buf[round(ang, 1)] = (d, q)
        self.lbl_pts.setText(str(len(self._scan_buf)))
        if len(self._scan_buf) >= 180:
            scan_mm = normalize_scan(
                [(a, d, q) for a, (d, q) in self._scan_buf.items()]
            )
            self.map_worker.enqueue(scan_mm)
            self.map_widget.set_scan(scan_mm)
        if "Menghubungkan" in self.lbl_status.text():
            self.lbl_status.setText("Terhubung & Berjalan")
            self.lbl_status.setStyleSheet("color:#51cf66;font-size:12px;")

    def _on_map_updated(self):
        x, y, t = self.engine.pose
        self.lbl_x.setText(f"X : {x/1000:.2f} m")
        self.lbl_y.setText(f"Y : {y/1000:.2f} m")
        self.lbl_theta.setText(f"θ : {t:.1f}°")

    def _on_stats(self, s):
        self.lbl_fps.setText(f"{s['fps']} pkt/s")

    def _on_error(self, msg):
        self.lbl_status.setText("Error")
        self.lbl_status.setStyleSheet("color:#ff6b6b;font-size:12px;")
        self.lbl_err.setText(msg)

    def _toggle_scan(self):
        self.map_widget.toggle_scan()
        self.btn_scan.setText(
            "👁  Sembunyikan Sinar Scan" if self.map_widget._show_scan
            else "👁  Tampilkan Sinar Scan"
        )

    def _save_map(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Simpan Peta", "peta_rviz_block.png", "PNG Image (*.png)"
        )
        if path:
            self.engine.save_png(path)
            QtWidgets.QMessageBox.information(self, "Tersimpan", f"Peta:\n{path}")

    def _reset_map(self):
        if QtWidgets.QMessageBox.question(
            self, "Reset", "Hapus peta?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        ) == QtWidgets.QMessageBox.Yes:
            self._scan_buf.clear()
            self.engine.reset()
            self.lbl_x.setText("X : -")
            self.lbl_y.setText("Y : -")
            self.lbl_theta.setText("θ : -")

    def closeEvent(self, event):
        self._timer.stop()
        self.map_worker.stop()
        self.lidar_worker.stop()
        event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window,     QtGui.QColor(28, 36, 48))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(220, 225, 235))
    app.setPalette(pal)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
