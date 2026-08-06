## =========================================================================
## RPLIDAR Orange/Red 4000pt/s 8M — Complete Visualizer
## Wasaka Robotic Team
##
## Fitur:
##   - Radar polar 360° dengan color-coding berdasarkan jarak
##   - Filter noise otomatis (jarak & kualitas sinyal)
##   - Highlight objek terdekat secara real-time
##   - Panel info: jarak terdekat, sudut, packet rate
##   - Full-scan buffer (setiap sudut diperbarui, bukan ditimpa)
##
## Kompatibel: Ubuntu / Linux (Jetson) — Tidak disarankan Windows
##
## Install dependencies terlebih dahulu:
##   pip install pyserial numpy PyQt5
##
## Jalankan:
##   python3 lidar_visualizer.py
##
## Wiring LiDAR:
##   Merah  -> 5V
##   Kuning -> GND
##   Hitam  -> TX (masuk ke RX Jetson)
##   Hijau  -> No Connection
## =========================================================================

import sys
import struct
import serial
import numpy as np
import time
from PyQt5 import QtWidgets, QtCore, QtGui

## =========================================================================
## KONFIGURASI — Sesuaikan sebelum menjalankan
## =========================================================================
SERIAL_PORT  = "/dev/ttyUSB0"  # Ganti: /dev/ttyTHS1 jika via GPIO UART
BAUD_RATE    = 115200
MIN_DIST_MM  = 100             # Jarak minimum valid = 10 cm
MAX_DIST_MM  = 8000            # Jarak maksimum valid = 8 m
MIN_QUALITY  = 1               # Kualitas minimum sinyal (abaikan 0)
GUI_REFRESH  = 30              # Interval refresh tampilan (ms)
## =========================================================================


class LidarPacketParser:
    HEADER_A    = 0xAA
    HEADER_B    = 0x55
    PACKET_SIZE = 47
    POINT_COUNT = 12

    def __init__(self):
        self.buffer     = bytearray()
        self.pkt_valid  = 0
        self.pkt_errors = 0

    def feed(self, raw_bytes):
        self.buffer.extend(raw_bytes)
        results = []
        while len(self.buffer) >= self.PACKET_SIZE:
            if (self.buffer[0] == self.HEADER_A and
                    self.buffer[1] == self.HEADER_B):
                packet = bytes(self.buffer[:self.PACKET_SIZE])
                self.buffer = self.buffer[self.PACKET_SIZE:]
                decoded = self._decode(packet)
                if decoded:
                    results.extend(decoded)
                    self.pkt_valid += 1
            else:
                self.buffer.pop(0)
                self.pkt_errors += 1
        return results

    def _decode(self, packet):
        measurements = []
        try:
            start_angle = struct.unpack_from("<H", packet, 4)[0] / 100.0
            end_angle   = struct.unpack_from("<H", packet, 6)[0] / 100.0
            angle_diff  = (end_angle - start_angle) % 360
            n           = self.POINT_COUNT
            for i in range(n):
                offset  = 8 + i * 3
                if offset + 2 >= len(packet):
                    break
                dist_mm = struct.unpack_from("<H", packet, offset)[0]
                quality = packet[offset + 2]
                angle   = (start_angle + angle_diff * i / max(n - 1, 1)) % 360
                if quality >= MIN_QUALITY and MIN_DIST_MM <= dist_mm <= MAX_DIST_MM:
                    measurements.append((angle, dist_mm, quality))
        except (struct.error, IndexError):
            pass
        return measurements

    def reset_stats(self):
        self.pkt_valid  = 0
        self.pkt_errors = 0


class LidarWorker(QtCore.QThread):
    data_signal  = QtCore.pyqtSignal(list)
    error_signal = QtCore.pyqtSignal(str)
    stats_signal = QtCore.pyqtSignal(dict)

    def __init__(self, port=SERIAL_PORT, baud=BAUD_RATE):
        super().__init__()
        self.port    = port
        self.baud    = baud
        self.running = False
        self.ser     = None
        self.parser  = LidarPacketParser()

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.5)
            self.running = True
        except serial.SerialException as exc:
            self.error_signal.emit(
                f"Gagal membuka port {self.port}.\nDetail: {exc}\n"
                f"Cek: ls -l /dev/ttyUSB* atau /dev/ttyTHS*"
            )
            return

        frame_count = 0
        t_ref = time.time()

        while self.running:
            try:
                raw = self.ser.read(256)
                if raw:
                    points = self.parser.feed(raw)
                    if points:
                        self.data_signal.emit(points)
                        frame_count += 1
                elapsed = time.time() - t_ref
                if elapsed >= 1.0:
                    self.stats_signal.emit({
                        "fps"    : round(frame_count / elapsed, 1),
                        "valid"  : self.parser.pkt_valid,
                        "errors" : self.parser.pkt_errors,
                    })
                    frame_count = 0
                    t_ref = time.time()
                    self.parser.reset_stats()
            except serial.SerialException as exc:
                self.error_signal.emit(f"Error serial: {exc}")
                break

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass


class RadarCanvas(QtWidgets.QWidget):
    RINGS_MM   = [1000, 2000, 4000, 6000, 8000]
    ANGLE_STEP = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(580, 580)
        self._points = []
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    def set_points(self, points):
        self._points = points
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        w, h   = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        max_r  = min(w, h) / 2.0 * 0.84

        painter.fillRect(self.rect(), QtGui.QColor(8, 10, 20))

        # Lingkaran jarak referensi
        for dist_mm in self.RINGS_MM:
            r     = max_r * dist_mm / MAX_DIST_MM
            alpha = 140 if dist_mm == MAX_DIST_MM else 60
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 200, 80, alpha), 1))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r))
            label = f"{dist_mm // 1000}m" if dist_mm >= 1000 else f"{dist_mm}mm"
            painter.setPen(QtGui.QColor(0, 180, 70, 180))
            painter.setFont(QtGui.QFont("Courier New", 8))
            painter.drawText(QtCore.QPointF(cx + r + 4, cy - 4), label)

        # Garis sudut
        painter.setFont(QtGui.QFont("Courier New", 9))
        for deg in range(0, 360, self.ANGLE_STEP):
            rad   = np.radians(deg - 90)
            x_end = cx + max_r * np.cos(rad)
            y_end = cy + max_r * np.sin(rad)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 160, 60, 50), 1))
            painter.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(x_end, y_end))
            xL = cx + (max_r + 20) * np.cos(rad)
            yL = cy + (max_r + 20) * np.sin(rad)
            painter.setPen(QtGui.QColor(60, 200, 100))
            painter.drawText(QtCore.QPointF(xL - 13, yL + 5), f"{deg}")

        # Titik robot
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 100), 2))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 100)))
        painter.drawEllipse(QtCore.QPointF(cx, cy), 5, 5)

        if not self._points:
            painter.setPen(QtGui.QColor(80, 80, 80))
            painter.setFont(QtGui.QFont("Courier New", 12))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter,
                             "Menunggu data LiDAR...")
            return

        closest_dist  = float("inf")
        closest_angle = 0.0
        closest_px    = cx
        closest_py    = cy

        for angle_deg, dist_mm, quality in self._points:
            ratio = dist_mm / MAX_DIST_MM
            if ratio > 1.0:
                continue
            rad  = np.radians(angle_deg - 90)
            px   = cx + max_r * ratio * np.cos(rad)
            py   = cy + max_r * ratio * np.sin(rad)
            r_ch = int(np.clip(510 * ratio,       0, 255))
            g_ch = int(np.clip(510 * (1 - ratio), 0, 255))
            size = max(2, int(5 * (1 - ratio * 0.6)))
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(r_ch, g_ch, 40, 220)))
            painter.drawEllipse(QtCore.QPointF(px, py), size, size)
            if dist_mm < closest_dist:
                closest_dist  = dist_mm
                closest_angle = angle_deg
                closest_px    = px
                closest_py    = py

        if closest_dist < float("inf"):
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 70, 70), 2))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPointF(closest_px, closest_py), 11, 11)
            pen_dash = QtGui.QPen(QtGui.QColor(255, 90, 90, 130), 1,
                                  QtCore.Qt.DashLine)
            painter.setPen(pen_dash)
            painter.drawLine(QtCore.QPointF(cx, cy),
                             QtCore.QPointF(closest_px, closest_py))
            painter.setPen(QtGui.QColor(255, 100, 100))
            painter.setFont(QtGui.QFont("Courier New", 10, QtGui.QFont.Bold))
            painter.drawText(QtCore.QPointF(closest_px + 14, closest_py + 4),
                             f"  {closest_dist / 1000:.2f}m | {closest_angle:.1f}deg")

    def sizeHint(self):
        return QtCore.QSize(620, 620)


class MainWindow(QtWidgets.QMainWindow):
    _STYLE = """
        QMainWindow, QWidget { background: #080c14; color: #00e060; }
        QLabel               { color: #00cc55; font-family: 'Courier New';
                               font-size: 13px; padding: 2px 0; }
        QLabel#title         { color: #00ff88; font-size: 17px;
                               font-weight: bold; }
        QLabel#sub           { color: #008844; font-size: 11px; }
        QLabel#value         { color: #ffffff; font-size: 14px;
                               font-weight: bold; }
        QLabel#warn          { color: #ff5050; font-size: 12px; }
        QLabel#good          { color: #00ff88; font-size: 12px; }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RPLIDAR 4000pt/s 8M  -  Wasaka Robotic")
        self.setMinimumSize(920, 680)
        self.setStyleSheet(self._STYLE)
        self._scan = {}
        self._build_ui()
        self._start_worker()

    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.radar = RadarCanvas()
        layout.addWidget(self.radar, stretch=4)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(220)
        pl = QtWidgets.QVBoxLayout(panel)
        pl.setAlignment(QtCore.Qt.AlignTop)
        pl.setSpacing(8)

        def lbl(text, obj=""):
            w = QtWidgets.QLabel(text)
            if obj:
                w.setObjectName(obj)
            return w

        def divider():
            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.HLine)
            line.setStyleSheet("border: 1px solid #0f3322;")
            return line

        pl.addWidget(lbl("RPLIDAR", "title"))
        pl.addWidget(lbl("4000 pt/s  |  Range 8 m", "sub"))
        pl.addWidget(divider())

        pl.addWidget(lbl("Port Serial:"))
        pl.addWidget(lbl(SERIAL_PORT, "value"))
        pl.addWidget(lbl("Status:"))
        self.lbl_status = lbl("Menghubungkan...", "sub")
        pl.addWidget(self.lbl_status)
        pl.addWidget(divider())

        pl.addWidget(lbl("Packet Rate:"))
        self.lbl_fps = lbl("-", "value")
        pl.addWidget(self.lbl_fps)
        pl.addWidget(lbl("Titik per Scan:"))
        self.lbl_pts = lbl("0", "value")
        pl.addWidget(self.lbl_pts)
        pl.addWidget(divider())

        pl.addWidget(lbl("Objek Terdekat:"))
        self.lbl_dist = lbl("-", "value")
        pl.addWidget(self.lbl_dist)
        pl.addWidget(lbl("Sudut:"))
        self.lbl_angle = lbl("-", "value")
        pl.addWidget(self.lbl_angle)
        pl.addWidget(divider())

        pl.addWidget(lbl("Legenda Warna:", "sub"))
        for txt, color in [
            ("< 1 m    Hijau Terang", "#00ff64"),
            ("1 - 3 m  Kuning-Hijau", "#ccff00"),
            ("3 - 6 m  Oranye",       "#ff8800"),
            ("6 - 8 m  Merah",        "#ff3030"),
        ]:
            w = QtWidgets.QLabel(f"  {txt}")
            w.setStyleSheet(f"color: {color}; font-family: 'Courier New';"
                            f"font-size: 11px;")
            pl.addWidget(w)

        pl.addWidget(divider())
        cfg = QtWidgets.QLabel(
            f"Min dist : {MIN_DIST_MM} mm\n"
            f"Max dist : {MAX_DIST_MM} mm\n"
            f"Baud     : {BAUD_RATE}"
        )
        cfg.setStyleSheet("color: #336644; font-size: 11px;"
                          "font-family: 'Courier New';")
        pl.addWidget(cfg)
        pl.addStretch()

        self.lbl_err = lbl("", "warn")
        self.lbl_err.setWordWrap(True)
        pl.addWidget(self.lbl_err)

        layout.addWidget(panel, stretch=1)

    def _start_worker(self):
        self.worker = LidarWorker()
        self.worker.data_signal.connect(self._on_data)
        self.worker.error_signal.connect(self._on_error)
        self.worker.stats_signal.connect(self._on_stats)
        self.worker.start()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(GUI_REFRESH)

    def _on_data(self, points):
        for angle, dist, quality in points:
            self._scan[round(angle, 1)] = (dist, quality)
        if self._scan:
            mk = min(self._scan, key=lambda k: self._scan[k][0])
            md, _ = self._scan[mk]
            self.lbl_dist.setText(f"{md/1000:.2f} m  ({md} mm)")
            self.lbl_angle.setText(f"{mk} deg")
            self.lbl_pts.setText(str(len(self._scan)))
        if "Menghubungkan" in self.lbl_status.text():
            self.lbl_status.setText("Terhubung & Berjalan")
            self.lbl_status.setStyleSheet("color: #00ff88; font-size: 12px;")

    def _refresh(self):
        if self._scan:
            self.radar.set_points(
                [(a, d, q) for a, (d, q) in self._scan.items()]
            )

    def _on_stats(self, stats):
        self.lbl_fps.setText(f"{stats['fps']} pkt/s")

    def _on_error(self, msg):
        self.lbl_status.setText("Error")
        self.lbl_status.setStyleSheet("color: #ff4040; font-size: 12px;")
        self.lbl_err.setText(msg)

    def closeEvent(self, event):
        self._timer.stop()
        self.worker.stop()
        self.worker.wait(2000)
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window,     QtGui.QColor(8, 12, 20))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(0, 220, 90))
    app.setPalette(pal)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
