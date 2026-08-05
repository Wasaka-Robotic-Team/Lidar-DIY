## GUI PinPoint Radar (Simple)
## Experimental Source Code for RPLIDAR
##
## Disclaimer:
## Users are expected to understand how to operate the RPLIDAR and configure
## the serial interface. Incorrect wiring or modifications to this program
## that result in malfunction or device damage will void the warranty.
##
## Accuracy Notice:
## The accuracy and reliability of this code are NOT guaranteed. Adjustments,
## tuning, or modifications may be required to achieve stable performance
## depending on your specific hardware and environment.
##
## Recommended Operating System:
## Ubuntu, other Linux distributions, or macOS are strongly recommended.
## Windows OS is not advised due to known serial interface limitations
## and restrictions.
##
## Author: Simply Smart X Teams
## For inquiries, contact: simply.smart.home.id@gmail.com | hello@simply-smart.net
##
## Redistribution of this source code without a valid license agreement
## is strictly prohibited. As this is a digital product, refund of 
## this product is not allowed.
##
## Requirements: Install numpy, pyqtgraph, and PyQt5 before use.
## Wiring Guide:
## - Red: VCC 5V DC
## - Black: TX
## - Yellow: GND
## - Green: Motor Speed Control or No Connection

import sys
import struct
import serial
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QGraphicsEllipseItem
import pyqtgraph as pg
## BEGIN OF CONFIGURATION
serialPort = "/dev/ttyUSB0" ## Change this !
## END OF CONFIGURATION
class LidarParser:
    def __init__(self):
        self.buffer = bytearray()

    def input(self, data):
        self.buffer.extend(data)
        packets = []

        while len(self.buffer) >= 47:
            if self.buffer[0] == 0xAA and self.buffer[1] == 0x55:
                packet = self.buffer[:47]
                self.buffer = self.buffer[47:]
                packets.append(packet)
            else:
                self.buffer.pop(0)
        return packets

    def decode_packet(self, packet):
        measurements = []
        if len(packet) < 47:
            return measurements

        count = 12
        start_angle = struct.unpack_from("<H", packet, 4)[0] / 100.0
        end_angle = struct.unpack_from("<H", packet, 6)[0] / 100.0
        angle_diff = (end_angle - start_angle) % 360
        angle_increment = angle_diff / (count - 1)

        for i in range(count):
            offset = 8 + i * 3
            distance = struct.unpack_from("<H", packet, offset)[0]
            angle = (start_angle + i * angle_increment) % 360
            if angle_diff > 180:
                angle_diff -= 360
            if distance != 0:
                measurements.append((angle, distance))
        return measurements

class LidarThread(QtCore.QThread):
    data_ready = QtCore.pyqtSignal(list)

    def __init__(self, port=serialPort, baudrate=115200):
        super().__init__()
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.running = True
        self.parser = LidarParser()

    def run(self):
        while self.running:
            data = self.ser.read(128)
            packets = self.parser.input(data)
            all_points = []
            for packet in packets:
                points = self.parser.decode_packet(packet)
                all_points.extend(points)
            if all_points:
                self.data_ready.emit(all_points)

    def stop(self):
        self.running = False
        self.ser.close()

class RadarViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GUI PinPoint Radar | Simply Smart X")
        self.resize(600, 600)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked()
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.hideAxis('left')

        self.plot = self.plot_widget.plot([], [], pen=None, symbol='o')

        self.plot_widget.setRange(xRange=[-6000, 6000], yRange=[-6000, 6000])
        self.plot_widget.setBackground('k')  # radar background

        self.draw_radar_grid()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.plot_widget)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.worker = LidarThread()
        self.worker.data_ready.connect(self.update_plot)
        self.worker.start()

    def draw_radar_grid(self):
        for r in range(1000, 6000, 1000):
            circle = QGraphicsEllipseItem(-r, -r, 2*r, 2*r)
            circle.setPen(pg.mkPen(color=(0, 255, 0, 60)))
            self.plot_widget.addItem(circle)

        for angle in range(0, 360, 30):
            rad = np.radians(angle)
            line = pg.PlotCurveItem(
                x=[0, np.cos(rad) * 6000],
                y=[0, np.sin(rad) * 6000],
                pen=pg.mkPen(color=(0, 255, 0, 60))
            )
            self.plot_widget.addItem(line)

    def update_plot(self, data):
        if len(data) > 0:
            print(data[:5])  # show a preview | Disable this if it affect the performance
            pass
        angles = np.radians([a for a, d in data])
        distances = [d for a, d in data]
        x = np.cos(angles) * distances
        y = np.sin(angles) * distances
        self.plot.setData(x, y)


    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    viewer = RadarViewer()
    viewer.show()
    sys.exit(app.exec_())
