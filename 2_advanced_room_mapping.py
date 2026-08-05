## GUI Room Mapping RPLIDAR
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
## Requirements: Install numpy and PyQt5 before use.
##
## Change The Configuration as Needed
##
## Wiring Guide:
## - Red: VCC 5V DC
## - Black: TX
## - Yellow: GND
## - Green: Motor Speed Control or No Connection

import sys
import serial
import struct
import threading
import numpy as np

from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QTimer
## BEGIN OF CONFIGURATION
## Serial Configuration
PORT = '/dev/ttyUSB0'  # Change this !
BAUD = 115200
## Map Configuration
MAP_SIZE_METERS = 16  # 16m x 16m map
MAP_RESOLUTION = 0.05  # meters per pixel
## END OF CONFIGURATION
MAP_SIZE_PIXELS = int(MAP_SIZE_METERS / MAP_RESOLUTION)
occupancy_grid = np.full((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS), -1, dtype=np.int8) 
angles = []
distances = []
ser = serial.Serial(PORT, BAUD, timeout=1)
class RadarWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GUI Room Mapping RPLIDAR | Simply Smart X')
        self.resize(600, 600)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.drawRadar(qp)
        qp.end()
    def drawRadar(self, qp):
        qp.fillRect(self.rect(), Qt.black)
        center = self.width() / 2, self.height() / 2
        radius = min(self.width(), self.height()) / 2 * 0.9
        qp.setPen(QPen(Qt.gray, 1))
        for r in range(1, 5):
            qp.drawEllipse(int(center[0] - radius * r / 5),
                        int(center[1] - radius * r / 5),
                        int(2 * radius * r / 5),
                        int(2 * radius * r / 5))
        cell_size = self.width() / MAP_SIZE_PIXELS
        for y in range(MAP_SIZE_PIXELS):
            for x in range(MAP_SIZE_PIXELS):
                value = occupancy_grid[y, x]
                if value == 1:
                    qp.setPen(QPen(Qt.red))
                    qp.drawPoint(int(x * cell_size), int(y * cell_size))
                elif value == 0:
                    qp.setPen(QPen(Qt.darkGray))
                    qp.drawPoint(int(x * cell_size), int(y * cell_size))
        qp.setPen(QPen(Qt.green, 2))
        for a, d in zip(angles, distances):
            r = (d / 8.0) * radius
            x = center[0] + r * np.cos(a)
            y = center[1] - r * np.sin(a)
            qp.drawPoint(int(x), int(y))
def read_lidar():
    global angles, distances
    buffer = b''
    while True:
        buffer += ser.read(512)
        while True:
            if len(buffer) < 8:
                break
            if buffer[0] != 0xAA or buffer[1] != 0x55:
                buffer = buffer[1:]
                continue
            LSN = buffer[3]
            size_needed = 8 + LSN * 3 + 2
            if len(buffer) < size_needed:
                break
            CT = buffer[2]
            FSA = struct.unpack('<H', buffer[4:6])[0] / 100
            LSA = struct.unpack('<H', buffer[6:8])[0] / 100
            data_points = []
            for i in range(LSN):
                offset = 8 + i * 3
                distance = struct.unpack('<H', buffer[offset:offset+2])[0]
                quality = buffer[offset+2]
                data_points.append((distance, quality))
            checksum = struct.unpack('<H', buffer[8 + LSN*3:10 + LSN*3])[0]
            angles.clear()
            distances.clear()
            robot_x = MAP_SIZE_METERS / 2
            robot_y = MAP_SIZE_METERS / 2
            angle_diff = (LSA - FSA) % 360
            for i, (distance, quality) in enumerate(data_points):
                if LSN == 1:
                    angle = FSA
                else:
                    angle = (FSA + angle_diff * i / (LSN - 1)) % 360
                if distance != 0:
                    distance_m = distance / 1000.0
                    theta = np.deg2rad(angle)
                    x = robot_x + distance_m * np.cos(theta)
                    y = robot_y + distance_m * np.sin(theta)
                    map_x = int(x / MAP_RESOLUTION)
                    map_y = int(y / MAP_RESOLUTION)
                    if 0 <= map_x < MAP_SIZE_PIXELS and 0 <= map_y < MAP_SIZE_PIXELS:
                        occupancy_grid[map_y, map_x] = 1
                    steps = int(distance_m / MAP_RESOLUTION)
                    for s in range(steps):
                        free_x = robot_x + (s * MAP_RESOLUTION) * np.cos(theta)
                        free_y = robot_y + (s * MAP_RESOLUTION) * np.sin(theta)
                        map_free_x = int(free_x / MAP_RESOLUTION)
                        map_free_y = int(free_y / MAP_RESOLUTION)
                        if 0 <= map_free_x < MAP_SIZE_PIXELS and 0 <= map_free_y < MAP_SIZE_PIXELS:
                            if occupancy_grid[map_free_y, map_free_x] == -1:
                                occupancy_grid[map_free_y, map_free_x] = 0
            angle_diff = (LSA - FSA) % 360
            for i, (distance, quality) in enumerate(data_points):
                if LSN == 1:
                    angle = FSA
                else:
                    angle = (FSA + angle_diff * i / (LSN - 1)) % 360
                if distance != 0:
                    distances.append(distance / 1000.0)
                    angles.append(np.deg2rad(angle))

            buffer = buffer[size_needed:]
if __name__ == '__main__':
    threading.Thread(target=read_lidar, daemon=True).start()
    app = QApplication(sys.argv)
    radar = RadarWidget()
    radar.show()
    sys.exit(app.exec_())
