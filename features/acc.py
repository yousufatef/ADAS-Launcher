import sys
import cv2
import serial
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk
import threading
import datetime

# Arduino Setup
arduino = None
arduino_connected = False
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600)
    time.sleep(2)
    arduino_connected = True
except Exception as e:
    print(f"[ERROR] Arduino connection failed: {e}")

# YOLOv8 setup
model = YOLO("yolov8n.pt")

# RealSense Setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

realsense_connected = True
try:
    pipeline.start(config)
except Exception as e:
    print(f"[ERROR] RealSense pipeline failed to start: {e}")
    realsense_connected = False
    exit(1)

# Distance thresholds
STOP_THRESHOLD = 2.0
TARGET_CLASSES = [0, 2, 7]  # person, car, truck

# Logging setup
log_file = open("acc_log.txt", "w")

class ACCApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ADAS - RealSense + Simulated LIDAR + Monitor")
        self.root.geometry("1300x540")

        self.video_label = tk.Label(root)
        self.video_label.grid(row=0, column=0)

        self.lidar_label = tk.Label(root)
        self.lidar_label.grid(row=0, column=1)

        self.last_command = "GO"
        self.running = True
        self.fps = 0

        self.update_thread = threading.Thread(target=self.update_frame)
        self.update_thread.daemon = True
        self.update_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_frame(self):
        align = rs.align(rs.stream.color)
        prev_time = time.time()

        while self.running:
            start_time = time.time()

            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                continue

            color_image = np.asanyarray(color.get_data())
            results = model(color_image, verbose=False)[0]

            h, w, _ = color_image.shape
            center_distance = depth.get_distance(int(w / 2), int(h / 2))

            # Filtered detection
            stop_detected = False
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls in TARGET_CLASSES:
                    cx = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                    cy = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                    obj_distance = depth.get_distance(cx, cy)
                    if obj_distance < STOP_THRESHOLD:
                        stop_detected = True
                        break

            command = "STOP" if stop_detected else "GO"
            if command != self.last_command and arduino_connected:
                try:
                    arduino.write((command + "\n").encode())
                    self.last_command = command
                except:
                    pass

            # Logging
            log_file.write(f"{datetime.datetime.now()}, Distance: {center_distance:.2f}, Command: {command}\n")

            # Draw
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                color_box = (0, 255, 0) if cls in TARGET_CLASSES else (100, 100, 100)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)

            # System Info
            self.fps = 1.0 / (time.time() - prev_time)
            prev_time = time.time()
            cv2.putText(color_image, f"FPS: {self.fps:.1f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 0), 2)
            cv2.putText(color_image, f"Distance: {center_distance:.2f} m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(color_image, f"Command: {command}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(color_image, f"RealSense: {'OK' if realsense_connected else 'Fail'}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(color_image, f"Arduino: {'OK' if arduino_connected else 'Fail'}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Show in GUI
            color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(color_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk

            # Fake LIDAR animation
            lidar_frame = np.zeros((480, 600, 3), dtype=np.uint8)
            cv2.putText(lidar_frame, f"Fake LIDAR Distance: {center_distance:.2f} m", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            angle = int(time.time() * 120) % 360
            x = int(300 + 100 * np.cos(np.radians(angle)))
            y = int(240 + 100 * np.sin(np.radians(angle)))
            cv2.circle(lidar_frame, (x, y), 10, (0, 255, 255), -1)
            cv2.line(lidar_frame, (300, 240), (x, y), (0, 255, 255), 2)

            lidar_rgb = cv2.cvtColor(lidar_frame, cv2.COLOR_BGR2RGB)
            lidar_pil = Image.fromarray(lidar_rgb)
            lidar_tk = ImageTk.PhotoImage(image=lidar_pil)
            self.lidar_label.configure(image=lidar_tk)
            self.lidar_label.image = lidar_tk

    def on_close(self):
        print("[INFO] Shutting down ACC system")
        self.running = False
        time.sleep(1)
        if arduino_connected:
            arduino.close()
        pipeline.stop()
        log_file.close()
        self.root.destroy()

# Main
if __name__ == '__main__':
    root = tk.Tk()
    app = ACCApp(root)
    root.mainloop()
