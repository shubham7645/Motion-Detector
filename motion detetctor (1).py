import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class MotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Detector")
        self.root.geometry("800x600")

        self.video_source = 0
        self.capture = cv2.VideoCapture(self.video_source)
        self.prev_frame = None

        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()

        self.btn_start = tk.Button(self.root, text="Start", command=self.start_motion_detection)
        self.btn_start.pack()

        self.btn_stop = tk.Button(self.root, text="Stop", command=self.stop_motion_detection, state=tk.DISABLED)
        self.btn_stop.pack()

    def start_motion_detection(self):
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.detect_motion()

    def stop_motion_detection(self):
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    def detect_motion(self):
        ret, frame = self.capture.read()

        if not ret:
            messagebox.showerror("Error", "Failed to capture video")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray

        frame_diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        self.prev_frame = gray

        if self.btn_stop['state'] == tk.NORMAL:
            self.root.after(10, self.detect_motion)

    def __del__(self):
        if self.capture is not None:
            self.capture.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = MotionDetectorApp(root)
    root.mainloop()
