import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from ultralytics import YOLO

# إعداد نافذة tkinter
root = tk.Tk()
root.title("Heatmap Model GUI")
root.geometry("400x200")

# تحميل الموديل
model = YOLO("yolov8n.pt")  # تأكد من استخدام موديل خفيف

# دالة لمعالجة الفيديو
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Error reading video file")
        return

    # إعداد خصائص الفيديو
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fps = cap.get(cv2.CAP_PROP_FPS)

    # إعداد حفظ الفيديو المُعدل
    output_filename = "heatmap_output_with_trails.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    # القاموس لتتبع الأشخاص بحسب معرفهم
    tracks = {}

    def generate_heatmap_with_trails(frame, detections, colormap=cv2.COLORMAP_JET, radius=10, intensity=0.6):
        heatmap = np.zeros_like(frame[:, :, 0], dtype=np.float32)
        for detection in detections:
            person_id = int(detection.index)
            if int(detection.cls.item()) == 0:  # التحقق من أن الجسم هو شخص
                x1, y1, x2, y2 = detection.xyxy[0].tolist()
                x_center = int((x1 + x2) // 2)
                y_center = int((y1 + y2) // 2)

                if person_id not in tracks:
                    tracks[person_id] = []
                tracks[person_id].append((x_center, y_center))

                for i in range(1, len(tracks[person_id])):
                    prev_point = tracks[person_id][i - 1]
                    current_point = tracks[person_id][i]
                    prev_point = (int(prev_point[0]), int(prev_point[1]))
                    current_point = (int(current_point[0]), int(current_point[1]))
                    cv2.line(frame, prev_point, current_point, (0, 255, 0), thickness=2)
                cv2.circle(heatmap, (x_center, y_center), radius, 1, -1)

        if np.max(heatmap) > 0:
            heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15, sigmaY=15)
            heatmap = np.uint8(255 * heatmap / np.max(heatmap))
        else:
            heatmap = np.uint8(heatmap)

        heatmap_color = cv2.applyColorMap(heatmap, colormap)
        combined = cv2.addWeighted(frame, 1 - intensity, heatmap_color, intensity, 0)
        return combined

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            messagebox.showinfo("Info", f"Video processing completed. Output saved as {output_filename}")
            break

        results = model.track(frame, persist=True, verbose=False)
        detections = results[0].boxes
        heatmap_frame_with_trails = generate_heatmap_with_trails(frame, detections)
        video_writer.write(heatmap_frame_with_trails)

    cap.release()
    video_writer.release()

# دالة لاختيار ملف الفيديو
def load_video():
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4 *.avi")])
    if video_path:
        process_video(video_path)

# زر لتحميل الفيديو
load_btn = tk.Button(root, text="Load Video", command=load_video)
load_btn.pack(pady=20)

# تشغيل واجهة المستخدم
root.mainloop()
