import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

model = YOLO("yolo11n.pt")
names = model.names

cap = cv2.VideoCapture("peaky_blinders_s03e01.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Parent directory for cropped images
parent_dir = "Cropped_Images"

# Define the subdirectories for saving cropped images based on class
save_dirs = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Create the parent directory if it doesn't exist
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

# Create the subdirectories inside the parent directory if they don't exist
for dir_name in save_dirs.values():
    full_dir = os.path.join(parent_dir, dir_name)
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

# Classes you want to detect and crop: 0 (Person), 2 (Car), 3 (Motorcycle), 5 (Bus), 7 (Truck)
class_ids = [0, 2, 3, 5, 7]

# Video writer
video_writer = cv2.VideoWriter("object_cropping_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

idx = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(im0, line_width=2, example=names)

    if boxes is not None:
        for box, cls in zip(boxes, clss):
            # Crop only the specified classes
            if int(cls) in class_ids:
                idx += 1
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                crop_obj = im0[int(box[1]): int(box[3]), int(box[0]): int(box[2])]

                # Save the cropped image to the corresponding folder inside "Cropped_Images"
                save_dir = os.path.join(parent_dir, save_dirs[int(cls)])
                cv2.imwrite(os.path.join(save_dir, f"{save_dirs[int(cls)]}_{idx}.png"), crop_obj)

    cv2.imshow("ultralytics", im0)
    video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
