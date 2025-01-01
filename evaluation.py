from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
from PIL import Image

DATASET_FOLDER = 'stop_sign_dataset'

directory = os.listdir(DATASET_FOLDER)

image_paths = []
for file in directory:
    image_paths.append(os.path.join(DATASET_FOLDER, file))

model = YOLO('/home/tugcekul/Desktop/deneme/runs/detect/train10/weights/best.pt')

output_dir = "output/images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

results = model.predict(image_paths)

for result in results:
    filename = os.path.join(output_dir, os.path.basename(result.path))
    result.save(filename=filename)

print("Images processed and saved.")