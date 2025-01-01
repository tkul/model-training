from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

required_dirs = [
    "datasets/train/images",
    "datasets/train/labels",
    "datasets/valid/images",
    "datasets/valid/labels"
]

for dir_path in required_dirs:
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Error: {dir_path} does not exist. Please create the folders and place the files.")

model = YOLO('/home/tugcekul/Desktop/deneme/runs/detect/train9/weights/best.pt')

params = {
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'optimizer': 'Adam'
}

data_yaml = "datasets/data.yaml"

results = model.train(
    data=data_yaml,
    **params
)

confidence_threshold = 0.15
test_results = model.val(conf=confidence_threshold)

print("mAP50:", test_results['mAP50'])
print("Precision:", test_results['precision'])
print("Recall:", test_results['recall'])

for img in test_results['images']:
    pred = model.predict(img)
    plt.imshow(pred.imgs[0])
    plt.show()
