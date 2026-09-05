from ultralytics import YOLO
import numpy as np
import cv2 as cv
from model_paths import PROJECT_ROOT, require_model


def train(data_dir=PROJECT_ROOT / "datasets" / "covid19"):
    model = YOLO("yolov8n-cls.yaml")
    model.train(data=str(data_dir), epochs=100)
    
    

def predict(img, st):
    model_path = require_model("lung_classification", st)
    model = YOLO(model_path)
    
    results = model.predict(img)
    result = results[0]
    
    class_names = result.names
    class_id = int(result.probs.top1) if hasattr(result.probs, "top1") else int(np.argmax(result.probs.data.tolist()))
    class_name = class_names[class_id].upper()
    _, width = img.shape[:2]
    cv.putText(img, class_name, (max(width - 220, 10), 60), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv.LINE_AA)
    
    
    st.subheader('Output Image')
    st.image(img, channels="BGR", use_container_width=True)
