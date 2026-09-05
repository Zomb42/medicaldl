
from ultralytics import YOLO
import numpy as np
import cv2 as cv
from model_paths import PROJECT_ROOT, require_model



def train(data_dir=PROJECT_ROOT / "data"):
    # Path to the YOLO configuration file for classification
    model = YOLO("yolov8n-cls.pt")
    
    # Path to your dataset
    save_dir = PROJECT_ROOT / 'runs' / 'muscleclassify'

    # Train the model
    results = model.train(data=str(data_dir), epochs=2, project=str(save_dir))


def predict(img, st):
    # Path to the best weights file after training
    model_path = require_model("fracture_classification", st)
    #model_path = '/Users/derickshi/Documents/Yolomedical/bonebest.pt'
    # Load the trained model
    model = YOLO(model_path)
    
    # Predict the class of the input image
    results = model.predict(img)
    result = results[0]
    
    # Get class names and probabilities
    class_names = result.names
    class_id = int(result.probs.top1) if hasattr(result.probs, "top1") else int(np.argmax(result.probs.data.tolist()))
    class_name = class_names[class_id].upper()
    
    # Display the class name on the image
    height, width = img.shape[:2]
    cv.putText(img, class_name, (width - 200, 60), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv.LINE_AA)
    
    # Display the image using Streamlit
    st.subheader('Output Image')
    st.image(img, channels="BGR", use_container_width=True)

