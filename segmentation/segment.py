from segmentation.masks_to_polygons import *
from ultralytics import YOLO
import cv2 as cv
from PIL import Image, ImageDraw
from model_paths import PROJECT_ROOT, require_model
    
def prepare_input():
    masks_to_polygons()
    split_train_test_val()
    
    

def train(data_config=PROJECT_ROOT / "segmentation" / "data.yaml"):
    
    model = YOLO("yolov8n-seg.yaml")
    model.train(data=str(data_config), epochs=100)
    
def predict(img, st):
    model_path = require_model("ultrasound_segmentation", st)
    H, W, _ = img.shape
    
    model = YOLO(model_path)
    results = model.predict(img)
    
            
    if results[0].masks is None:
        st.warning("No segmentation masks were detected for this image.")
        return

    print("\n[INFO]Number of masks detected:", len(results[0].masks))
    
    mask_out =None
    for result in results:
        for _, mask_ in enumerate(result.masks.data):

            mask_gray = mask_.cpu().numpy() * 255
            mask_gray = cv.resize(mask_gray, (W, H))
            
            if mask_out is None:
                mask_out = mask_gray
            else:
                mask_out = cv.bitwise_or(mask_out, mask_gray)
            
            
            
        
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    for result in results:
        for mask_ in result.masks:
            polygon = mask_.xy[0]
            draw.polygon(polygon,outline=(0,255,0), width=5)
            
    st.subheader('Output Image')
    cols = st.columns(2)
    cols[0].image(mask_out, clamp=True, channels='GRAY', use_container_width=True)
    cols[1].image(im, use_container_width=True)
    
