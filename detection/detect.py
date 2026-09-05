from ultralytics import YOLO
from PIL import Image
from model_paths import PROJECT_ROOT, require_model


def train(data_config=PROJECT_ROOT / "datasets" / "blood_cells" / "data.yaml"):
    model = YOLO('yolov8n.yaml')  # build a new model from scratch
    model.train(data=str(data_config), epochs=100)  # train the model

    # or you can run following in command line:
    # yolo detect train data=data.yaml model="yolov8n.yaml" epochs=1

def predict(img, confidence, st):
    # detection model
    model_path = require_model("blood_cell_detection", st)
    model = YOLO(model_path)
     
     # Predict
    results = model.predict(img, conf=confidence)
    result = results[0]
    
    print("\n[INFO] Number of objects detected : ", len(result.boxes) ) #this stuff gets printed to terminal
    
    
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        #im.show()  # show image
        #im.save('results.jpg')  # save image
        
    st.subheader('Output Image')
    st.image(im, use_container_width=True)
