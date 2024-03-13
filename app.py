import joblib
import cv2
import numpy as np
from tensorflow.keras.saving import load_model
from tensorflow.keras.utils import load_img,img_to_array
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array,load_img
import base64
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
import os
app = FastAPI()

# Add CORS middleware to allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

uuid = 0

def process_rf_image(image):
    img = cv2.imread(image)
    img=cv2.resize(img,(224,224))
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l_hist = cv2.calcHist([lab_image], [0], None, [256], [0, 256]).flatten()
    a_hist = cv2.calcHist([lab_image], [1], None, [256], [0, 256]).flatten()
    b_hist = cv2.calcHist([lab_image], [2], None, [256], [0, 256]).flatten()


    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256]).flatten()
    s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256]).flatten()

    ft=np.array([np.concatenate([l_hist, a_hist, b_hist,h_hist,s_hist,v_hist])])
    ft=ft.reshape(ft.shape[0], -1)
    return ft

def process_knn_image(image):
    img = cv2.imread(image)
    img=cv2.resize(img,(224,224))
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l_hist = cv2.calcHist([lab_image], [0], None, [256], [0, 256]).flatten()
    a_hist = cv2.calcHist([lab_image], [1], None, [256], [0, 256]).flatten()
    b_hist = cv2.calcHist([lab_image], [2], None, [256], [0, 256]).flatten()

    ft=np.array([np.concatenate([l_hist, a_hist, b_hist])])
    ft=ft.reshape(ft.shape[0], -1)
    return ft

def process_cnn_image(img):
    img = load_img(img, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize pixel values
    return img_array

print("Loading Models")

conjunctiva_cnn_model_path = './Models/conj_model_98.h5'
conjunctiva_knn_model_path = './Models/knn_conj_99.joblib'
conjunctiva_rf_model_path = './Models/RF_CJ_99.joblib'

palm_cnn_model_path = './Models/palm_model_98f.h5'
palm_knn_model_path = './Models/knn_palm_98.joblib'
palm_rf_model_path = './Models/RF_PM_97.joblib'

finger_nails_cnn_model_path = './Models/fing_model_95.h5'
finger_nails_knn_model_path = './Models/knn_fn_98.joblib'
finger_nails_rf_model_path = './Models/RF_FN_97.joblib'

final_model_path='./Models/svm_model_100.joblib'

conjunctiva_cnn_model = load_model(conjunctiva_cnn_model_path,compile=False)
conjunctiva_knn_model = joblib.load(conjunctiva_knn_model_path)
conjunctiva_rf_model = joblib.load(conjunctiva_rf_model_path)

finger_nails_cnn_model = load_model(finger_nails_cnn_model_path,compile=False)
finger_nails_knn_model = joblib.load(finger_nails_knn_model_path)
finger_nails_rf_model = joblib.load(finger_nails_rf_model_path)

palm_cnn_model = load_model(palm_cnn_model_path,compile=False)
palm_knn_model = joblib.load(palm_knn_model_path)
palm_rf_model = joblib.load(palm_rf_model_path)

final_model=joblib.load(final_model_path)

print('All models loaded')

@app.get('/')
async def index():
    return {"message": "Welcome to Anemia Detection System!"}

# /conjuctiva route
@app.post('/conjunctiva')
async def predict_conjuctiva(request: Request):
    global uuid
    uuid +=1
    data = await request.json()
    image_data = data['image_data']
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = f"conjunctiva_{uuid}.png"
    with open(image, 'wb') as f:
        f.write(image_bytes)
    
    # Call the predict function
    prediction_cnn = conjunctiva_cnn_model.predict(process_cnn_image(image))
    prediction_knn = conjunctiva_knn_model.predict(process_knn_image(image))
    prediction_rf = conjunctiva_rf_model.predict(process_rf_image(image))

    predictions1="{:.5f}".format(prediction_cnn[0][0])
    predictions2=prediction_knn[0]
    predictions3=prediction_rf[0]
    
    # Delete the image file after prediction
    # print(predictions)
    os.remove(image)
    uuid -=1
    return {'cnn':str(predictions1),'knn':str(predictions2),'rf':str(predictions3) }

# /finger_nail route
@app.post('/finger_nail')
async def predict_finger_nails(request: Request):
    global uuid
    uuid +=1
    data = await request.json()
    image_data = data['image_data']
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = f"finger_nail_{uuid}.png"
    with open(image, 'wb') as f:
        f.write(image_bytes)
        
    
    # Call the predict function
    prediction_cnn = finger_nails_cnn_model.predict(process_cnn_image(image))
    prediction_knn = finger_nails_knn_model.predict(process_knn_image(image))
    prediction_rf = finger_nails_rf_model.predict(process_rf_image(image))

    predictions1="{:.5f}".format(prediction_cnn[0][0])
    predictions2=prediction_knn[0]
    predictions3=prediction_rf[0]
    
    # Delete the image file after prediction
    # print(predictions)
    os.remove(image)
    uuid -=1
    return {'cnn':str(predictions1),'knn':str(predictions2),'rf':str(predictions3) }

# /palm route
@app.post('/palm')
async def predict_palm(request: Request):
    global uuid
    uuid +=1
    data = await request.json()
    image_data = data['image_data']
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = f"palm_{uuid}.png"
    with open(image, 'wb') as f:
        f.write(image_bytes)
    
    # Call the predict function
    prediction_cnn = palm_cnn_model.predict(process_cnn_image(image))
    prediction_knn = palm_knn_model.predict(process_knn_image(image))
    prediction_rf = palm_rf_model.predict(process_rf_image(image))

    predictions1="{:.5f}".format(prediction_cnn[0][0])
    predictions2=prediction_knn[0]
    predictions3=prediction_rf[0]
    
    # Delete the image file after prediction
    # print(predictions)
    os.remove(image)
    uuid -=1
    return {'cnn':str(predictions1),'knn':str(predictions2),'rf':str(predictions3) }

# /final route
@app.post('/final')
async def predict_final(request: Request):
    data = await request.json()
    arr = data['features']
    print(arr)
    float_elements = [float(element) for element in arr[:3]]
    int_elements = [int(element) for element in arr[3:]]
    features=float_elements + int_elements
    pred=final_model.predict_proba([features])
    no="{:.2f}".format(pred[0][0]*100)
    yes="{:.2f}".format(pred[0][1]*100)
    return {'no':str(no), 'yes':str(yes)}

