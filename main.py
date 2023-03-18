from feature_extraction import FeatureExtraction
from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import numpy as np

app = FastAPI()


@app.post("/predict")
async def fetch_predictions(file: UploadFile = File(...)):
    """
    to read the image content from api request
    extract the features, make inference in onnxruntime
    and return the json response

    # since model was trained on pytorch, model expects input in form
    # batch size X Number of channels X Image Height X Image Width
    # we reshape the numpy image array (Channels, Image Height, Image Width)
    # to include the batch size 1
    # so final shape is (batch size, Channels, Image Height, Image Width)
    """
    image_array = FeatureExtraction(await file.read()).get_features()
    image_array_reshaped = image_array.reshape((1, 3, 300, 300))
    ort_session = ort.InferenceSession("model.onnx")
    outputs = ort_session.run(None, {"actual_input": image_array_reshaped})
    prediction = get_sigmoid(outputs[0].flatten()[0])
    return {"1": prediction, "0": 1-prediction}


def get_sigmoid(x):
    """
    to apply sigmoid function to model output
    """
    return 1/(1 + np.exp(-x))
