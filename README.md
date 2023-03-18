# Fast API and ONNXRUNTIME - PyTorch Image classifaction Model Deployment

<p align="center">
  <img width="800" height="450" src="https://user-images.githubusercontent.com/78400305/226119180-ba850e99-5e54-4716-85d9-1eb11805b7ab.png">
</p>

- This repo is a part for following project 
[Web Scraping with product search relevance using NLP, rules and image classification](https://github.com/jithinanievarghese/product-search-relevance/blob/main/README.md)
- The [image classifcation model](https://github.com/jithinanievarghese/image_classification_pytorch) we trained on PyTorch is deployed using onnnxruntime and Fast API
- The PyTorch Model was exported to onnx format using the [following code](https://github.com/jithinanievarghese/export_model_to_onnx)
- Here we are using onnnxruntime instead of PyTorch to run our inference and serving it using Fast API

## Why  ONNXRUNTIME over PyTorch or Tensorflow inference

- The cpu version of deep learning framewok like pytorch alone consumes nearly 700mb of server size
- This can be a bottleneck while deployment, considering the resources and latency in inference.
- [ONNX (Open Neural Network Exchange )](https://onnx.ai/) is an open format built to represent machine learning models  i.e a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.
- After the coverting the pytorch model to onnx format
- With [onnxruntime](https://onnxruntime.ai/docs/get-started/with-python.html) we can make faster inference without any pytorch or any other deep learning framework dependencies 

## Usage
- pip install - r  requirements.txt
- uvicorn main:app --reload
- load the image path and send request to `http://127.0.0.1:8000/predict`  
  the response of the request will be the probability of image to be a spider man related image
  ```python
  import requests

  headers = {
      'accept': 'application/json',
      'Content-Type': 'application/json'}

  files = {'file': open(image_path, 'rb')}
  response = requests.post(
      'http://127.0.0.1:8000/predict', 
      headers=headers, 
      files=files)
  response.json()
  ```
  `output`
  ```python
  {'1': 0.9999956862174809, '0': 4.313782519083098e-06}
  ```
