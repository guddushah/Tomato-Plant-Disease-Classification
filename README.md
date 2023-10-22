# TOMATO PLANT DISEASE CLASSIFICATION

## Problem Statement:
#### This project predicts the fungal and bacterial diseases in tomato plants with an android app and help farmers identify the particular type of disease early to apply specific treatment to control the economic loss.
#### Dataset: https://www.kaggle.com/datasets/arjuntejaswi/plant-village

## Project Architecture
- #### tf model built using CNN architecture
- #### Website built in ReactJS to send plant images as http request to backend
- #### Backend built using FastAPI framework(mostly used for numpy convertion of the image data)
- #### tf-serving server created using tensorflow/serving docker image (deployed the project file to this image) for serving the http request from website. This server is basically created to control the version management of the models.
- #### Created GCP bucket in google cloud for uploading the model to the cloud and using google sdk for uploading the function to serve the requests coming from android app
- #### Created an android app in React Nativeto predicting the label of the uploaded plant image file.

  ### URL- https://us-central1-eternal-trees-394521.cloudfunctions.net/predict
  ### Postman Demo (making http call using API)
https://github.com/guddushah/Tomato-Plant-Disease-Classification-Deep-Learning/assets/40028193/93c33ed9-0186-490c-a4a2-1746590b715c


  ### Website Demo
https://github.com/guddushah/Tomato-Plant-Disease-Classification-Deep-Learning/assets/40028193/cfede4cf-0336-4343-b41b-8d0eb1d77992


  ### App Demo
https://github.com/guddushah/Tomato-Plant-Disease-Classification-Deep-Learning/assets/40028193/b8b25fe6-9501-4c5b-b846-383814c21552


  
