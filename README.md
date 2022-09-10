# grp3capstone-final

Capstone project for Group3 --- Base Model with basic implementation of Encoder + Word Embedding + LSTM using pretrained models

Project Objective - Automated Captioning of Images

### Source of dataset:

Flickr8k: https://github.com/goodwillyoga/Flickr8k_dataset

Flickr30k: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

Coco: https://cocodataset.org/#home

### Python files:

### ResNet50/InveptionV3 + LSTM

#### ImageCaptionGroup3_BaseLine_Inception.ipynb

Encoder - InceptionV3, Word Embedding - Glove + LSTM, Decoder - FC, Data Set - FLickr8k, Model - Keras + Tensorflow, Evaluation - Bleu Results

#### ImageCaptionGroup3_BaseLine_resnet50.ipynb

Encoder - Resnet50, Word Embedding - Glove + LSTM, Decoder - FC, Data Set - FLickr8k, Model - Keras + Tensorflow, Evaluation - Bleu Results

#### ImageCaptionGroup3_BaseLine_resnet50_LSTM.ipynb

Encoder - Resnet50, Word Embedding - Glove + LSTM, Decoder - LSTM + FC, Data Set - FLickr8k, Model - Keras + Tensorflow, Evaluation - Bleu Results,

#### ImageCaptionGroup3_Flickr30_Resnet.ipynb

Encoder - Resnet50, Word Embedding - Glove + LSTM, Decoder - LSTM + FC, Data Set - FLickr30k, Model - Keras + Tensorflow, Evaluation - Bleu Results,


### Attention

#### ImageCaptionGroup3_Attention_VGG16.ipynb

Encoder - VGG16, Word Embedding - Decoder - GRU + Attention, Data Set - FLickr8k, Model - Keras + Tensorflow, Evaluation - Bleu Results

#### ImageCaptionGroup3_Attention_Resnet50.ipynb

Encoder - Resnet50, Word Embedding - Decoder - GRU + Attention, Data Set - FLickr8k, Model - Keras + Tensorflow, Evaluation - Bleu Results

#### ImageCaptionGroup3_Attention_Resnet50_Flickr30.ipynb

Encoder - Resnet50, Word Embedding - Decoder - GRU + Attention, Data Set - FLickr30k, Model - Keras + Tensorflow, Evaluation - Bleu Results


### Transformer

#### ImageCaptiongenerator_Transformer_InceptionV3.ipynb

Encoder - InceptionV3, Word Embedding - Decoder - Transformer, Data Set - FLickr8k, Model - Keras + Tensorflow, Evaluation - Bleu Results

#### ImageCaptiongenerator_Transformer_Restnet50.ipynb

Encoder - Resnet50, Word Embedding - Decoder - Transformer, Data Set - FLickr8, Model - Keras + Tensorflow, Evaluation - Bleu Results

#### ImageCaptiongenerator_Transformer_Resnet50_No_MHA.ipynb

Encoder - Resnet50, Word Embedding - Decoder - Transformer, Data Set - FLickr8, Model - Keras + Tensorflow, Evaluation - Bleu Results

#### ImageCaptiongenerator_Transformer_Resnet50_Flickr30.ipynb

Encoder - Resnet50, Word Embedding - Decoder - Transformer, Data Set - FLickr30, Model - Keras + Tensorflow, Evaluation - Bleu Results

#### ImageCaptiongenerator_Transformer_Resnet50_No_MHA_Flickr30.ipynb

Encoder - Resnet50, Word Embedding - Decoder - Transformer, Data Set - FLickr30, Model - Keras + Tensorflow, Evaluation - Bleu Results



### Steps for deployment:

Run colab and save model.h5 and encoded_tokenizer.pkl files

Create FastAPI folder with all the files.

Create main.py with all the imports and methods required to predict the image uploaded from local drive

Create Dockerfile with all configuration and CMD exec to run the FastAPI on gunicorn with uvicorn workers and timeout to 600

   Changed pip to pip3 in 'RUN pip3 install -r requirements.txt' as there are errors related to tensorflow version with python 2.7 version

Create requirements.txt file with all the packages required to be installed for the application to run

Go to Google cloud and create application.

Download and install Google cloud SDK

Open Command prompt and run the below commands

   cd <folder_path_on_local>

   gcloud init --it sets your gmail

   gcloud config get-value project

   gcloud builds submit --tag gcr.io/deploy-aic-lstm/aiclstm --deploy-aic-lstm is project id created in cloud and aiclstm is service name we intend to give

This create container image in google clour run

Search for Cloud run in Cloud console and click on create service

Enter configurations as required and click on deploy

It takes few minutes for the URL to be displayed

For CI/CD,click on 'Enable continuous deployment' on home page of Cloud run and enter repo.
