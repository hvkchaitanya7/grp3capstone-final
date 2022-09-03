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

