

Auto Image captioning

**Group 3**

Sujatha

Monica

Yoga for ….

Trick Photography…

Suhail

Chaitany

a

Tennis at …

Make money when…





Agenda

q **Summary of the Tech paper**

• Problem statement

q **Solution - Model**

• Pipeline

• Papers Reference

• **Performance**

q **Architecture**

q **Key Differentiators**

• CNN + LSTM





Understanding the problem

Problem

Statement

Preliminary

Solutions

ü Template based Text Generation

ü Object detection and phrase building

ü Image and text in same vector space and

**Automated Image Captioning using**

**AIML**

ranking

ü Rigid and Hand designed

ü Fail to generate new combination of trained

objects

**Key requirements**

ü Identify Object

ü Derive Object relations

ü Activities of the objects

ü Understand the Context

ü Auto Describe the image





Architecture – Neural Image Caption Generator

Word

Word

Word

GoogelNet

CNN(Encoder)

LSTM(Decoder)

NIC is based end-to-end on a neural network consisting of a vision CNN followed by a language generating RNN

Inspiration

Text Transitions of RNN

ü RNN Encoders converting Text to Rich Vector representation

ü RNN Decoders generating Target Sentence

Summary

ü CNN as Encoder for providing rich Vector representation of image (last hidden layer from a pretrained classifier)

ü RNN Decoders generating Target Sentence





Model - Neural Image Caption Generator

Input gate

Forget gate

Output gate

LSTM

Gates

Context

Word prediction

Probability

Image Vector from CNN

St - One hot

encoding

End to End

Loss function

Data Sets

Evaluation

Results

ü Flickr8k

**Categor PASCAL Flickr30 Flickr8**

**y**

**SBU**

ü BLEU

ü Flickr30k

ü MSCOCO

ü SBU

NIC

59

66

63

28





Architecture – Automatic Image Captioning Based on ResNet50 and LSTM with Soft

Attention

RESNET50(Encoder)

LSTM(Decoder)

AICRL is based end-to-end on a neural network consisting of a RESNET50 CNN followed by a language

generating RNN

Inspiration

ü The main drawbacks of the work are the quick model overfitting, so they use the heavy and expensive

GoogLeNet with 22 hidden layers and the absence of attention layer that significantly improved the description

accuracy

Summary

ü Extract visual features, which use ResNet50 network as the encoder to generate a one-dimensional vector

representation of the input images

ü Soft Attention mechanism





Model

Training

and

Optimizatio

n

Pick Model

& Hyper

Parameters

Pre

Processing

Feature

Engineering

Evaluate

Model

Deploy on

Server

Integrate

and Test

• 1000 images

(dev) and

captions

• Captions and

ratings

provided in

Flickr8k\_Text

• Rating

• Test model

using 1000 test

images

• Check the

rating of the

Test images

• User Input

• Test the

• ResNet50 - 1D

• TensorFlow/Py

torch based

Model

• Hyper

• Feature

extraction

using

pretrained

Resnet50 to a

1D Vector

• Soft Attention

to extract more

relevant

parameters

Learning Rate

, Epochs,

Batch Size,

Drop out

• Read the

Annotations

and Image

data mapping

from

Flickr8k\_Text

to build Input –

Image and

Corresponding

Caption(s)

• FastAPI/Herok

u

• Cloud

Computing

• Hosting

Services

• API

• LSTM

• Cross Entropy

Loss Function ,

Adam/SGD

Optimizer

derived

Standard

BLEU

iteratively

• X train =

Image

• Y Train =

Caption

caption

• Compare the

outcome from

LSTM Vs

Rated Image

caption data

generated for

image as User

input provided

on a Web

page

• Hyper

feature of the

image

parameters –

determined

Iteratively

derived from

Pre processing

**Files**

• Optimizer

• LSTM

Implementatio

n

• ResNet50

Model

• LSTM

• Flickr8k\_Text

• BLEU

•

•

• Flickr8k.token

• Flickr\_8k.trainI

mages

Flickr8k\_Text

BLEU

• BLEU

• BLEU





Thank You!!

