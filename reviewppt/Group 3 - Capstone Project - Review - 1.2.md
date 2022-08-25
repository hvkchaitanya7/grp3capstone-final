

Automated Image captioning

**Group 3**

Sujatha Kancharla

Monica

Nukarapu

Suhail Pasha

Yoga for ….

Trick Photography…

Chaitanya Harkara

**Mentor**

Nayan Jha

Tennis at …

Make money when…





Agenda

q **Solution - Model**

• Architecture

q **Problem Statement**

• Problem statement

• Results

q **Architecture**

q **Learnings and Next steps**

• Base Model1 – InceptionV3 +

LSTM

• Base Model2 – ResNet50

+LSTM





Understanding the problem

Problem

Statement

Automated Image Captioning using AIML

Automated Image captioning involves in creating an automated caption for

an Image by deriving the best context of the contents of the image.

Broadly the solution should

Key Requirements

a. Identify multiple objects within the image

b. Derive the relationship between the objects in the image based on their

attributes

c. Derive the caption based on the derived context of the image in Natural

language (English)

Technology

**Tools** : Natural Language Toolkit, TensorFlow, PyTorch, Keras

·

· **Deployments:** FastAPI, Cloud Application Platform | Heroku, Streamlit,

Cloud Computing, Hosting Services, and APIs | Google Cloud





Architecture – Automatic Image Captioning Based on InceptionV3 and LSTM

Word

Word

Word

InceptionV3 - CNN(Encoder)

LSTM(Decoder)

Reference

NIC is based end-to-end on a neural network consisting of a vision CNN followed by a language generating RNN

Inspiration

Text Transitions of RNN

ü RNN Encoders converting Text to Rich Vector representation

ü RNN Decoders generating Target Sentence

Summary

ü CNN as Encoder for providing rich Vector representation of image (last hidden layer from a pretrained classifier)

ü RNN Decoders generating Target Sentence





Base model2 – Automatic Image Captioning Based on ResNet50 and LSTM

ResNet50(Encoder)

LSTM(Decoder)

Reference

AICRL is based end-to-end on a neural network consisting of a RESNET50 CNN followed by a language generating RNN

Inspiration

ü Adding attention layer on ResNet50 significantly improved the description accuracy

Summary

ü Extract visual features, which use ResNet50 network as the encoder to generate a one-dimensional vector representation of the input images

ü Soft Attention mechanism





Base model1- Implementation using InceptionV3

Model

Pick Model

& Hyper

Parameters

Training

and

Optimizatio

n

Pre-

Processing

Feature

Engineering

Evaluate

Model

Deploy on

Server

Integrate

and Test

• 1000 images (dev)

and captions

• 1000 images

(Test) and

• Image Feature

extraction using

pretrained

**InceptionV3** to a

1D **Vector**

• Hyper parameters

Learning Rate ,

Epochs, Batch

Size, Drop out

derived iteratively

• X train = Image

(6000)

• Y Train = Caption

derived from Pre-

processing and

feature

• **InceptionV3** - 1D

\- ( 2048,)

• **Keras**/Pytorch

based Model

• LSTM

• **Cross Entropy**

**Loss Function ,**

**Adam**

• Hyper parameters

– determined

Iteratively

• Test model using

1000 test images

• Check the rating

of the Test images

• User Input

• Test the caption

generated for

image as User

input provided on

a Web page

• Read the

Annotations and

Image data

mapping from

Flickr8k\_Text to

build Input –

Image and

Corresponding

Caption(s)

captions

• Captions and

ratings provided in

Flickr8k\_Text

• Rating Standard

**BLEU**

• Compare the

outcome from

LSTM Vs Rated

Image caption

data using BLEU

rating standard

**(299,299) to**

**(2048,)**

• FastAPI/Heroku

• Cloud Computing

• Hosting Services

• API

• Dictionary based

on **Glove 200**

**attribute** word

embedding from

Stanford.edu

• Soft Attention to

extract more

engineering

relevant feature of

the image

• InceptionV3

Model

• LSTM

• Glove Word

Embedding (200)

**Files**

• Optimizer

• LSTM

Implementation

• Flickr8k\_Text

• BLEU

•

•

• Flickr8k.token

• Flickr\_8k.trainIma

ges

Flickr8k\_Text

BLEU

• Flickr8k\_Text

• BLEU





Base model2- Implementation using ResNet50

Model

Pick Model

& Hyper

Parameters

Training

and

Optimizatio

n

Pre-

Processing

Feature

Engineering

Evaluate

Model

Deploy on

Server

Integrate

and Test

• 1000 images (dev)

and captions

• 1000 images

(Test) and

• Image Feature

extraction using

pretrained

**Resnet50** to a 1D

Vector (229,229,3)

to (2048,)

• Dictionary based

on **Glove 200**

**attribute** word

embedding from

Stanford.edu

• Hyper parameters

Learning Rate ,

Epochs, Batch

Size, Drop out

derived iteratively

• X train = Image

(6000)

• Y Train = Caption

derived from Pre-

processing and

feature

• **ResNet50 - 1D -**

**( 2048,)**

• **Keras**/Pytorch

based Model

• LSTM

• **Cross Entropy**

**Loss Function ,**

**Adam**

• Hyper parameters

– determined

Iteratively

• Test model using

1000 test images

• Check the rating

of the Test images

• User Input

• Test the caption

generated for

image as User

input provided on

a Web page

• Read the

Annotations and

Image data

mapping from

Flickr8k\_Text to

build Input –

Image and

Corresponding

Caption(s)

captions

• Captions and

ratings provided in

Flickr8k\_Text

• Rating Standard

**BLEU**

• Compare the

outcome from

LSTM Vs Rated

Image caption

data using BLEU

rating standard

• FastAPI/Heroku

• Cloud Computing

• Hosting Services

• API

• Soft Attention to

extract more

relevant feature of

the image

engineering

• ResNet50

Model

• LSTM

• Glove word

embedding(20

\0)

**Files**

• Optimizer

• LSTM

Implementatio

n

• Flickr8k\_Text

• BLEU

•

•

• Flickr8k.token

• Flickr\_8k.trainI

mages

Flickr8k\_Text

BLEU

• Flickr8k\_Text

• BLEU





Base model1- Implementation using InceptionV3

**Decoder**

**Encoder(InceptionV3)**

2048,

299,299

256

Image

Relu

**Loss** - Cross Entropy

**Optimizer** - Adam

**Word Embedding(Glove)**

**Epochs** – 45

**Learning rate** – 0.0001

1652,

200

34 word

Max

256

**Efficacy Measurement** – BLEU ( Trigram)

Relu

Softmax

1652 >10

rep words

**No of Images**

**Average BLEU**

**Score (>=0.1)**

**#BLEU Score**

**(<0.1)**

Bleu

score

200

46(0.21)

154





Base model2- Implementation using ResNet50

**Encoder(ResNet50)**

**Decoder**

229,229,

2048,

3

256

Image

Relu

**Loss** - Cross Entropy

**Optimizer** - Adam

**Word Embedding(Glove)**

**Epochs** – 45

**Learning rate** – 0.0001

1652,

200

34 word

Max

256

**Efficacy Measurement** – BLEU ( Trigram)

Relu

Softmax

1652 >10

rep words

**No of Images**

**Average BLEU**

**Score (>=0.1)**

**# BLEU Score**

**(<0.1)**

Bleu

score

200

49(0.234)

151





Output captions – InceptionV3





Output captions - ResNet50





Learnings and Next steps

**T**

**r validation model**

Learnings

**than Bleu score**

Next Steps

**Other Decoder(s)**

ü Captions evaluation using Bleu score

based on Expert caption vs prediction is

not high

ü

LSTM Attention

ü Transformers

**Other Pre-Trained CNN ( Encoder)**

**Models**

ü Captions seem to be less accurate if more

objects and back grounds are involved

ü GoogleNet

ü Predicted captions with low Bleu score

**Other Decoder Models (Word**

**Embeddings)**

seem appropriate

ü Captions generated from such model will

have words only limited to training

dataset which means model may not

generate captions for images with new

content.

ü

ü

Word2Vec

BERT

**Data**

ü COCO





Thank You!!





Model - LSTM

Input gate

Forget gate

Output gate

LSTM

Gates

Context

Word prediction

Probability

Image Vector from CNN

Word Embedding

End to End

Loss function





Architecture – Keras Implementation

inputs1 = Input(shape=(2048,))

fe1 = Dropout(0.5)(inputs1)

fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max\_length,))

se1 = Embedding(vocab\_size, embedding\_dim, mask\_zero=True)(inputs2)

se2 = Dropout(0.5)(se1)

se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])

decoder2 = Dense(256, activation='relu')(decoder1)

outputs = Dense(vocab\_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)

