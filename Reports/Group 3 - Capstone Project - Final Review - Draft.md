

Automated Image captioning

**Group 3**

Sujatha Kancharla

Monica Nukarapu

Suhail Pasha

Yo ga for ….

Trick Photography…

Chaitanya Harkara

**Mentor**

Nayan Jha

Tennis at …

Make money when…





Agenda

q **Problem Statement**

q **Solution - Model**

• Architecture

• Understanding the problem

q **Architecture**

• Results

• Base Model ( Already discussed)

• InceptionV3 + LSTM

• ResNet50 +LSTM

• UI Integration

q **Learnings and Next steps**

• Attention –

• VGG16 + Attention

• Resnet50 + Attention

• Transformers

• InceptionV3 + Transformers

• Resnet50 + Transformers





Understanding the problem

Problem Statement

Automated Image Captioning using AIML

Automated Image captioning involves in creating an automated caption for an Image by

deriving the best context of the contents of the image.

Key Requirements

Broadly the solution should

a. Identify multiple objects within the image

b. Derive the relationship between the objects in the image based on their attributes

c. Derive the caption based on the derived context of the image in Natural language (English)

Technology

· **Tools** : Natural Language Toolkit, TensorFlow, PyTorch, Keras

· **Deployments:** FastAPI, Cloud Application Platform | Heroku, Streamlit, Cloud Computing,

Hosting Services, and APIs | Google Cloud





Base model – Automatic Image Captioning Based on Resnet50/InceptionV3 and LSTM

Word

Word

Word

Resent50/InveceptionV3 - CNN(Encoder)

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





Base model- Implementation Encoder (InceptionV3)+ LSTM

Pick Model &

Hyper

Parameters

Model

Training and

Optimization

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

• 1000 images (Test)

and captions

• Captions and ratings

provided in

Flickr8k\_Text

• Rating Standard

**BLEU**

• Compare the

outcome from

LSTM Vs Rated

Image caption data

using BLEU rating

standard

• Hyper parameters

Learning Rate ,

Epochs, Batch Size,

Drop out derived

iteratively

• X train = Image

(6000)

• Y Train = Caption

derived from Pre-

processing and

feature engineering

• Image Feature

extraction using

pretrained

**InceptionV3** to a 1D

**Vector (299,299) to**

**(2048,)**

• Dictionary based on

**Glove 200 attribute**

word embedding

from Stanford.edu

• **InceptionV3** - 1D - (

2048,)

• **Keras**/Pytorch

based Model

• Test model using

250 test images

• Check the rating of

the Test images

• User Input

• Test the caption

generated for image

as User input

• Read the

Annotations and

Image data

mapping from

Flickr8k\_Text to

build Input – Image

and Corresponding

Caption(s)

• FastAPI/Heroku

• Cloud Computing

• Hosting Services

• API

• LSTM

• **Cross Entropy Loss**

**Function , Adam**

• Hyper parameters –

determined

provided on a Web

page

Iteratively

**Files**

• InceptionV3 Model

• LSTM

• Glove Word

Embedding (200)

• Optimizer

• LSTM

Implementation

• Flickr8k\_Text

• BLEU

• Flickr8k\_Text

• BLEU

• Flickr8k.token

• Flickr\_8k.trainImag

es

•

Flickr8k\_Text

• BLEU





Base model- Implementation using ResNet50 + LSTM

Pick Model &

Hyper

Parameters

Model

Training and

Optimization

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

• 1000 images (Test)

and captions

• Captions and ratings

provided in

Flickr8k\_Text

• Image Feature

extraction using

pretrained

**Resnet50** to a 1D

Vector (229,229,3)

to (2048,)

• Dictionary based on

**Glove 200 attribute**

word embedding

from Stanford.edu

• Soft Attention to

extract more

• Hyper parameters

Learning Rate ,

Epochs, Batch Size,

Drop out derived

iteratively

• X train = Image

(6000)

• Y Train = Caption

derived from Pre-

processing and

feature engineering

• **ResNet50 - 1D -**

**( 2048,)**

• **Keras**/Pytorch

based Model

• LSTM

• **Cross Entropy Loss**

**Function , Adam**

• Hyper parameters –

determined

• Test model using

1000 test images

• Check the rating of

the Test images

• User Input

• Test the caption

generated for image

as User input

• Read the

Annotations and

Image data

mapping from

Flickr8k\_Text to

build Input – Image

and Corresponding

Caption(s)

• FastAPI/Heroku

• Cloud Computing

• Hosting Services

• API

• Rating Standard

**BLEU**

• Compare the

outcome from

LSTM Vs Rated

Image caption data

using BLEU rating

standard

provided on a Web

page

Iteratively

relevant feature of

the image

• ResNet50

Model

• LSTM

• Glove word

embedding(200)

**Files**

• Optimizer

• LSTM

Implementation

• Flickr8k\_Text

• BLEU

• Flickr8k\_Text

• BLEU

• Flickr8k.token

• Flickr\_8k.trainI

mages

• Flickr8k\_Text

• BLEU





Base model- Implementation using InceptionV3

**Decoder**

**Encoder(InceptionV3)**

2048,

299,299

Image

256

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

Bleu

score

**No of Images**

**Average BLEU Score #BLEU Score (<0.1)**

**(>=0.1)**

250

46(0.21)

204





Base model- Implementation using ResNet50

**Encoder(ResNet50)**

**Decoder**

229,229,

3

2048,

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

**Average BLEU Score # BLEU Score (<0.1)**

**(>=0.1)**

Bleu

score

250

61(0.287)

189





Output captions – Base Model(LSTM)

Inception V3

Resnet50





Attention – Automatic Image Captioning Based on ResNet50/VGG16, Attention and LSTM

Feature

vector

ResNet50(Encoder)

LSTM(Decoder)

Reference

AICRL is based end-to-end on a neural network consisting of a RESNET50 CNN followed by a language generating RNN

Inspiration

ü Adding attention layer on ResNet50 significantly improved the description accuracy

Summary

ü Extract visual features, which use ResNet50 network as the encoder from the input images

ü Soft Attention mechanism





Attention - Implementation using Encoder(ResNet50) + Attention

Pick Model &

Hyper

Parameters

Model

Training and

Optimization

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

• 1000 Validation

images

• Captions and ratings

provided in

Flickr8k\_Text

• Rating Standard

**BLEU**

• Compare the

outcome from

Attention Vs Rated

Image caption data

using BLEU rating

standard

• Image Feature

extraction using

pretrained

**Resnet50** from

input (224,224,3) to

(7,7,2048)

• Word embedding

from using Keras

• Soft Attention to

extract more

weights for feature

of the image

• Hyper parameters

Learning Rate ,

Epochs, Batch Size,

Drop out derived

iteratively

• X train = Image

(6000)

• Y Train = Caption

derived from Pre-

processing and

feature engineering

• Test model using

250 test images

• Check the rating of

the Test images

• User Input

• Test the caption

generated for image

as User input

• Read the

Annotations and

Image data

mapping from

Flickr8k\_Text to

build Input – Image

and Corresponding

Caption(s)

• **ResNet50**

**Keras**/Pytorch

based Model

• GRU

• **SparseCategoricalCr**

**ossentropy, Adam**

• FastAPI/Heroku

• Cloud Computing

• Hosting Services

• API

provided on a Web

page

**Files**

• ResNet50

Model

• GRU

• Keras Word

embedding

• Flickr8k\_Text

• BLEU 1 , BLEU 2

, BLUE 3 , BLUE4

• Flickr8k\_Text

• BLEU 1 , BLEU 2

, BLUE 3 , BLUE4

• Flickr8k.token

• Flickr\_8k.trainI

mages

• Optimizer

• GRU (LSTM)

• Flickr8k\_Text

• BLEU 1 , BLEU 2

, BLUE 3 , BLUE4

• Flickr30





Attention - Implementation using ResNet50 + GRU

features

Context

(64,49,1)

(224,224,3)

(64,7,7,20

\48)

(64,7,7,20

\48)

(64,49,256)

Relu

tanh

Context

**Word Embedding**

(64, 8329)

(64, 1, 256)

**GRU**





Automated Captions generated- (Attention) + Flickr8 + 70 Epochs

**Resnet50**

**VGG16**





Transformers- Implementation using Encoder(ResNet50) + Transformers

Transformer architecture

• Transformer architecture using Multi head attention on

Image input

• Transformer architecture helps parallelization of

sequential data

• GPUs are used efficiently processing the multiple words

in the sentence





Transformers- Implementation using Encoder(ResNet50/InceptionV3) + Transformers

Pick Model &

Hyper

Parameters

Model

Training and

Optimization

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

• **Keras**/Pytorch

based Model

• 1000 images (Test)

and captions

• Captions provided

in Flickr8k\_Text

• Rating Standard

**BLEU**

• Compare the

outcome from

LSTM Vs Rated

Image caption data

using BLEU rating

standard

• Hyper parameters

Learning Rate ,

Epochs, Batch Size,

Drop out derived

iteratively

• X train = Image

(6000)

• Y Train = Caption

derived from Pre-

processing and

feature engineering

• Image Feature

extraction using

pretrained

**Resnet50** for image

input Vector

(229,229,3) to

(64,49,512)

• Word embedding

from Keras

• Feature vectore +

position Encoding

• Word Embedding +

Position Encoding

• Mutlihead attention

• Transformers

• **SparseCategoricalCr**

**ossentropy, Adam**

• Hyper parameters –

determined

• Test model using

250 test images

• Check the rating of

the Test images

• User Input

• Test the caption

generated for image

as User input

• Read the

Annotations and

Image data

mapping from

Flickr8k\_Text to

build Input – Image

and Corresponding

Caption(s)

• FastAPI/Heroku

• Cloud Computing

• Hosting Services

• API

provided on a Web

page

• Muli head attention

Iteratively

• Multi Head = 8

**Files**

• Flickr8k\_Text

• Flickr30k\_Text

• ResNet50 Model

• Keras Word

embedding

• Optimizer

• LSTM

Implementation

• Flickr8k

• Flickr30k

• Flickr8k\_Text

• BLEU 1 , BLEU 2

, BLUE 3 , BLUE4

• Flickr8k.token

• Flickr\_8k.trainI

mages

• BLEU 1 , BLEU 2

, BLUE 3 , BLUE4

•

BLEU 1 , BLEU 2

, BLUE 3 , BLUE4

• Position encoding

• Flickr30k





Transformer - Implementation using ResNet50

(224,224,3)

(64,7,7,2048)

(64,7,7,20

\48)

(64,49,512)

(64,49,512)

(64,49,512)

(64,49,512)

(64,49,512)

(64,32)

(64,32)

(64,32,512

(64,32,512)

(64,32 5001)

(5001)





Sample Ouput – Resnet50+70 Epochs + No MHA + Transformer + Flickr8





Automated Captions– Transformers- InceptionV3 + Flickr8k





Automated Captions – Transformers – Resnet50 + Flickr30k





Automated Captions – Transformers – Resnet50 + Flickr30k





Results Snapshot view for various models

**Model**

**Image Feature extraction**

**Hyper Parameters and training details**

**Data Set**

**Results ( Data captured after running on 250 random Test Images)**

Inception V3

Loss - Cross Entropy

Optimizer – Adam

Epochs – 45

Learning rate – 0.0001

Training – 8000 images

Flickr8

Flickr8

BLEU-3 count: 46

BLEU-3 count: 61

BLEU-3 count average 0.21

BLEU-3 count average 0.2875

**Base Model - LSTM**

Resnet50

Flickr8

BLEU-1 count: 176

BLEU-2 count: 71

BLEU-3 count: 26

BLEU-4 count: 4

BLEU-1 Average: 0.2559

BLEU-2 Average: 0.2275

BLEU-3 Average: 0.2541

BLEU-4 Average: 0.2975

Loss - SparseCategoricalCrossentropy

Optimizer – Adam

Epochs – 45

Learning rate – 0.0001

Training – 8000 images

VGG16

**Attention**

Flickr8/

Flickr30

BLEU-1 count: 178

BLEU-2 count: 89

BLEU-3 count: 35

BLEU-4 count: 14

BLEU-1 Average: 0.2773 BLEU-1 Average: 0.2527

BLEU-2 Average: 0.2501 BLEU-2 Average: 0.2550

BLEU-3 Average: 0.2901 BLEU-3 Average: 0.2908

BLEU-4 Average: 0.2608 BLEU-4 Average: 0.2751

Resnet50

Flickr8

Flickr8

Flickr30

BLEU-1 count: 195

BLEU-2 count: 103

BLEU-3 count: 44

BLEU-4 count: 10

BLEU-1 Average: 0.3115

BLEU-2 Average: 0.2738

BLEU-3 Average: 0.3011

BLEU-4 Average: 0.2824

Inception V3

Loss - SparseCategoricalCrossentropy

Optimizer – Adam

Epochs – 70

Training – 8000 images

Multi Head - 8

BLEU-1 count: 200

BLEU-2 count: 100

BLEU-3 count: 40

BLEU-4 count: 12

BLEU-1 Average: 0.2679

BLEU-2 Average: 0.2412

BLEU-3 Average: 0.2981

BLEU-4 Average: 0.3194

**Transformers**

Resnet50

Loss - SparseCategoricalCrossentropy

Optimizer – Adam

Epochs – 35

Training – 25000 images

Multi Head – 8

BLEU-1 count: 177

BLEU-2 count: 82

BLEU-3 count: 22

BLEU-4 count: 7

BLEU-1 Average: 0.2516

BLEU-2 Average: 0.2232

BLEU-3 Average: 0.2901

BLEU-4 Average: 0.2837





Deployment model

1)Create application in Google cloud and

Fast API & UI

Integration

Google Cloud

Integration

install Cloud SDK

Model generation

2)Open command prompt on local

machine and enter below commands:

• Create FAST API App

with reference to

specific model like LSTM

, Attention and

cd C:\Model

Deployment\deploy\_aic\_lstm-main

• Train model and save

weights in Colab

notebook for

• LSTM

• Attention

• Transformer

• Save Encoded Tokenizer

model

gcloud init --it sets your

• Build and deploy on

Google Cloud using the

input from the Fast API

• Set the configuration to

2GB

gmail

Transformer

gcloud config get-value

• Build a python file which

replicates the model in

the colab

• Build HTML to take

Image input and retrieve

captions

project

gcloud builds submit --tag

gcr.io/deploy-aic-lstm/aiclstm --deploy-

aic-lstm is project id created in cloud and

aiclstm is service name we intend to give

3)Got to Gcloud UI

->cloud run

• Models trained on

Flickr8 and Flickr30

• Tokenizer file created on

captions provided with

Flickr8 and Flickr30

• Docker file from Fast API

->Create service aiclstm and enter

• Models from Colab are

integrated and executed

locally

implementation , HTML , details

Configuration , Support

Functions , Start





Caption Generation – UI Integration( Fast API)





Challenges and mitigation

**Challenge**

**Description**

**Mitigation**

Flickr30k data set processing

·

Flickr30K is of size 4G

·

Downloaded the image files on Google drive

·

The number of files and size created an issue with

processing while training the models

·

Build the Numpy files from Pretrained Resnet50 on Google

drive

·

Incrementally move the Numpy files to Collab and while

training and execution

Processing Multiple epochs for Flickr30K ·

Flickr30K is of size 4G

·

·

Train the models incrementally for 10-15 epoch at time

and build save the weights

·

The number of files and size created an issue with

processing while training the models

Start the next training with from previous training end

point by loading saved weights

·

Performance of training- Each epoch runs ~10 min

Evaluation of captions using Bleu score

Transformers Implementation

·

·

Improving evaluation criteria of Bleu score for the

captions

·

·

Blues score were calculated for Single , Bi gram, Tri gram

and N Grams for values >0.1 and averages across all the

caption generation for Test images

Captions were created processed only for Word

embedding + Positional encoding and directly passed to

Encoder decoder MHA block

Additional MHA for Word embedding(captions) was

part of the model which had to be removed from

traditional approach of transformer implementation

Integration with Fast API /Cloud

·

·

Limitation in Heroku for 512

·

·

To deploy on GCP Cloud

Tensorflow models need high configuration (> 1GB

RAM)

Give higher configuration to deploy on cloud

Github upload for huge files( >25 MB)

More than 25 MB files were not opening on github

Deployed github lfs and uploaded files





Learnings and Next steps

Next Steps

Learnings

**To look for better validation model than Bleu score**

ü FLICKR30 size was huge and upload to collab was a

challenge. It was partitioned into multiple sets and

uploaded in Google drive

ü Tuning Hyper Parameters

**Training(s)**

ü UI Integration with Transformer code was delayed

ü Transformers + Flickr30

ü Attention + Flickr30

**Data**

ü Performance issues with training Transformers over

Flickr30 as it error out with Allocated memory

issues ..etc – To implement training in parts by using

the generated weights from previous epochs

ü

Flickr30





Thank You!!





Model - LSTM

Input gate

Forget gate

Output gate

LSTM Gates

Context

Word prediction

Probability

Image Vector from CNN

Loss function

Word Embedding

End to End





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

