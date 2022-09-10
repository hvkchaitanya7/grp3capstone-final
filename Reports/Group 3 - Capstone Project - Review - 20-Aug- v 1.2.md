Automated Image captioning Automated Image captioning 
Yoga for …. Trick Photography… 
Tennis at … Make money when… 
Group 3 

Sujatha Kancharla 


Monica Nukarapu 
Suhail Pasha 
Chaitanya Harkara 

Mentor 

Nayan Jha 


Agenda Agenda 
. 
Problem Statement 
• 
Understanding the problem 
. 
Architecture 
• 
Base Model ( Already discussed) 
• 
InceptionV3 + LSTM 
• 
ResNet50 +LSTM 
• 
Attention – 
• 
VGG16 + Attention 
• 
Resnet50 + Attention 
• 
Transformers 
• 
InceptionV3 + Transformers 
• 
Resnet50 + Transformers 
. 
Solution -Model 
• 
Architecture 
• 
Results 
• 
UI Integration 
. 
Learnings and Next steps 

Understanding the problem Understanding the problem 
Problem Statement 
Key Requirements 
Technology 
Automated Image Captioning using AIML 

Automated Image captioning involves in creating an automated caption for an Image by 
deriving the best context of the contents of the image. 

Broadly the solution should 

a. Identify multiple objects within the image 
b. Derive the relationship between the objects in the image based on their attributes 
c. Derive the caption based on the derived context of the image in Natural language (English) 
. 
Tools : Natural Language Toolkit, TensorFlow, PyTorch, Keras 
. 
Deployments: FastAPI, Cloud Application Platform | Heroku, Streamlit, Cloud Computing, 
Hosting Services, and APIs | Google Cloud 


Base model –Automatic Image Captioning Based on Resnet50/InceptionV3 and LSTM Base model –Automatic Image Captioning Based on Resnet50/InceptionV3 and LSTM 
Features 
Resent50/InveceptionV3 -CNN(Encoder) LSTM(Decoder) 
Caption 
Word Word Word 
Reference 
NIC is based end-to-end on a neural network consisting of a vision CNN followed by a language generating RNN 

Inspiration 
Text Transitions of RNN 

• 
RNN Encoders converting Text to Rich Vector representation 
• 
RNN Decoders generating Target Sentence 
Summary 
• 
CNN as Encoder for providing rich Vector representation of image (last hidden layer from a pretrained classifier) 
• 
RNN Decoders generating Target Sentence 

Base model-Implementation Encoder (Resnet50/InceptionV3)+ LSTM Base model-Implementation Encoder (Resnet50/InceptionV3)+ LSTM 
• 
Read the 
Annotations and 
Image data 
mapping from 
Flickr8k_Text to 
build Input – 
Image 
and Corresponding 
Caption(s) 
Feature 
Engineering 
Pre 
Processing 
• 
Image Feature 
extraction using 
pretrained 
InceptionV3 toa 1D 
Vector (299,299) to 
(2048,) 
• 
Dictionary based on 
Glove 200 attribute 

word embedding 
from Stanford.edu 

Pick Model & 
Hyper 
Parameters 
• 
InceptionV3 -1D -( 
2048,) 
• 
Keras/Pytorch 
based Model 
• 
LSTM 
• 
Cross Entropy Loss 
Function , Adam 
• 
Hyper parameters – 
determined 
Iteratively 
Model 
Training and 
Optimization 
• 
Hyper parameters 
Learning Rate , 
Epochs, Batch Size, 
Drop out derived 
iteratively 
• 
X train = Image 
(6000) 
• 
Y Train = Caption 
derived from Preprocessing 
and 
feature engineering 
Evaluate 
Model 
Integrate 
and Test 
Deploy on 
Server 
•FastAPI/Heroku 
•Cloud Computing 
•Hosting Services 
•API 
• 
1000 images (dev) 
and captions 
• 
1000 images (Test) 
and captions 
• 
Captions and ratings 
provided in 
Flickr8k_Text 
• 
Rating Standard 
BLEU 

• 
Compare the 
outcome from 
LSTM Vs Rated 
Image caption data 
using BLEU rating 
standard 
Key Steps



• 
Test model using 
250 test images 
• 
Check the rating of 
the Test images 
• 
User Input 
• 
Test the caption 
generated for image 
as User input 
provided on a Web 
page 
Files 
•Flickr8k.token 
•Flickr_8k.trainImag 
es 
•InceptionV3 Model 
•LSTM 
•Glove Word 
Embedding (200) 
Inputs•Optimizer 
•LSTM 
Implementation 
•Flickr8k_Text 
•Flickr8k_Text 
•BLEU •BLEU 
•Flickr8k_Text 
•BLEU 

Base model-Implementation using InceptionV3 Base model-Implementation using InceptionV3 
Decoder 

Encoder(InceptionV3) 

InceptionV3 
Flickr8299,299 
Image 
2048, 
Drop out+ DenseRelu 
256 
Word Embedding(Glove) 
Glove Word1652, 
Caption Data34 word 
Drop out 
ing256 
200Max 
eddEmb1652 >10 
rep words 


LSTM(256) 
Dense(256) 
Relu 
Dense(1652) 
Softmax 
Loss -Cross Entropy 
Optimizer -Adam 

Epochs –45 
Learning rate –0.0001 

Efficacy Measurement –BLEU ( Trigram) 


No of Images Average BLEU Score 
(>=0.1) 
#BLEU Score (<0.1) 
250 46(0.21) 204 
Bleu 
score 

Encoder(ResNet50) Decoder 
Base model-Implementation using ResNet50 
Decoder 
Base model-Implementation using ResNet50 
Loss -Cross Entropy 
Optimizer -Adam 
Epochs –45 
Learning rate –0.0001 
229,229, 
3 
Image 
2048, 
256 
Relu 
Drop out+ Dense 
ResNet50 
Flickr8 
LSTM(256) 
Dense(256) 
Dense(1652) 
1652, 
200 
Word Embedding(Glove) 
Caption Data34 word 
Max 
1652 >10 
rep words 
Glove WordEmbedding
Drop out 
256 
Efficacy Measurement –BLEU ( Trigram)
Relu 

Softmax 
Bleu 
score 
No of Images Average BLEU Score 
(>=0.1) 
# BLEU Score (<0.1) 
250 61(0.287) 189 


Output captions –Base Model(LSTM) 
Inception V3 Resnet50 
Output captions –Base Model(LSTM) 
Inception V3 Resnet50 

Attention –Automatic Image Captioning Based on ResNet50/VGG16, Attention and LSTM Attention –Automatic Image Captioning Based on ResNet50/VGG16, Attention and LSTM 
ResNet50(Encoder) LSTM(Decoder) 
Caption 
Reference 
Feature 
vector 

AICRL is based end-to-end on a neural network consisting of a RESNET50 CNN followed by a language generating RNN 

Inspiration 
• 
Adding attention layer on ResNet50 significantly improved the description accuracy 
Summary 
• 
Extract visual features, which use ResNet50 network as the encoder from the input images 
• 
Soft Attention mechanism 

Attention -Implementation using Encoder(ResNet50) + Attention Attention -Implementation using Encoder(ResNet50) + Attention 
• 
Read the 
Annotations and 
Image data 
mapping from 
Flickr8k_Text to 
build Input – 
Image 
and Corresponding 
Caption(s) 
Files 
•Flickr8k.token 
•Flickr_8k.trainI 
mages 
•Flickr30 
Feature 
Engineering 
Pick Model & 
Hyper 
Parameters 
Model 
Training and 
Optimization 
Evaluate 
Model 
Integrate 
and Test 
Deploy on 
Server 
Pre 
Processing 
• 
ResNet50 
Keras/Pytorch 
based Model 
• 
GRU 
• 
SparseCategoricalCr 
ossentropy, Adam 
InputsKey Steps



• 
Image Feature 
extraction using 
pretrained 
Resnet50 from 
input (224,224,3) to 
(7,7,2048) 
• 
Word embedding 
from using Keras 
• 
Soft Attention to 
extract more 
weights for feature 
of the image 
• 
Hyper parameters 
Learning Rate , 
Epochs, Batch Size, 
Drop out derived 
iteratively 
• 
X train = Image 
(6000) 
• 
Y Train = Caption 
derived from Preprocessing 
and 
feature engineering 
• 
1000 Validation 
images 
• 
Captions and ratings • 
Test model using 
provided in 250 test images 
Flickr8k_Text • 
Check the rating of 
• 
Rating Standard • 
FastAPI/Heroku the Test images 
BLEU • 
Cloud Computing • 
User Input 
• 
Compare the • 
Hosting Services • 
Test the caption 
outcome from • 
API generated for image 
Attention Vs Rated as User input 
Image caption data provided on a Web 
using BLEU rating page 
standard 

•ResNet50 
Model 
•GRU 
•Keras Word 
embedding 
•Optimizer 
•GRU (LSTM) 
•Flickr8k_Text 
•Flickr8k_Text 
•BLEU 1 , BLEU 2 
, BLUE 3 , BLUE4 
•BLEU 1 , BLEU 2 
, BLUE 3 , BLUE4 
•Flickr8k_Text 
•BLEU 1 , BLEU 2 
, BLUE 3 , BLUE4 

Attention -Implementation using ResNet50 + GRU Attention -Implementation using ResNet50 + GRU 
features Context 


softmax 
Attention Weights 
Dense(256)
Attention layerDense(512)
Dense(512) 
Encoder layerFlickr Image
(224,224,3) 


Resent50/Inception 
Feature Vector 
(64,7,7,20 


48) 



(64,7,7,20 


48) 



tanh 

(64,49,1) 


(64,49,256) 


Relu 


Context 

Word Embedding 

GRU 
FC1FC2(64, 8329)
Word Embedding 
Feature with Attention 
Caption Data 
Word Embedding 
(64, 1, 256) 



Automated Captions generated-(Attention) Automated Captions generated-(Attention) 
VGG16 Resnet50 




Transformers-Implementation using Encoder(ResNet50/InceptionV3) + Transformers Transformers-Implementation using Encoder(ResNet50/InceptionV3) + Transformers 
• 
Read the 
Annotations and 
Image data 
mapping from 
Flickr8k_Text to 
build Input – 
Image 
and Corresponding 
Caption(s) 
Files 
•Flickr8k.token 
•Flickr_8k.trainI 
mages 
•Flickr30k 
Feature 
Engineering 
Pick Model & 
Hyper 
Parameters 
Model 
Training and 
Optimization 
Evaluate 
Model 
Integrate 
and Test 
Deploy on 
Server 
Pre 
Processing 
• 
Hyper parameters 
Learning Rate , 
Epochs, Batch Size, 
Drop out derived 
iteratively 
• 
X train = Image 
(6000) 
• 
Y Train = Caption 
derived from Preprocessing 
and 
feature engineering 
• 
Image Feature 
extraction using 
pretrained 
Resnet50 for image 
input Vector 
(229,229,3) to 
(64,49,512) 
• 
Word embedding 
from Keras 
• 
Muli head attention 
• 
Keras/Pytorch 
based Model 
• 
Feature vectore + 
position Encoding 
• 
Word Embedding + 
Position Encoding 
• 
Mutlihead attention 
• 
Transformers 
• 
SparseCategoricalCr 
ossentropy, Adam 
• 
Hyper parameters – 
determined 
Iteratively 
• 
Multi Head = 8 
• 
1000 images (Test) 
and captions 
• 
Captions provided 
in Flickr8k_Text 
• 
Rating Standard 
BLEU 

• 
Compare the 
outcome from 
LSTM Vs Rated 
Image caption data 
using BLEU rating 
standard 
InputsKey Steps



• 
FastAPI/Heroku 
• 
Cloud Computing 
• 
Hosting Services 
• 
API 
• 
Test model using 
250 test images 
• 
Check the rating of 
the Test images 
• 
User Input 
• 
Test the caption 
generated for image 
as User input 
provided on a Web 
page 
•ResNet50 Model 
•Keras Word 
embedding 
•Position encoding 
•Optimizer 
•LSTM 
Implementation 
•Flickr8k 
•Flickr30k 
•Flickr8k_Text 
•Flickr30k_Text 
•BLEU 1 , BLEU 2 
, BLUE 3 , BLUE4 
•BLEU 1 , BLEU 2 
, BLUE 3 , BLUE4 
•Flickr8k_Text 
•BLEU 1 , BLEU 2 
, BLUE 3 , BLUE4 

Transformer -Implementation using ResNet50 Transformer -Implementation using ResNet50 
Encoder layerDecoder LayerCaption Flickr Image
Feature VectorPosition Encoding(1d) 
Word EmbeddingMulti head Attention LayerEncoder + Decoder Multi 
head Attention layerFeed ForwardSoftmax 
Dense(5001) 
Resent50/InceptionPosition Encoding(2d)
Multi head AttentionLayerFeed ForwardLayer Normalization 
(224,224,3) (64,7,7,20 
48) 
(64,7,7,2048) 
Dense(512)
(64,49,512) (64,49,512) (64,49,512) (64,49,512) 
(64,32) (64,32) (64,32,512) (64,32,512 (64,32,512) 
Layer Normalization 
Layer NormalizationLayer NormalizationLayer Normalization(64,32 5001) (5001) 
(64,49,512) 

Automated Captions –Transformers –Resnet50 + Flickr8k Automated Captions –Transformers –Resnet50 + Flickr8k 

Automated Captions–Transformers-InceptionV3 + Flickr8k Automated Captions–Transformers-InceptionV3 + Flickr8k 

Automated Captions –Transformers –Resnet50 + Flickr30k Automated Captions –Transformers –Resnet50 + Flickr30k 

Results Snapshot view for various models Results Snapshot view for various models 
Model Image Feature extraction Hyper Parameters and training details Data Set Results ( Data captured after running on 250 random Test Images) 
Base Model -LSTM 
Inception V3 Loss -Cross Entropy 
Optimizer – 
Adam 
Epochs – 
45 
Flickr8 
Flickr8 
BLEU-3 count: 46 
BLEU-3 count: 61 
BLEU-3 count average 0.21 
BLEU-3 count average 0.2875 
Resnet50 Learning rate – 
0.0001 
Training – 
8000 images 
Flickr8 BLEU-1 count: 176 BLEU-1 Average: 0.2559 
Attention 
VGG16 
Resnet50 
Loss -SparseCategoricalCrossentropy 
Optimizer – 
Adam 
Epochs – 
45 
Learning rate – 
0.0001 
Training – 
8000 images 
Flickr8 
BLEU-2 count: 71 
BLEU-3 count: 26 
BLEU-4 count: 4 
BLEU-1 count: 178 
BLEU-2 count: 89 
BLEU-3 count: 35 
BLEU-2 Average: 0.2275 
BLEU-3 Average: 0.2541 
BLEU-4 Average: 0.2975 
BLEU-1 Average: 0.2773 
BLEU-2 Average: 0.2501 
BLEU-3 Average: 0.2901 
BLEU-4 count: 14 BLEU-4 Average: 0.2608 
Flickr8 BLEU-1 count: 195 BLEU-1 Average: 0.3115 
Inception V3 Loss -SparseCategoricalCrossentropy 
Optimizer – 
Adam 
Epochs – 
70 
Training – 
8000 images 
Multi Head -8 
Flickr8 
BLEU-2 count: 103 
BLEU-3 count: 44 
BLEU-4 count: 10 
BLEU-1 count: 200 
BLEU-2 count: 100 
BLEU-2 Average: 0.2738 
BLEU-3 Average: 0.3011 
BLEU-4 Average: 0.2824 
BLEU-1 Average: 0.2798 
BLEU-2 Average: 0.2540 
Transformers BLEU-3 count: 40 BLEU-3 Average: 0.2973 
Resnet50 
BLEU-4 count: 12 BLEU-4 Average: 0.2985 
Loss -SparseCategoricalCrossentropy Flickr30 BLEU-1 count: 177 BLEU-1 Average: 0.2516 
Optimizer – 
Adam BLEU-2 count: 82 BLEU-2 Average: 0.2232 
Epochs – 
35 BLEU-3 count: 22 BLEU-3 Average: 0.2901 
Training – 
25000 images BLEU-4 count: 7 BLEU-4 Average: 0.2837 
Multi Head – 
8 


Caption Generation –UI Integration( Fast API) 
UI was integrated for Basemodel LSTM implementation 
Caption Generation –UI Integration( Fast API) 
UI was integrated for Basemodel LSTM implementation 

Learnings and Next steps 
Next Steps 
Learnings 
Learnings and Next steps 
Next Steps 
Learnings 
To look for better validation model than Bleu score 

• 
FLICKR30 size was huge and upload to collab was a 
challenge. It was partitioned into multiple sets and 
uploaded in Google drive 
• 
UI Integration with Transformer code was delayed 
• 
Performance issues with training Transformers over 
Flickr30 as it error out with Allocated memory 
issues ..etc – 
To implement training in parts by using 
the generated weights from previous epochs 
• 
Tuning Hyper Parameters 
• 
Run for 100 + Epochs for Transformers ( iteratively using weights 
from previous iterations) 
• 
UI integration + Cloud deployment -Transformer Code 
Training(s) 

• 
Transformers + Flickr30 
• 
Attention + Flickr30 
Other (Word Embeddings) we can try 

• 
Word2Vec 
• 
Glove 
• 
BERT 
Data 

• 
Flickr30 

Thank You!! Thank You!! 

Model -LSTM Model -LSTM 
LSTM Gates 
Input gate 
Forget gate 
Output gate 
Context 
Word prediction 
Probability 
End to End 
Image Vector from CNN 
Word Embedding 
Loss function 

Architecture –Keras Implementation Architecture –Keras Implementation 
inputs1 = Input(shape=(2048,)) 
fe1 = Dropout(0.5)(inputs1) 
fe2 = Dense(256, activation='relu')(fe1) 
inputs2 = Input(shape=(max_length,)) 
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2) 
se2 = Dropout(0.5)(se1) 
se3 = LSTM(256)(se2) 
decoder1 = add([fe2, se3]) 
decoder2 = Dense(256, activation='relu')(decoder1) 
outputs = Dense(vocab_size, activation='softmax')(decoder2) 
model = Model(inputs=[inputs1, inputs2], outputs=outputs) 



