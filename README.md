# sign-language-recognition-using-convolutional-neural-networks
This repository is the extension of the article [CITE] and contains the detailed report about the literature review and details about the usage, testing and
implementation of the mobile application. 

## Code
> Mobile application developed for the research: https://github.com/adartemiuk/ASL_recognition_app

> Models source code and training scripts: https://github.com/adartemiuk/ASL_recognition_models

## 2. Background and related work
wrzucić TABELKI 2.1 oraz 2.2 z magisterki (Wykaz modeli realizujących zadania rozpoznawania gestów oraz Wykaz najczęściej wykorzystywanych zbiorów danych do treningu modeli rozpoznających gesty). Dorzucić też podsummowanie przeglądu literatury (na co najmniej 2 strony - streszczenie przeglądu literatury z magisterki)

### List of compared models performing gesture recognition tasks: 

| Author  | Model | Gesture type | Sign language type | Accuracy |
|------------- | ------------- | ------------- |  ------------- |  ------------- | 
| Suharjito et al. [[1]](#1) | i3D Inception  | dynamic |  Argentinean |  100% |
| Makarov et al. [[2]](#2) | QuadroConvPoolNet  | static |  American  |  about 100% |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell |

### List of the most frequently used datasets for training sign recognition models

| Dataset  | Source | Gesture type | Sign language type | Number of gestures | Number of samples |
|------------- | ------------- | ------------- |  ------------- |  ------------- | ------------- | 
| MNIST | Kaggle [44]  | static |  American |  24 | 34,637 |
| ASL Alphabet |Kaggle [45]  | static |  American  |  29 | 87,029 |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell | Content Cell |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell | Content Cell |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell | Content Cell |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell | Content Cell |

## 3.2. Model
[ADD: details of final model - finally consisting of 15 convolutional layers and multiple other types of layers]

|Layer (type)     |            Output Shape        |      Param #   |
|------------- | ------------- | ------------- |
|conv2d_40 (Conv2D)      |     (None, 112, 112, 96)  |    11712     |
|max_pooling2d_22 (MaxPooling) | (None, 56, 56, 96)   |     0       |  
|conv2d_41 (Conv2D)    |       (None, 56, 56, 128)  |     307328    |
|conv2d_42 (Conv2D)     |      (None, 56, 56, 128)   |    409728    |
|max_pooling2d_23 (MaxPooling) | (None, 28, 28, 128)  |     0       |  
|conv2d_43 (Conv2D)    |       (None, 28, 28, 256)   |    295168   | 
|conv2d_44 (Conv2D)     |      (None, 28, 28, 256)    |   590080   | 
|conv2d_45 (Conv2D)      |     (None, 28, 28, 256)   |    590080   | 
|max_pooling2d_24 (MaxPooling) | (None, 14, 14, 256)  |     0       |  
|conv2d_46 (Conv2D)     |      (None, 14, 14, 256)   |    590080    |
|conv2d_47 (Conv2D)    |       (None, 14, 14, 256)   |    590080    |
|conv2d_48 (Conv2D)     |      (None, 14, 14, 256)   |    590080    |
|max_pooling2d_25 (MaxPooling) | (None, 7, 7, 256)   |      0       |  
|conv2d_49 (Conv2D)    |       (None, 7, 7, 512)     |    1180160   |
|conv2d_50 (Conv2D)    |       (None, 7, 7, 512)     |    2359808   |
|conv2d_51 (Conv2D)      |     (None, 7, 7, 512)     |    2359808  | 
|max_pooling2d_26 (MaxPooling) | (None, 4, 4, 512)    |     0      |   
|conv2d_52 (Conv2D)      |     (None, 4, 4, 1024)    |    4719616  | 
|conv2d_53 (Conv2D)       |    (None, 4, 4, 1024)    |    9438208   |
|conv2d_54 (Conv2D)       |    (None, 4, 4, 1024)    |    9438208   |
|max_pooling2d_27 (MaxPooling) | (None, 2, 2, 1024)   |     0       |  
|flatten_4 (Flatten)     |     (None, 4096)        |      0         |
|dense_3 (Dense)         |     (None, 1024)        |      4195328   |
|dropout_1 (Dropout)    |      (None, 1024)        |      0         |
|dense_4 (Dense)        |      (None, 37)          |      37925     |
|------------- | ------------- | ------------- |
|Total params: |37,703,397| |
|Trainable params: | 37,703,397 | |
|Non-trainable params:| 0 | |
|------------- | ------------- | ------------- |

## 3.3. Mobile Application Testing the Network
The application allow users to select the network, delegate, and number of threads for inference. The camera continuously capture frames and provide them to the loaded model. Before delivering frames to the network, the images must be appropriately processed. This process includes segmentation to separate the hand from the background by converting the color space from RGB to HSV, a method that effectively distinguishes skin tones. The segmented image is then converted to grayscale and resized to fit the input layer of the neural network. The installed network makes predictions and provide feedback to the user in the form of text with the percentage prediction result.

[ADD: The details about the usage, testing and implementation of the Mobile Application]

## 3.4. Methods of Network Optimization
A detailed description of the used optimization methods

### Network quantization
Network quantization involves reducing the precision of parameters and intermediate activation maps that are typically stored in floating-point notation. Gholami et al. [[3]](#3) indicate that the first step is to define a quantization function that maps the real values of weights or activations to lower precision values. Usually, such a function maps these values to integer values, according to the following formula:

$Q(r)=Int(\frac{r}{S})$

where: \
Q(r) - quantization value\
Int - integer value\
r - real input value\
S - scaling factor

The above formula refers to symmetric quantization, where values are clipped using a symmetric range of values [-a, a]. Two main approaches in quantization can be distinguished: QAT (Quantization Aware Training) and PTQ (Post Training Quantization). In the first approach, quantization is performed during network training, where parameters are quantized after each gradient update. A major drawback, as indicated by the authors, is the computational cost of retraining the neural network. The second approach, PTQ, does not require retraining the network, as parameters are quantized after the network has been trained. It is a faster and simpler approach compared to QAT, but the model may suffer more in terms of detection accuracy.

###  Knowledge Distillation
Another approach of network optimization is Knowledge Distillation, which involves transferring information from a larger, more complex model to a less complex network. This method is often described in articles as a teacher-student format, where the teacher (the more complex network) imparts its knowledge to the student (the less complex network). Zhang et al. [2019] describe that the information flow is facilitated through a second (intermediate) network using data specifically labeled by the previous network. By utilizing synthetic data, the risk of overfitting the network is reduced, and very good function approximation is ensured. Moreover, this approach enables the compression and acceleration of complex networks.

###  Layer Decomposition
D


## References
<a id="1">[1]</a> Suharjito, Suharjito & Gunawan, Herman & Thiracitta, Narada & Nugroho, Ariadi. (2018). Sign Language Recognition Using Modified Convolutional Neural Network Model. 1-5. 10.1109/INAPR.2018.8627014. 

<a id="2">[2]</a> Makarov, Ilya & Veldyaykin, Nikolay & Chertkov, Maxim & Pokoev, Aleksei. (2019). American and russian sign language dactyl recognition. PETRA '19: Proceedings of the 12th ACM International Conference on PErvasive Technologies Related to Assistive Environments. 204-210. 10.1145/3316782.3316786. 

<a id="3">[3]</a> Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M. W., Keutzer, K.: A survey of quantization methods for efficient neural network inference. In arXiv preprint arXiv:2103.13630, 2021. 
