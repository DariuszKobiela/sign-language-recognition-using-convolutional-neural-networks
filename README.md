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

## 3.3. Mobile Application Testing the Network
The application allow users to select the network, delegate, and number of threads for inference. The camera continuously capture frames and provide them to the loaded model. Before delivering frames to the network, the images must be appropriately processed. This process includes segmentation to separate the hand from the background by converting the color space from RGB to HSV, a method that effectively distinguishes skin tones. The segmented image is then converted to grayscale and resized to fit the input layer of the neural network. The installed network makes predictions and provide feedback to the user in the form of text with the percentage prediction result.

[ADD: The details about the usage, testing and implementation of the Mobile Application]

## 3.4. Methods of Network Optimization
A detailed description of the used optimization methods

## References
<a id="1">[1]</a> Suharjito, Suharjito & Gunawan, Herman & Thiracitta, Narada & Nugroho, Ariadi. (2018). Sign Language Recognition Using Modified Convolutional Neural Network Model. 1-5. 10.1109/INAPR.2018.8627014. 

<a id="2">[2]</a> Makarov, Ilya & Veldyaykin, Nikolay & Chertkov, Maxim & Pokoev, Aleksei. (2019). American and russian sign language dactyl recognition. PETRA '19: Proceedings of the 12th ACM International Conference on PErvasive Technologies Related to Assistive Environments. 204-210. 10.1145/3316782.3316786. 
