# sign-language-recognition-using-convolutional-neural-networks
This repository is the extension of the article [CITE] and contains the detailed report about the literature review and details about the usage, testing and
implementation of the mobile application. 

## Code
> Mobile application developed for the research: https://github.com/adartemiuk/ASL_recognition_app

> Models source code and training scripts: https://github.com/adartemiuk/ASL_recognition_models

## 2. Background and related work
wrzucić TABELKI 2.1 oraz 2.2 z magisterki (Wykaz modeli realizujących zadania rozpoznawania gestów oraz Wykaz najczęściej wykorzystywanych zbiorów danych do treningu modeli rozpoznających gesty). Dorzucić też podsummowanie przeglądu literatury (na co najmniej 2 strony - streszczenie przeglądu literatury z magisterki)

## 3.3. Mobile Application Testing the Network
The application allow users to select the network, delegate, and number of threads for inference. The camera continuously capture frames and provide them to the loaded model. Before delivering frames to the network, the images must be appropriately processed. This process includes segmentation to separate the hand from the background by converting the color space from RGB to HSV, a method that effectively distinguishes skin tones. The segmented image is then converted to grayscale and resized to fit the input layer of the neural network. The installed network makes predictions and provide feedback to the user in the form of text with the percentage prediction result.

[ADD: The details about the usage, testing and implementation of the Mobile Application]

## 3.4. Methods of Network Optimization
A detailed description of the used optimization methods
