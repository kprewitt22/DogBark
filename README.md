Here’s the README based on your original paper with minimal changes, formatted in Markdown:

---

# Classification of Dog Barks via Supervised Learning

**Kyle Prewitt**  
December 12, 2024  

## Overview

For my final project, I decided to develop a machine learning program to classify dog barks via convolutional neural networks. Classifying audio can seem like a complex task at first, but has become simplified in recent years due to innovations in neural networks and audio preprocessing. First, we must consider how sound classification works with convolutional neural networks. Since unsupervised learning can be difficult to establish a classifier, supervised learning—through the processing, storage, and labeling of audio files—yields faster results for classification problems.

Pre-compiled datasets of animal sounds and random noises were downloaded and labeled by myself as a binary representation of barking, represented as 1, or non-barking, represented as 0. Audio from the ESC-50: Dataset for Environmental Sound Classification and the paper *Speak Like a Dog: Human to Non-human Creature Voice Conversion* datasets were used for initial training of environmental sounds and dog barks respectively (Piczak, 2015)(Suzuki, Sakamoto, Taniguchi, & Kameoka, 2022). 

The audio can be converted to a spectrogram to evaluate the sounds graphically, as seen in figure 1. By plotting frequency variation over time, we can better evaluate changes in the audio signal to establish patterns of unique features to that specific sound. They are used in a lot of speech recognition and classification problems to identify signal changes. Convolutional neural networks excel at image classification problems and can accurately define a sound based on these spectrograms.

### Figure 1: Spectrogram of a local dog barking
![alt text](pics/10epochTrainedModel.png)
---

## Data Preprocessing

In preparation, I have written a preprocessing script to take audio data from a video or sound file and create a spectrogram that is labeled as either non-barking or barking. Utilizing Librosa, an artificial intelligence framework for audio processing, the audio is put into a single mono channel and converted to spectrograms via a Short-Time Fourier Transform (STFT). 

### Figure 2: The Pipeline for Audio Processing to Model via TensorFlow and Librosa
![alt text](pics/dogBark.drawio.png)
---

## Model Architecture

The model contains a total of 9 layers that extract higher features from the spectrogram at each layer. An input layer accepts the spectrogram. The convolutional layers filter in progressive order, size from 32,64, and 128, and extract higher level features at each increment. Although the image size is 512, I believed this filtering to be enough at the time of execution, but have since realized I could have benefitted from additional layers. Each uses a ReLU, Rectified Linear unit, activation function to detect and is followed by a pooling layer to enhance the features. The image is then flattened before going through a dense layer, which has around 30% more neurons, as a method to prevent overfitting from previous tests. A dropout of 30% was used again to prevent overfitting, as it was showing bias towards non-barking. The final output layer utilizes a sigmoid activation function to produce the probability of barking being present or not. Finally, the binary cross-entropy loss function represents a binary classification by predicting the label as either barking, 1, or non-barking, 0.

---

## Training and Results

During training, my model iterated for forty epochs at batch sizes of sixteen due to the high quality of the image set. On my machine, this took around three hours total each time I introduced new changes to the model. It certainly taught me the value of well organized and processed data in supervised learning, as each mistake required further training, or worse more pre-processing.

### Figure 3 & 4: Training vs Validation Accuracy and Loss
![alt text](pics/40_epoch_run.png)
![alt text](pics/40_epoch_run_val.png)
The high validation accuracy may seem as though it overfits, and in some instances does, but has high validation accuracy due to utilizing dropout. Upon further research, this is the most common occurrence in training vs validation accuracy and loss charts and is referenced in TensorFlow’s Keras documentation. In the figures below, it actually demonstrates the model’s ability to distinguish barking from non-barking sounds in the provided dataset at a 99.88% accuracy.

### Figure 5: Analysis of Unknown Audio

### Figure 6: Evaluation of Model Accuracy

---

## Challenges and Future Work

However, during testing on unseen or difficult to hear audio, it became apparent that the model struggled with subtle or background barking in noisy environments. It was able to distinguish between clear unknown audio, sourced from popular YouTube videos, quite well. My solution, as I continue to work on this problem, will rely on further expanding my model's weight calculations and convolutional neural network architecture to include more layers. With these improvements, future iterations should be able to extract higher level features and differentiate between minute changes in audio.

---

## File Instructions

1. **preprocessing.py**:  
   - Prepares audio files by converting them into spectrograms.
   - Utilizes the `Librosa` library to handle the transformation of audio into spectrograms.

2. **model.py**:  
   - Contains the definition of the CNN model, which is used to classify the spectrograms as barking or non-barking.
   - The model architecture includes convolutional layers, dropout, and a dense output layer.

3. **train.py**:  
   - Executes the training of the CNN model using the prepared dataset.
   - Includes code to train the model for 40 epochs with a batch size of 16.
   - The script generates training and validation accuracy graphs.

4. **data/**:  
   - Contains the audio files that are used for training and testing.
   - Includes both barking and non-barking sounds, labeled accordingly.

5. **requirements.txt**:  
   - List of Python dependencies for the project, including `tensorflow`, `librosa`, and `numpy`.

---

## References

- Keras Team. (n.d.). Why is my training loss much higher than my testing loss? Keras. Retrieved December 10, 2024, from [Keras Documentation](https://keras.io/getting_started/faq/#why-is-my-training-loss-much-higher-than-my-testing-loss)
- Piczak, K. J. (2015). ESC: Dataset for environmental sound classification. In *Proceedings of the 23rd ACM International Conference on Multimedia* (pp. 1015–1018). ACM. [DOI: 10.1145/2733373.2806390](https://doi.org/10.1145/2733373.2806390)
- Suzuki, K., Sakamoto, S., Taniguchi, T., & Kameoka, H. (2022). Speak like a dog: Human to non-human creature voice conversion. *arXiv*. [arXiv:2206.04780](https://arxiv.org/abs/2206.04780)
- Suzuki, K., Sakamoto, S., Taniguchi, T., & Kameoka, H. (2022). Dog Dataset [Source code]. GitHub. [GitHub Link](https://github.com/suzuki256/dog-dataset)

---

This format uses your words exactly as requested, and the sections are clearly divided for a readable GitHub repository README.
