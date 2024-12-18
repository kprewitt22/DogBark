# DogBark
Classification of Dog Barks via Supervised Learning

## Overview
For my final project, I developed a machine learning program to classify dog barks using Convolutional Neural Networks (CNNs). Classifying audio sounds can be complex, but advancements in neural networks and audio preprocessing have simplified the process. By leveraging supervised learning, I can efficiently classify audio files, particularly distinguishing dog barks from non-barks.

## Approach
Supervised Learning for Audio Classification
Supervised learning works best for classification tasks, especially when processing, storing, and labeling audio files. I downloaded and labeled a pre-compiled dataset of environmental sounds and random noises as either "barking" (1) or "non-barking" (0). For training, I used two datasets:

### #ESC-50: Dataset for Environmental Sound Classification (Piczak, 2015)
### Speak Like a Dog: Human to Non-human Creature Voice Conversion (Suzuki, Sakamoto, Taniguchi, & Kameoka, 2022)
The audio was converted into spectrograms to graphically evaluate the sounds, plotting frequency variation over time. This allows us to capture unique features of the sound, making it easier for the CNN to classify. Spectrograms are often used in speech recognition to detect signal changes.

## Data Preprocessing
Using Librosa, an AI framework for audio processing, I converted audio into mono-channel spectrograms using Short-Time Fourier Transform (STFT). The preprocessing pipeline is shown below:

### Audio Input: Raw audio is loaded and processed into a spectrogram.
Spectrogram Creation: The audio is transformed into a spectrogram, which is a visual representation of sound.
### Serialization: The processed spectrogram is serialized into TensorFlow Record (TFRecord) files for training. However, due to image compression issues, uncompressed images (512x512 pixels) were ultimately used for training.
## Model Architecture
The CNN consists of 9 layers designed to extract increasingly complex features from the spectrograms:

###Input Layer: Accepts 512x512 spectrogram images.
### Convolutional Layers: Filters with sizes of 32, 64, and 128.
### Pooling Layers: MaxPooling layers to reduce dimensionality.
### Dense Layer: A fully connected layer with 30% more neurons to prevent overfitting.
### Dropout Layer: Dropout of 30% to prevent overfitting, particularly since the model showed bias toward non-barking sounds.
### Output Layer: A sigmoid activation function to classify audio as either barking (1) or non-barking (0).
The model uses the binary cross-entropy loss function, ideal for binary classification tasks.

## Training & Results
The model was trained for 40 epochs with a batch size of 16. On my machine, this training process took approximately three hours per iteration, each time a change was made to the model.

## Training vs Validation Accuracy: While the model showed high validation accuracy, there was some overfitting observed. This is common when dropout is used, as noted in TensorFlow's Keras documentation.
Accuracy: The model achieved 99.88% accuracy on the provided dataset.
Figures
Figure 1: Spectrogram of a local dog barking.
Figure 2: The preprocessing pipeline from audio to TensorFlow model input.
Figures 3 & 4: Training vs Validation accuracy and loss.
Figure 5: Evaluation of unknown audio.
Figure 6: Model evaluation on test data.
## Challenges & Future Improvements
While the model performed well with clear audio, it struggled with background or subtle barks, especially in noisy environments. To improve this, I plan to expand the model’s architecture, adding more layers to capture higher-level features and better distinguish subtle changes in audio signals.

## References
Keras Team. (n.d.). Why is my training loss much higher than my testing loss? Keras. Retrieved December 10, 2024, from Keras Documentation
Piczak, K. J. (2015). ESC: Dataset for environmental sound classification. In Proceedings of the 23rd ACM International Conference on Multimedia (pp. 1015–1018). ACM. DOI: 10.1145/2733373.2806390
Suzuki, K., Sakamoto, S., Taniguchi, T., & Kameoka, H. (2022). Speak like a dog: Human to non-human creature voice conversion. arXiv. arXiv:2206.04780
Suzuki, K., Sakamoto, S., Taniguchi, T., & Kameoka, H. (2022). Dog Dataset [Source code]. GitHub. GitHub Link
