# VIAPF: Vocal Isolation and Profanity Filtering
## Deep Learning for Vocal Isolation and Profanity Filtering

**Yusuf Morsi**  
Department of Electrical and Computer Engineering  
University of California, San Diego  
La Jolla, CA 92093  
ymorsi@ucsd.edu

## File Structure

- **README.md**
- **SUCCESSFUL.ipynb** - Notebook including vocal isolation code using Open-UnMix and profanity detection code using Google Cloud
- **FAILURE.ipynb** - Initial autoencoder model that was scrapped
- **CSE190.pdf** - LaTeX report
- **images/** - Contains project diagrams and visualizations

## Abstract

This report discusses my project, in which I take in audio tracks in order to isolate vocals and mute profanity. Returned tracks include the isolated vocals, the isolated vocals with the profanity muted, and the original track without profanity. The main goal of this project is to transform a song with instruments and inappropriate vocals in order to make it more user-friendly. Some avoid listening to music with instruments due to them causing headaches, so this is a solution that heeds that issue. When it comes to profanity in music, a more obvious problem, some prefer songs without curse words. This can be done either by businesses who want to play child-friendly music or parents that do not want their children listening to profanity. This project uses advanced audio processing and machine learning techniques. The final solution uses Open-Unmix for vocal isolation and Google Speech-to-Text for profanity detection (Pydub is also used to mute curse words).

## 1. Introduction

The music industry is currently at a stage where new technologies are constantly being introduced to manipulate audio. This project focuses on processing music tracks to isolate musical instruments and remove profanity. The primary goal is to take a song with instrumental, vocal, and profanity elements and produce an output that not only removes the unwanted elements but also returns in a clean, low-noise manner. This profanity-removal aspect has numerous applications, including creating clean versions of songs for radio stations, businesses, or personal use.

The project implements advanced ML and audio-processing techniques in order to achieve its objectives. This mainly involves using deep learning models to separate vocals from instrumentals, followed by the application of speech recognition algorithms (namely Google Cloud Speech-to-Text) to detect and mute profane words. Two different model architectures were explored: one using a state-of-the-art pre-trained model that is able to successfully isolate vocals, and another that applies an autoencoder-based neural network which was unfortunately less effective when it comes to runtime, efficiency, and noise. The challenges, methodologies, and outcomes of these approaches are discussed in detail.

## 2. Related Works

Vocal isolation has been a significant area of research in audio signal processing. There are multiple models that compete in vocal isolation, the best being BandSplit RNN, and follow-ups including Hybrid Demucs, DEMUCS, Conv-TasNET. Open-Unmix, the model we used, ranks as number 15 in overall SDR (score to distortion).

## 3. Architecture

### 3.1 Open-Unmix Model

<img src="/api/placeholder/800/400" alt="Architecture of the Open-Unmix Model" />

The Open-Unmix (UMX) model is a highly ranked and used model designed for music source separation (bass, drums, and vocals). It operates through multiple steps:

- **Spectrogram Transformation:** First, the input is converted into spectrogram format with STFT.

- **Neural Network:**
  - **Encoder:** As expected of a neural network, we have an encoder. This model uses Bi-LSTM networks to find temporal movement and features from the spectrogram.
  - **Bottleneck:** As expected once again, we have a bottleneck, which reduces the dimensionality while still keeping important features.
  - **Decoder:** We now have a decoder that reconstructions the spectrogram, and allows us to have an output to investigate.

- **Training:** This model is trained on the MUSDB18HQ dataset, which has 150 music tracks, all of which are isolated. The goal here is to minimize MSE between the predictions and real spectrograms.

### 3.2 Custom Autoencoder-Based Neural Network

<img src="/api/placeholder/800/400" alt="Architecture of the Autoencoder Model" />

Our autoencoder implements a convolutional neural network to isolate the vocals, also using the MUSDB18HQ dataset. Our architecture mainly consists of two components that we have seen in Open Unmix's model: the encoder and the decoder.

#### 3.2.1 Encoder

The encoder compresses the input audio spectrogram into a lower-dimensional representation, capturing the essential features needed for reconstruction. The encoder consists of three sequential convolutional blocks, each with:

- **Conv2D Layers:** 2D convolutional with 3x3 filters (32, 64, 128 filters)
- **LeakyReLU:** After each layer, a LeakyReLU activation function is implemented so more hectic patterns are recognized
- **Batch Normalization:** Batch normalization layers are used to normalize the inputs to each layer
- **MaxPooling and Dropout:** After each block, we have a max pooling layer (2x2), which is used to reduce the spatial dimensions. The dropout layer (0.25) is implemented to prevent overfitting

Layer Structure:
- **Layers 1 & 2:** Conv2D (32 filters, 3x3) → LeakyReLU → BatchNorm
- **Layers 3 & 4:** Conv2D (64 filters, 3x3) → LeakyReLU → BatchNorm
- **Layers 5 & 6:** Conv2D (128 filters, 3x3) → LeakyReLU → BatchNorm

#### 3.2.2 Decoder

The decoder mirrors the encoder's structure but in reverse order, expanding the data back to its original dimensions:

Layer Structure:
- **Layers 7 & 8:** Conv2D (128 filters, 3x3) → LeakyReLU → BatchNorm
- **Layers 9 & 10:** Conv2D (64 filters, 3x3) → LeakyReLU → BatchNorm
- **Layers 11 & 12:** Conv2D (32 filters, 3x3) → LeakyReLU → BatchNorm

## 4. Experiments

### 4.1 Dataset and Preprocessing

The MUSDB18-HQ dataset was used for our training, which has high-quality tracks with different stems for vocals, bass, drums, and other. The data was split into training and test sets, and covered a multitude of genres.

### 4.2 Model Training and Evaluation

#### 4.2.1 Open-Unmix Model

- **Training:** Pretrained on the Musdb18 dataset
- **Evaluation:** The model was evaluated using Signal-to-Distortion Ratio (SDR), Signal-to-Interference Ratio (SIR), and subjective listening tests

#### 4.2.2 Custom Autoencoder-based Model

- **Training:** The model was trained from scratch using the same dataset
- **Evaluation:** Performance issues were noted due to noise and pauses

<img src="/api/placeholder/800/400" alt="Autoencoder Output Spectrogram showing noise" />

## 5. Results

### 5.1 Vocal Isolation Performance

The models were tested on several songs, including 'I Knew You Were Trouble', 'Dirty Diana', 'Bohemian Rhapsody', and 'Maps'. Here are the spectrograms showing the results:

<img src="/api/placeholder/800/400" alt="Spectrogram of 'I Knew You Were Trouble' - Open-Unmix" />

*Figure 4: 'I Knew You Were Trouble' spectrogram comparison (MSE: 2.0109)*

<img src="/api/placeholder/800/400" alt="Spectrogram of 'Dirty Diana' - Open-Unmix" />

*Figure 5: 'Dirty Diana' spectrogram comparison (MSE: 1.9992)*

<img src="/api/placeholder/800/400" alt="Spectrogram of 'Bohemian Rhapsody' - Open-Unmix" />

*Figure 6: 'Bohemian Rhapsody' spectrogram comparison (MSE: 2.0095)*

<img src="/api/placeholder/800/400" alt="Spectrogram of 'Maps' - Open-Unmix" />

*Figure 7: 'Maps' spectrogram comparison*

### 5.2 Profanity Detection Performance

The system's effectiveness in detecting and muting profanity showed mixed results:

- **Transcription Accuracy:** 27.91%
- **Percent Profanity Removed:** 20.00%

While the numerical metrics show room for improvement, the audio output demonstrated effective muting of most curse words in practice.

## 6. Conclusion

This project demonstrates the importance of proper neural network architecture in audio processing tasks. The pre-trained Open-Unmix model proved significantly more effective than our custom autoencoder approach, highlighting the value of well-constructed neural network architecture in achieving desired results.
