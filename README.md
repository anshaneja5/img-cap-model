# Image Captioning using DenseNet201 and LSTM

This project implements an image captioning system using a combination of DenseNet201 for image feature extraction and LSTM for caption generation. The model is trained on the Flickr30k dataset and uses a custom data generator to handle the image-caption pairs. The project involves the entire pipeline from image preprocessing, text tokenization, model training, and inference with BLEU score evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)

## Project Overview
The goal of this project is to automatically generate captions for images. The model extracts image features using DenseNet201 and generates a sequence of words (captions) using an LSTM model. 

Key features of the project:
- Image feature extraction using a pre-trained DenseNet201.
- Tokenization and padding for text processing.
- A custom data generator for efficient handling of image-caption pairs.
- Model evaluation using BLEU score.

## Dataset
We use the **Flickr30k** dataset, which consists of 30,000 images and 5 captions per image. 

### Data Preprocessing
1. **Image Preprocessing**: Images are resized to 224x224 pixels and normalized to the range [0, 1].
2. **Text Preprocessing**: 
    - Convert captions to lowercase.
    - Remove non-alphabetic characters and short words.
    - Add start (`startseq`) and end (`endseq`) tokens to each caption.

## Model Architecture

### Image Feature Extraction
We use **DenseNet201** to extract image features from the images. DenseNet201 is pre-trained on ImageNet and provides high-level features for each image.

### Caption Generation
The image features are passed through a Dense layer and reshaped. Captions are tokenized and passed through an LSTM. The model is structured as follows:
1. Image features are concatenated with sentence embeddings.
2. The concatenated features are fed into an LSTM layer.
3. The output is passed through Dense layers to generate a probability distribution over the vocabulary.

### Model Summary
```bash
____________________________________________________________________________________________
Layer (type)                       Output Shape           Param #     Connected to
================================================================================================
input_1 (InputLayer)               [(None, 1920)]         0           input_1[0][0]               
input_2 (InputLayer)               [(None, max_length)]   0           input_2[0][0]               
____________________________________________________________________________________________
Dense_1 (Dense)                    (None, 256)            491776      input_1[0][0]               
Reshape_1 (Reshape)                (None, 1, 256)         0           Dense_1[0][0]              
Embedding_1 (Embedding)            (None, max_length, 256)  vocab_size input_2[0][0]
Concatenate_1 (Concatenate)        (None, max_length + 1, 256) 0       Reshape_1[0][0]
LSTM_1 (LSTM)                      (None, 256)            525312      Concatenate_1[0][0]         
Dense_2 (Dense)                    (None, 128)            32896       LSTM_1[0][0]               
Dense_3 (Dense)                    (None, vocab_size)     vocab_size   Dense_2[0][0]
____________________________________________________________________________________________

```
## Training

The model is trained using the following:

* **Loss Function**: Categorical cross-entropy.
* **Optimizer**: Adam.
* **Callbacks**:
    * ModelCheckpoint: Save the best model.
    * EarlyStopping: Stop training if no improvement is seen.
    * ReduceLROnPlateau: Reduce learning rate on plateau.

The training is done for 5 epochs using a custom `CustomDataGenerator` for handling the image-caption pairs.

## Evaluation

The model is evaluated using the **BLEU score**, which measures the similarity between the generated captions and actual captions.

### BLEU Score

The BLEU score for this model on the test set is as follows: 0.79

Here is the preview of the model : 

![image](https://github.com/user-attachments/assets/a59e33f9-8978-4a8e-be9b-1c13c7330cd9)
![image](https://github.com/user-attachments/assets/319382f7-489a-4ae2-a247-a8e57df6772e)

Future Improvements : The Bleu score can be improved by training on a larger dataset or by using attention mechanism. I am planning to add attention in this project in future as some predictions are still off, like if its a man, its always predicting man in red shirt, so will add attention in future. 
