# Yoga Recommendation

This repository provides a machine learning-based approach to recommend personalized yoga exercises and routines. Below is the detailed documentation outlining the approach, data preprocessing, model architecture, results, and next steps.

---

## Table of Contents
1. [Approach](#approach)
2. [Data Selection](#data-selection)
3. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Results](#results)
7. [Next Steps](#next-steps)

---

## Approach
This project aims to build a recommender system that will recommend yoga exercises to users depending on their current emotional state. The users will have the opportunity of selecting their mood and the system will generate appropriate yoga exercises for them. The approach utilizes a bidirectional LSTM (Long Short-Term Memory) recurrent neural network since the model is able to effectively remember the contexts of the words before and after the words when interpreting moods to provide relevant recommendations. Once the emotional state is identified, a Python dictionary module is used to map each mood to specific yoga exercises, ensuring the recommendations are both relevant and personalized

---
## Data Selection
In this project, the dataset created is 15,000 records and is completely fictitious and was created using ChatGPT. This has been done with great care, so as to overcome different challenges, and still sound human-like with more natural tones. Features of the dataset include:

- **Realistic Examples**: The dataset contains inputs of real-world situations where many emotions are intertwined or some slang is used to make it real.

- **Variations and Diversity**: It adds a few spelling mistakes, uses some synonyms, and employs reworded versions of already existing examples for diversity and strength.

- **Balanced distribution**: A fair number of examples are provided for every mood so that balance of classes is not skewed and training is competitive.

- **Noise introduction**: Some amount of minor noise was also introduced like small random changes and imperfections to simulate better real user input to the data.


---

## Data Preprocessing

 1.**Handling Missing Data**:

 - Check for any missing or null values in the dataset and handle them appropriately.
   - Identify and address inconsistencies or gaps in the data, particularly in critical features like the `Mood` attribute.

2. **Identifying Unique Categories**: 
   - Extract all unique categories present in the `Mood` feature to understand the range of moods represented in the dataset.

3. **Text Preprocessing Using NLTK**:
   - Normalize the text by converting all characters to lowercase to maintain consistency.
   - Use NLTK's stopword list to remove irrelevant or unnecessary words that do not contribute to the context(eg the,is,etc).This helps focus on meaningful words that are essential for analysis.

### 3. Feature Engineering:
In the feature engineering process, I applied one-hot encoding to the Mood category. This technique transforms each unique mood into a separate binary feature, where:

A value of 1 indicates the presence of that mood.
A value of 0 indicates its absence.
After applying one-hot encoding, a new dataframe is created containing these binary-encoded features along with their corresponding column names

### 4. Preprocessing Example sentences(user input)

The next step in preprocessing focuses on handling the example sentences (user inputs). Here, I utilized TensorFlow's tokenizer for this process. The tokenizer performs the following tasks:

- **Tokenization**: It splits the words in each sentence into individual tokens.
- **Unique ID Assignment**: Each token is assigned a unique numerical ID, making the data suitable for machine learning or deep learning models, as these models work with numerical inputs rather than text.

The rationale behind using TensorFlow's tokenizer instead of basic NLTK-based tokenization (like `word_tokenize`) is that while NLTK can split sentences into tokens, it does not assign numerical IDs to the tokens. TensorFlow's tokenizer bridges this gap by mapping each token to a unique ID based on the specified vocabulary size (`vocab_size`). Additionally, similar words across sentences may be assigned similar IDs, enabling the model to recognize patterns and relationships more effectively.

### 5.Padding Sentences
After tokenizing the sentences, I applied padding to ensure all sentences have a uniform length. This step addresses the variability in sentence lengths within the dataset by standardizing them to a fixed size. Here's the process:

Finding the Maximum Sentence Length: The maximum length of the sentences in the dataset is determined. This serves as the reference length for padding.
Applying Padding: Using TensorFlow's pad_sequences method, each sentence is padded with 0 values in the empty spaces. Padding can be applied to either the beginning (pre) or the end (post) of the sentence, depending on the configuration.
The primary purpose of padding is to ensure that all sequences are of equal length, as machine learning and deep learning models require input data to have consistent dimensions.

---

## Model Architecture
- **Neural Network Layers**:
  - **Embedding Layer**: During preprocessing, we tokenized the sentences using TensorFlow's tokenizer, which converts words into unique numerical IDs. Although these IDs let the model process the input, they don't carry any semantic meaning or capture relationships between words. For instance, the IDs for "king" and "queen" could be completely unrelated numerically.
   The embedding layer takes the input sequences (padded numerical IDs) and maps them into vectors of fixed dimensions.input_dim: This means the size of the vocabulary-that is, the total unique number of words or tokens existing in the dataset.output_dim: This is the dimension of the dense vector representation for each word.
  - Hidden layers: Fully connected layers with ReLU activation to capture nonlinear relationships.
  - Output layer: Generates a list of recommended yoga poses.  So intial Embedding for each word is (W,128) Where W is the total number of words and each word is having a dimensions of 128
- **Bidirectional LSTM** 
    -After the embedding layer, the model uses a Bidirectional LSTM (Long Short-Term Memory) RNN to process the sequential data. This specific type of model is beneficial for capturing context from both past and future words in a sentence. Here Since is captures dependencies in both forward abd backward directionand the dimension which we have set is 512. So each word  will have 1024 dimension beacuse (512)forward+(512)backward
  -  **Dense Layer**:
  - Initial Dense Layer parameters(W,128) this helps us to reduce dimensionaity while retaining meaningful features
  - Next Dense Layer Parameter(W,64) further reducing the dimension for compact representation uisng ReLu activation in the both the cases to introduce non linearity
  - Final Dense Layer Parameter (y.shape[1],Softmax)  here the y.shape[1] is the mood which we have to predict by using softmax it turn each value into probability for each class such that when we sum these values it turns out to be 1. So the idea here is to find the maximum prob class.
  - **Parameters Used**:
  - **Optimizer used**- Adam
  - **Loss**-Categorical Cross Entropy Since the problem statement is multi class classification
  - **Metrics**-Accuracy
  - **Early Stopping**-The idea behind early stopping is to prevent the model from overfitting by monitoring a specific metric (like validation loss or validation accuracy) during training.
If the metric does not improve over a specified number of epochs (determined by the patience parameter), training is stopped early

### Next Step:
- The next step is to map each mood to its corresponding yoga routine. This way, when we create the prediction function, it will first determine the user's mood and then suggest a suitable yoga exercise based on that mood. This two-step process ensures personalized and relevant recommendations for the user.
- mood_yoga_mapping = {
    "Happy": "Vinyasa Yoga",
    "Sad": "Yin Yoga",
    "Stressed": "Restorative Yoga",
    "Relaxed": "Meditation",
  }

---

## Results
![image](https://github.com/user-attachments/assets/e0054d10-c057-40c8-9d16-7a2544cfb1e9)

### 1. Evaluation Metrics:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- Macro Average

### 2.Model Performance Summary
Accuracy: 0.79 (79%)

The model demonstrates a balanced performance across different mood classes, with an overall accuracy of 79%. 
Macro Average:

Precision: 0.80
Recall: 0.79
F1-Score: 0.79
Weighted Average:

Precision: 0.80
Recall: 0.79
F1-Score: 0.79

### 3. Insights:
- The perfect accuracy and performance across all metrics are due to the use of synthetic data for training and evaluation. Synthetic data is typically clean, well-labeled, and free from real-world noise or ambiguity. This creates an ideal environment for the model, resulting in flawless performance.
---
## Next Step
1. Switch to a Transformer-based Model
Instead of using a Bidirectional LSTM, we can go for a transformer-based model like an Encoder-Decoder architecture. Models like BERT, GPT, or even newer ones like LLaMA are way better for tasks like sentiment analysis. Why? Because they don’t just process sequences like LSTMs—they actually look at the whole sentence at once using attention mechanisms. Plus, they’re super flexible and can be fine-tuned for specific tasks, making them much more powerful for something like analyzing sentiment.

2. Upgrade to Sentence-level Embeddings
The embeddings we usually get from TensorFlow’s tokenization or similar tools are word-level. While they do a decent job, they miss out on capturing the context of the entire sentence. Switching to sentence-level embeddings from models like Hugging Face’s LLaMA or OpenAI’s embeddings would be a game-changer. These embeddings aren’t just about words—they understand the meaning of the whole sentence, which is crucial for tasks like sentiment analysis.

3. Use Real-world Data
If we’re training a sentiment analysis model, it’s better to move away from clean, synthetic datasets and use real-world data instead. Real-world data is messy, sure, but that’s exactly what makes it valuable—it reflects how people actually communicate. Adding a step to clean and rephrase the data in a human tone can make it even better. This helps the model handle all the quirks and nuances of real-world inputs, like slang, typos, and different writing styles.



