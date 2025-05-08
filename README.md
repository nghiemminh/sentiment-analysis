# Sentiment Analysis IMDb Reviews 

This repository contains code perform sentiment analysis on IMDb movie reviews (positive or negative) using: a simple Neural Network,  Bidirectional LSTM (Long Short-Term Memory) and DistilBERT. Sentiment analysis is a natural language processing (NLP) task that involves determining the sentiment expressed in a given piece of text.

### Data 
The dataset used for this sentiment analysis task is the [IMDb movie reviews dataset](https://www.kaggle.com/datasets/pawankumargunjan/imdb-review), which consists of 50,000 labeled movie reviews (25,000 training and 25,000 testing). Both train and test set have balanced number of classes with 50-50 ratio. In the entire collection, no more than 30 reviews are allowed for any given movie because reviews for the same movie tend to have correlated ratings.  In the labeled train/test sets, a negative review has a rating <= 4 out of 10, and a positive review has a rating >= 7 out of 10. Thus reviews with more neutral ratings are not included in the train/test sets.

### Model Architecture and Results

The models are trained on train set, then validate on validation set to get the best architecture. Finally, models perform on the test set to get unbiased performance estimators.

**1. Neural Network**: Tokenized using Keras tokenizer. Trained a Neural Network with an Embedding layer, a GlobalAveragePooling, follows by 3 Dense layers and 2 Dropout. Achieved 0.5 Accuracy, 0.95 Recall, and 0.65 F1
**2. BiLSTM**: Tokenized using Keras tokenizer. Trained a network of an Embedding Layer, 2 Bidirectional LSTM layers, then follows by 2 Dense layers and 2 Droupout. Achieved 0.86 Accuracy on a balanced test set.
**3. DistilBERT**: Tokenized using the Hugging Face transformers library's DistilBERT tokenizer. Fine-tuning with 3 epochs on GPU. Achived **0.93 Accuracy** on a balanced test set.

---> DistilBERT is the best performance model.

### Dependencies
Choose the lastest versions of any of the dependencies below: 
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [seaborn](https://seaborn.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [sklearn](https://scikit-learn.org/stable/)
- [tensorflow](https://www.tensorflow.org/)
- [transformers](https://huggingface.co/docs/transformers/en/index)
- [nltk](https://www.nltk.org/)
