# 📰 Fake News Detector using NLP Techniques

## Authors
**Author 1**: [Kuba Czech](https://github.com/KubaCzech)
**Index Number**: 156035

**Author 2**: [Piotr Balewski](https://github.com/PBalewski)
**Index Number**: 156037

## Description

With misinformation spreading rapidly on social media, having lightweight tools to flag potentially fake content is crucial. This project implements a **Fake News Detection** system using **Natural Language Processing (NLP)** techniques. We explore multiple vectorization strategies and classification models to distinguish between real and fake news.

## 📂 Dataset

We use a publicly available dataset from Kaggle:\
[Fake News Detection Dataset by Emine Yetim](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data?select=News+_dataset)

Dataset consist off over 40k articles, below there is its characteristic:
![Content](images/dataset_content.jpg)

## ⚙️ Project Structure

The core pipeline involves:

1. **Preprocessing**:

   - Lowercasing
   - Punctuation removal
   - Stopword removal
   - Tokenization
   - Lemmatization

2. **Text Vectorization Approaches**:

   - **Contextual Embeddings** using **DistilBERT**
   - **Static Word Embeddings** using **GloVe 50d** vectors
   - **TF-IDF (Term Frequency-Inverse Document Frequency)**

3. **Classification Models Tested** (for each vectorization method):

   - Support Vector Machine (SVM)
   - Logistic Regression (LR)
   - k-Nearest Neighbors (kNN)
   - Decision Tree
   - Random Forest
   - Custom Feedforward Neural Network (FNN)


## 📊 Evaluation Metrics

Each model was evaluated using the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

A comprehensive comparison of all models and approaches is presented at the bottom of the final report.

## 💥 Results

### Final Accuracy Comparison

| Model              | Contextual | Static   | TF-IDF   |
|-------------------|------------|----------|----------|
| **SVM**           | 0.9475     | 0.9025   | 0.9275   |
| **Logistic Reg.** | 0.9575     | 0.9150   | 0.9200   |
| **kNN**           | 0.9325     | 0.8925   | 0.5000   |
| **Decision Tree** | 0.8700     | 0.8325   | 0.8300   |
| **Random Forest** | 0.9400     | 0.9025   | 0.9375   |
| **Neural Network**| 0.9600     | 0.9075   | 0.9200   |

### Key Takeaways

- **Contextual word embeddings (DistilBERT)** achieved the **highest accuracy across all models**, especially when used with Neural Networks and Logistic Regression.
- However, **generating contextual embeddings is computationally expensive and time-consuming**.
- **TF-IDF** performed surprisingly  well on this dataset, especially with models like Random Forest and SVM, but it **may not generalize well** to other domains without a significantly larger and more diverse text base.
- **Static word embeddings (GloVe 50d)** offer a **solid compromise**, delivering **over 90% accuracy** with minimal computational cost, making them more suitable for real-world or large-scale applications.

### Limitations

- Results are dataset-specific and may not translate directly to more diverse or real-world fake news datasets.
- TF-IDF in particular is highly domain-sensitive and may underperform without extensive corpus coverage.
- Resource constraints were not explicitly measured but are a critical consideration in deployment scenarios.


### Conclusion

While contextual embeddings provide the best raw performance, **static word embeddings appear to be the most practical choice** when balancing **accuracy, efficiency, and scalability**.


## 💻 Technologies Used

- Python 3.9+
- Scikit-learn
- Tensorflow, Torch
- Transformers (HuggingFace)
- NLTK, re
- Matplotlib (for visualization)


## 📘 Report

For a detailed walkthrough of the methodology, experiments, and findings, refer to the [`project.ipynb`](./project.ipynb).


## 📜 License

This project is licensed under the MIT License.

