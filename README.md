# 🏨 Sentiment Analysis on Hotel Reviews

This project focuses on analyzing the sentiment of hotel reviews using traditional machine learning techniques. The goal is to classify user feedback as **positive** or **negative** based on the textual content of their reviews.

---

## 📌 Features

- Text preprocessing and cleaning (removal of stopwords, punctuation, etc.)
- TF-IDF vectorization for feature extraction
- Model training using:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
- Evaluation using accuracy, precision, recall, F1-score
- Data visualization of results and class distribution

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- NLTK (for NLP preprocessing)

---

## 📂 Dataset

- A labeled dataset of hotel reviews containing:
  - Review text
  - Corresponding sentiment label (positive or negative)
- Dataset is preprocessed before training (e.g., lowercase conversion, stopword removal, tokenization)

---

## ⚙️ How it Works

1. **Preprocessing**
   - Cleaning text: Lowercasing, punctuation & stopword removal
   - Tokenization
   - Lemmatization (optional)

2. **Vectorization**
   - Using **TF-IDF** to convert text into numerical features

3. **Model Training**
   - Training multiple models on the TF-IDF features
   - Splitting data into training and test sets
   - Hyperparameter tuning for best performance

4. **Evaluation**
   - Accuracy, precision, recall, F1-score
   - Confusion matrix visualization
   - Bar plots of model comparisons

---

## 📊 Results

✅ **SVM achieved the highest accuracy and F1-score.**

---

## 🔁 Future Work

- Implement deep learning models (LSTM, CNN)
- Incorporate BERT or DistilBERT embeddings
- Build a web-based interface for real-time sentiment prediction
- Handle multilingual hotel reviews (e.g., Hindi, Spanish)

---
## Sample Output
Review: "The room was clean and the staff was friendly."
Predicted Sentiment: Positive

Review: "Terrible experience. The AC didn’t work and it was noisy."
Predicted Sentiment: Negative
