# Automated Classification of Electrical Product PDFs  

This repository contains the **Automated Classification of Electrical Product PDFs** project, a machine learning solution for classifying PDF documents into predefined electrical categories: *Lighting*, *Fuses*, *Cables*, or *Others*. The project combines text extraction, natural language processing (NLP), and machine learning to achieve efficient and accurate classification.

---

## 🚀 Features  
- Extracts text from PDF documents using **Apache Tika**.  
- Cleans and preprocesses extracted text with advanced **NLP techniques**.  
- Employs an **XGBoost** classifier for text-based classification.  
- Provides an interactive user interface with **Streamlit**.  
- Includes pre-trained model and vectorizer for quick deployment.  

---

## 📂 File Overview  

| **File**               | **Description**                                                                 |
|-------------------------|---------------------------------------------------------------------------------|
| `README.md`            | Documentation for the repository.                                               |
| `app.py`               | Main application code for the Streamlit-based interactive web app.              |
| `experiments.ipynb`    | Jupyter notebook for exploratory data analysis (EDA) and experimentation.       |
| `processing data.ipynb`| Jupyter notebook for preprocessing the data and preparing it for training.       |
| `model.h5`             | Trained XGBoost model for PDF classification.                                   |
| `vectorizer.pkl`       | Pretrained CountVectorizer for converting text to numeric format.               |
| `labelencoder.pkl`     | Label encoder for decoding model predictions to category labels.                |
| `requirements.txt`     | List of Python packages required for the project.                               |
| `packages.txt`         | Additional dependencies or system-level requirements for deployment.            |

---

## 🛠️ Installation and Setup  

### Prerequisites  
Ensure you have the following installed:  
- Python 3.7+  
- Recommended: A virtual environment  

### Installation  

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/your-username/electrical-pdf-classifier.git
   cd electrical-pdf-classifier
   ```  

2. **Install Dependencies**:  
   Install Python dependencies listed in `requirements.txt`:  
   ```bash
   pip install -r requirements.txt
   ```  

   For system-level packages listed in `packages.txt`, install them as per your OS.

3. **Download NLTK Resources**:  
   Required NLTK corpora are downloaded during the first run. Ensure an active internet connection.  

---

## 🎮 Usage  

1. **Run the Streamlit App**:  
   Start the app locally:  
   ```bash
   streamlit run app.py
   ```  

2. **Input and Predict**:  
   - Provide a **URL** to a PDF file containing electrical product details.  
   - Specify a timeout duration for the text extraction process (default: 120 seconds).  
   - Click **Predict Class** to get the classification result.  

3. **Output**:  
   - Displays the predicted category (*Lighting*, *Fuses*, *Cables*, or *Others*).  
   - Shows the extracted text from the PDF for verification.  

---

## 🧪 Model Training  

- **Data Preparation**:  
  - Text extracted from electrical product PDFs.  
  - Preprocessed using tokenization, lemmatization, and stopword removal.  

- **Model**:  
  - Used **XGBoost Classifier** for high accuracy.  
  - Tuned hyperparameters with **GridSearchCV**.  

- **Saved Artifacts**:  
  - Trained model (`model.h5`)  
  - Pretrained vectorizer (`vectorizer.pkl`)  
  - Label encoder (`labelencoder.pkl`)  

---

## 📝 Contribution  

We welcome contributions to improve this project!  
- Found a bug? [Open an Issue](https://github.com/your-username/electrical-pdf-classifier/issues).  
- Have a feature request? Let us know.  

### Steps to Contribute  

1. Fork this repository.  
2. Create a branch (`git checkout -b feature-branch`).  
3. Commit your changes (`git commit -m "Add feature"`).  
4. Push to the branch (`git push origin feature-branch`).  
5. Create a pull request.  

---

## 📜 License  

This project is licensed under the [MIT License](LICENSE).  

---

## 💡 Acknowledgements  

- [Apache Tika](https://tika.apache.org/) for PDF text extraction.  
- [Streamlit](https://streamlit.io/) for building interactive apps.  
- [NLTK](https://www.nltk.org/) for natural language processing.  
- [XGBoost](https://xgboost.readthedocs.io/) for machine learning model implementation.  

---
