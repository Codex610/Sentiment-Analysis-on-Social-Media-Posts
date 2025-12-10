## 📘 Sentiment Analysis on Social Media Posts — Flask + LSTM

A deep learning–powered web application that predicts Positive or Negative sentiment from social media posts using an LSTM neural network.
This project uses Flask for deployment and a modern, responsive UI built with Bootstrap 5 and glassmorphism design.

### 🚀 Features

✔ Deep learning LSTM sentiment classification

✔ Modern, responsive UI with Bootstrap 5

✔ Emoji-based sentiment visualization

✔ Displays sentiment + confidence score

✔ NLP preprocessing (stopwords, stemming, cleaning)

✔ Lightweight Flask backend

✔ Easy to deploy anywhere (Render, Railway, etc.)

### 🧠 Technologies Used
- Frontend

- HTML5

- CSS3

- Bootstrap 5

- Jinja2 Templates

- Backend

- Python

- Flask

- TensorFlow / Keras

- NLTK

- NumPy

- Tokenizer (Pickle)

## 📂 Project Structure
### Sentiment-Analysis-Flask/
''' txt
├── main.py               # Flask Web App
├── lstm_model.h5         # Trained LSTM Model
├── tokenizer.pkl         # Saved Tokenizer
│
├── templates/
│   └── index.html        # Web UI
│
└── static/               # (Optional)
'''

### ⚙️ How It Works
#### 1. Text Preprocessing

- Lowercasing

- Removing unwanted characters

- Stopword removal

- Stemming with PorterStemmer

#### 2. Tokenization

- Converts text → integer sequences

- Pads to fixed length (maxlen=100)

#### 3. LSTM Model

- Embedding Layer

- LSTM Layer

- Dense (sigmoid) output

- Predicts value between 0 and 1

#### 4. Sentiment Logic

- > 0.5 → Positive 😊

- ≤ 0.5 → Negative 😡

**🛠 Installation**
#### 1. Clone this repository
''' git clone https://github.com/your-username/sentiment-analysis-flask.git '''
''' cd sentiment-analysis-flask '''

#### 2. Create a virtual environment
'''
python -m venv env
env\Scripts\activate        # Windows
source env/bin/activate     # Mac/Linux
'''

#### 3. Install dependencies
''' pip install -r requirements.txt '''


**Or manually:**

''' pip install flask tensorflow nltk numpy '''

#### 4. Run the app
''' python main.py '''

#### 5. Open in browser
''' http://127.0.0.1:5000 '''

### 🖼 Screenshots
Homepage
![Homepage](images/homepage.png)

Prediction Result
![Prediction](images/prediction.png)


(Add real screenshots inside your project folder)

## 🧪 Model Notes

- LSTM model trained on labeled text dataset

- Tokenizer saved as tokenizer.pkl

- Model saved as lstm_model.h5

- Max sequence length = 100

If needed, I can generate a full training notebook (.ipynb).

## 📌 Future Improvements

- Add Neutral sentiment class

- Add progress bar for confidence

- Add GRU/Bidirectional LSTM model

- Deploy on Render/Railway/AWS

- Store predictions in database

- Add user login system

### 🤝 Contributing

Contributions are welcome!
Open an issue or submit a pull request.
