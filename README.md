
#  SummarAIze – Intelligent News Article Analysis

## Overview

**SummarAIze** is a Flask-based web application that performs intelligent news article analysis using advanced **Natural Language Processing (NLP)** and **AI models**.
Given any news article URL, the system automatically fetches, parses, and analyzes the content to help readers understand, evaluate, and verify news quickly and effectively.

---

## Features

* **Automatic Article Retrieval** – Fetches and parses full articles directly from the web.
* **Abstractive Summarization** – Generates human-like summaries in original words using **Google Gemini + LangChain**.
* **Extractive Summarization** – Highlights the 5 most important sentences directly from the article text.
* **Sentiment Analysis** – Detects the overall sentiment (Positive, Negative, or Neutral).
* **Authenticity Check** – Cross-verifies coverage with major outlets (BBC, Reuters, NYT, AP) and rates reliability as *High, Medium, or Low*.
* **Related Coverage** – Suggests 3 reliable articles from other sources covering the same topic.
* **Keyword Extraction** – Identifies top keywords to highlight main themes.
* **Reading Time Estimation** – Calculates approximate read time based on word count.
* **Subjectivity Detection** – Evaluates whether the article is factual, opinionated, or mixed.
* **Responsive Frontend** – Clean HTML/CSS interface with instant feedback and mobile support.

---

##  Tech Stack

* **Backend:** Flask (Python)
* **Frontend:** HTML5, CSS3 (responsive, modern design)
* **AI/NLP:**

  * [LangChain](https://www.langchain.com/) with **Google Gemini** (Generative AI)
  * [Newspaper3k](https://newspaper.readthedocs.io/) for article scraping & parsing
  * [TextBlob](https://textblob.readthedocs.io/) for subjectivity & sentiment support
  * [NLTK](https://www.nltk.org/) for stopwords & tokenization
* **Search Integration:** DuckDuckGo via LangChain tool
* **Environment:** Python 3.x, dotenv for config management

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/summarAIze.git
cd summarAIze
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate    # On macOS/Linux
venv\Scripts\activate       # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Make sure `requirements.txt` includes: `flask`, `newspaper3k`, `langchain`, `langchain-google-genai`, `nltk`, `textblob`, `python-dotenv`, `validators`, `requests`.)*

### 4. Setup Environment Variables

Create a **.env** file in the project root:

```ini
FLASK_SECRET_KEY=your_secret_key
GOOGLE_API_KEY=your_google_api_key
```

### 5. Run the App

```bash
python app.py
```

The app will be available at: `http://127.0.0.1:5000/`

---

## Future Enhancements

* Multi-language support
* Bias detection in reporting
* Browser extension for instant analysis
* Batch article analysis
* Compare article
  

---

