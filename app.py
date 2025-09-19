import os
from flask import Flask, request, render_template, flash, redirect, url_for
import requests
import validators
from newspaper import Article, Config

# AI and NLP Libraries
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import nltk
from textblob import TextBlob
from collections import Counter
import re

# --- INITIAL SETUP ---

# Load environment variables from .env file
load_dotenv()

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize Flask App
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_default_dev_secret_key')

# --- CONFIGURE AI MODELS AND AGENTS ---

# Google Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Web search tool
search_tool = DuckDuckGoSearchResults(k=3)

# Initialize LangChain Agent
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# --- HELPER FUNCTIONS ---

def get_article_from_url(url):
    """Fetch and parse article text from URL using newspaper3k."""
    try:
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        config.request_timeout = 15
        
        article = Article(url, config=config)
        article.download()

        if article.download_state != 2:
            return None

        article.parse()

        if not article.text or len(article.text) < 100:
            return None

        return {
            "title": article.title or "No Title Found",
            "text": article.text,
            "authors": ', '.join(article.authors) if article.authors else "Unknown Author",
            "publish_date": article.publish_date.strftime('%B %d, %Y') if article.publish_date else "Unknown Date",
            "top_image": article.top_image or ""
        }
    except Exception as e:
        print(f"Error getting article: {e}")
        return None


def analyze_article_with_ai(text, url):
    """Use Gemini + LangChain for abstractive and extractive summarization, sentiment, authenticity, related news."""
    analysis_results = {
        "abstractive_summary": "Could not generate abstractive summary.",
        "extractive_summary": "Could not generate extractive summary.",
        "sentiment": "Could not determine sentiment.",
        "authenticity": "Could not check authenticity.",
        "related_articles": []
    }

    truncated_text = text[:4000]  # keep context small for efficiency

    # --- 1. Abstractive Summary ---
    try:
        abs_prompt = PromptTemplate.from_template("""
            Provide an **abstractive summary** of the following article text in 5–7 sentences. 
            Use your own words, do not copy exact phrases.
            Article Text: "{text}"
        """)
        abs_chain = abs_prompt | llm
        abs_result = abs_chain.invoke({"text": truncated_text})
        analysis_results["abstractive_summary"] = abs_result.content
    except Exception as e:
        analysis_results["abstractive_summary"] = f"Error during abstractive summarization: {e}"

    # --- 2. Extractive Summary ---
    try:
        ext_prompt = PromptTemplate.from_template("""
            Provide an **extractive summary** of the following article text. 
            Select the 5 most important sentences directly from the text without rephrasing.
            Article Text: "{text}"
        """)
        ext_chain = ext_prompt | llm
        ext_result = ext_chain.invoke({"text": truncated_text})
        analysis_results["extractive_summary"] = ext_result.content
    except Exception as e:
        analysis_results["extractive_summary"] = f"Error during extractive summarization: {e}"

    # --- 3. Sentiment ---
    try:
        sentiment_prompt = PromptTemplate.from_template("""
            Analyze the overall sentiment of the article. 
            Reply with one word: Positive, Negative, or Neutral.
            Article Text: "{text}"
        """)
        sentiment_chain = sentiment_prompt | llm
        sentiment_result = sentiment_chain.invoke({"text": truncated_text})
        analysis_results["sentiment"] = sentiment_result.content.strip()
    except Exception as e:
        analysis_results["sentiment"] = f"Error during sentiment analysis: {e}"

    # --- 4. Authenticity Check ---
    try:
        auth_prompt = f"""
            As a fact-checker, assess the authenticity of this article from: {url}.
            Search the web to see if major, reliable outlets (Reuters, BBC, AP, NYT) reported the same.
            Give reliability rating: High, Medium, or Low.
            Write a short explanation (2–4 sentences).
            Article Text: "{truncated_text}"
        """
        auth_result = agent.invoke(auth_prompt)
        analysis_results["authenticity"] = auth_result["output"] if isinstance(auth_result, dict) else auth_result
    except Exception as e:
        analysis_results["authenticity"] = f"Error during authenticity check: {e}"

    # --- 5. Related Articles ---
    try:
        related_prompt = f"""
            Find 3 reliable news sources covering the same story.
            Format each as:
            Source: [Name], URL: [link]

            Article Text: "{truncated_text}"
        """
        related_result = agent.invoke(related_prompt)
        related_text = related_result["output"] if isinstance(related_result, dict) else related_result

        articles_list = []
        for line in related_text.split("\n"):
            if "Source:" in line and "URL:" in line:
                try:
                    source_part, url_part = line.split("URL:", 1)
                    source = source_part.replace("Source:", "").strip()
                    article_url = url_part.strip()
                    articles_list.append({"source": source, "url": article_url})
                except ValueError:
                    continue
        analysis_results["related_articles"] = articles_list
    except Exception as e:
        analysis_results["related_articles"] = [{"source": f"Error finding related: {e}", "url": "#"}]

    return analysis_results


def perform_local_analysis(text):
    """Local NLP analysis (keywords, reading time, subjectivity)."""
    local_results = {}

    # Keywords
    try:
        words = re.findall(r'\b\w{4,}\b', text.lower())
        stop_words = set(nltk.corpus.stopwords.words('english'))
        filtered = [w for w in words if w not in stop_words]
        local_results["keywords"] = [w for w, _ in Counter(filtered).most_common(20)]
    except Exception:
        local_results["keywords"] = []

    # Reading Time
    word_count = len(text.split())
    local_results["reading_time"] = f"{max(1, word_count // 200)} min read"

    # Subjectivity
    subjectivity = TextBlob(text).sentiment.subjectivity
    if subjectivity > 0.6:
        local_results["subjectivity"] = "Opinionated / Subjective"
    elif subjectivity < 0.4:
        local_results["subjectivity"] = "Factual / Objective"
    else:
        local_results["subjectivity"] = "Mixed / Neutral"

    return local_results


# --- FLASK ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form.get('url', '').strip()

        if not url or not validators.url(url):
            flash('Please enter a valid article URL.')
            return redirect(url_for('index'))

        article_data = get_article_from_url(url)
        if not article_data:
            flash("Could not retrieve article. It might be invalid, paywalled, or too short.")
            return redirect(url_for('index'))

        try:
            ai_analysis = analyze_article_with_ai(article_data['text'], url)
            local_analysis = perform_local_analysis(article_data['text'])

            results = {**article_data, **ai_analysis, **local_analysis, "url": url}
            return render_template('index.html', results=results)
        except Exception as e:
            flash(f"Unexpected error: {e}")
            return redirect(url_for('index'))

    return render_template('index.html', results=None)


if __name__ == '__main__':
    app.run(debug=True)
