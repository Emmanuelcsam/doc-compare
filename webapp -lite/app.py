# app.py - Improved Flask webapp for document comparison

import logging
import subprocess
import sys
import os
import importlib
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import io
import base64
import difflib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx', 'md'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Set up logging
logging.basicConfig(
    filename='doc_compare_web.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
logging.getLogger('').addHandler(console_handler)

logging.info("Starting the document comparison webapp.")

# Function to install or upgrade a package
def install_or_upgrade_package(package_name):
    logging.info(f"Checking and installing/upgrading package: {package_name}")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', package_name])
        logging.info(f"Successfully installed/upgraded {package_name}.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install/upgrade {package_name}: {e}")

# Additional package for Flask
install_or_upgrade_package('flask')

# List of required packages and their import names
required_packages = {
    'PyPDF2': 'PyPDF2',
    'python-docx': 'docx',
    'sumy': 'sumy',
    'nltk': 'nltk',
    'scikit-learn': 'sklearn',
    'spacy': 'spacy'
}

# Check each package
for pkg, mod in required_packages.items():
    try:
        importlib.import_module(mod)
        logging.info(f"{mod} is already installed.")
        install_or_upgrade_package(pkg)
        importlib.reload(importlib.import_module(mod))
    except ImportError:
        logging.info(f"{mod} not found. Installing now.")
        install_or_upgrade_package(pkg)
        importlib.import_module(mod)

# Import modules
import PyPDF2
from docx import Document
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import spacy

# NLTK downloads
nltk_downloads = ['punkt', 'vader_lexicon', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
for resource in nltk_downloads:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'sentiment/{resource}' if resource == 'vader_lexicon' else resource)
    except LookupError:
        nltk.download(resource, quiet=True)

# spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Helper functions (adapted from script)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    logging.info(f"Extracting text from {filepath} (extension: {ext})")
    try:
        if ext in ['.txt', '.md']:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        elif ext == '.pdf':
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = '\n'.join([page.extract_text() or '' for page in reader.pages])
        elif ext == '.docx':
            doc = Document(filepath)
            text = '\n'.join([para.text for para in doc.paragraphs])
        else:
            raise ValueError("Unsupported file type")
        return text
    except Exception as e:
        logging.error(f"Failed to extract text: {e}")
        raise

def find_paragraph_source(sentence, paragraphs):
    for idx, para in enumerate(paragraphs):
        if sentence in para or any(word in para for word in sentence.split() if len(word) > 5):
            return idx + 1
    return None

def generate_summary(text, num_sentences):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return [str(s) for s in summary]

def extract_entities(text):
    doc = nlp(text)
    return {(ent.text.lower(), ent.label_) for ent in doc.ents}

def analyze_sentiment(paragraphs):
    sentiments = [sia.polarity_scores(para)['compound'] for para in paragraphs]
    avg = sum(sentiments) / len(sentiments) if sentiments else 0
    return avg, sentiments

def get_topic_modeling(text1, text2, num_topics):
    try:
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tfidf = vectorizer.fit_transform([text1, text2])
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(tfidf)
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic in lda.components_:
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            topics.append(top_words)
        topics1 = lda.transform(tfidf[0:1])[0]
        topics2 = lda.transform(tfidf[1:2])[0]
        return topics, topics1, topics2
    except:
        return None, None, None

def highlight_entities(text):
    doc = nlp(text)
    highlighted = text
    for ent in reversed(doc.ents):  # Reverse to avoid offset issues
        highlighted = highlighted[:ent.start_char] + f'<span class="entity entity-{ent.label_}">{ent.text}</span>' + highlighted[ent.end_char:]
    return highlighted

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files:
            return render_template('index.html', error='Missing files')
        
        file1 = request.files['file1']
        file2 = request.files['file2']
        
        if file1.filename == '' or file2.filename == '':
            return render_template('index.html', error='No selected files')
        
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return render_template('index.html', error='Invalid file types')
        
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        file1.save(path1)
        file2.save(path2)
        
        summary_sentences = int(request.form.get('summary_sentences', 20))
        similarity_threshold = float(request.form.get('similarity_threshold', 0.6))
        num_topics = int(request.form.get('num_topics', 5))
        
        try:
            text1 = extract_text_from_file(path1)
            text2 = extract_text_from_file(path2)
            
            # Improved paragraph splitting: handle multiple newlines better
            paras1 = [p.strip() for p in text1.split('\n\n') if p.strip()]
            paras2 = [p.strip() for p in text2.split('\n\n') if p.strip()]
            
            sum1 = generate_summary(text1, summary_sentences)
            sum2 = generate_summary(text2, summary_sentences)
            
            # Summaries with sources, entities, and highlighted entities
            sum1_data = []
            for s in sum1:
                source = find_paragraph_source(s, paras1)
                entities = ', '.join([ent.text for ent in nlp(s).ents]) if nlp(s).ents else 'None'
                highlighted = highlight_entities(s)
                sum1_data.append({'text': s, 'highlighted': highlighted, 'source': source, 'entities': entities})
            
            sum2_data = []
            for s in sum2:
                source = find_paragraph_source(s, paras2)
                entities = ', '.join([ent.text for ent in nlp(s).ents]) if nlp(s).ents else 'None'
                highlighted = highlight_entities(s)
                sum2_data.append({'text': s, 'highlighted': highlighted, 'source': source, 'entities': entities})
            
            ents1 = extract_entities(text1)
            ents2 = extract_entities(text2)
            common_ents = sorted(ents1.intersection(ents2))
            unique_ents1 = sorted(ents1 - ents2)
            unique_ents2 = sorted(ents2 - ents1)
            
            avg_sent1, sent1 = analyze_sentiment(paras1)
            avg_sent2, sent2 = analyze_sentiment(paras2)
            
            pos1 = [(i+1, paras1[i][:100], sent1[i]) for i in range(len(sent1)) if sent1[i] > 0.5]
            neg1 = [(i+1, paras1[i][:100], sent1[i]) for i in range(len(sent1)) if sent1[i] < -0.5]
            pos2 = [(i+1, paras2[i][:100], sent2[i]) for i in range(len(sent2)) if sent2[i] > 0.5]
            neg2 = [(i+1, paras2[i][:100], sent2[i]) for i in range(len(sent2)) if sent2[i] < -0.5]
            
            topics, topics1, topics2 = get_topic_modeling(text1, text2, num_topics)
            topic_data = []
            if topics:
                for i, words in enumerate(topics):
                    topic_data.append({'words': ', '.join(words), 'weight1': topics1[i], 'weight2': topics2[i]})
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(paras1 + paras2)
            sim_matrix = cosine_similarity(tfidf_matrix[:len(paras1)], tfidf_matrix[len(paras1):])
            
            similar_pairs = []
            html_differ = difflib.HtmlDiff()
            for i in range(sim_matrix.shape[0]):
                for j in range(sim_matrix.shape[1]):
                    score = sim_matrix[i, j]
                    if score >= similarity_threshold:
                        para1_ents = {ent.text.lower() for ent in nlp(paras1[i]).ents}
                        para2_ents = {ent.text.lower() for ent in nlp(paras2[j]).ents}
                        overlap = ', '.join(para1_ents.intersection(para2_ents)) if para1_ents.intersection(para2_ents) else 'None'
                        # Generate diff table
                        diff_table = html_differ.make_table(
                            paras1[i].splitlines(keepends=True),
                            paras2[j].splitlines(keepends=True),
                            fromdesc=f"{filename1} Para {i+1}",
                            todesc=f"{filename2} Para {j+1}"
                        )
                        similar_pairs.append({
                            'para1': i+1,
                            'text1': paras1[i],
                            'para2': j+1,
                            'text2': paras2[j],
                            'score': score,
                            'overlap': overlap,
                            'diff_table': diff_table
                        })
            
            unique1 = []
            for i in range(len(paras1)):
                if max(sim_matrix[i]) < similarity_threshold:
                    highlighted = highlight_entities(paras1[i])
                    unique1.append((i+1, paras1[i], highlighted, sent1[i]))
            
            unique2 = []
            for j in range(len(paras2)):
                if max(sim_matrix[:, j]) < similarity_threshold:
                    highlighted = highlight_entities(paras2[j])
                    unique2.append((j+1, paras2[j], highlighted, sent2[j]))
            
            # Clean up files
            os.remove(path1)
            os.remove(path2)
            
            return render_template('result.html',
                                   filename1=filename1, filename2=filename2,
                                   sum1_data=sum1_data, sum2_data=sum2_data,
                                   common_ents=common_ents, unique_ents1=unique_ents1, unique_ents2=unique_ents2,
                                   avg_sent1=avg_sent1, avg_sent2=avg_sent2,
                                   pos1=pos1, neg1=neg1, pos2=pos2, neg2=neg2,
                                   topic_data=topic_data,
                                   similar_pairs=similar_pairs,
                                   unique1=unique1, unique2=unique2)
        except Exception as e:
            logging.error(f"Processing error: {e}")
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
