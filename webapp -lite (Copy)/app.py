# app.py - Greatly Improved Flask Web App for Document Comparison with Advanced NLP

import logging
import subprocess
import sys
import os
import importlib
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
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

logging.info("Starting the enhanced document comparison web app.")

# Function to install or upgrade a package
def install_or_upgrade_package(package_name):
    logging.info(f"Checking and installing/upgrading package: {package_name}")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', package_name])
        logging.info(f"Successfully installed/upgraded {package_name}.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install/upgrade {package_name}: {e}")

# Additional package for enhanced Flask
install_or_upgrade_package('flask')

# List of required packages and their import names
required_packages = {
    'PyPDF2': 'PyPDF2',
    'python-docx': 'docx',
    'sumy': 'sumy',
    'nltk': 'nltk',
    'scikit-learn': 'sklearn',
    'spacy': 'spacy',
    'transformers': 'transformers',
    'sentence-transformers': 'sentence_transformers',
    'keybert': 'keybert',
    'textstat': 'textstat',
    'bertopic': 'BERTopic'
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
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import textstat
from bertopic import BERTopic

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

# Load advanced models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
keybert_model = KeyBERT()
topic_model = BERTopic(nr_topics="auto", min_topic_size=5)

# Helper functions

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

def generate_extractive_summary(text, num_sentences):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return [str(s) for s in summary]

def generate_abstractive_summary(text):
    try:
        summary = summarizer(text, max_length=300, min_length=100, do_sample=False)[0]['summary_text']
        return summary
    except:
        return "Unable to generate abstractive summary."

def extract_keyphrases(text):
    keyphrases = keybert_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=20)
    return set([kp[0].lower() for kp in keyphrases])

def calculate_readability(text):
    return {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'smog_index': textstat.smog_index(text),
        'dale_chall': textstat.dale_chall_readability_score(text)
    }

def interpret_readability(scores):
    ease = scores['flesch_reading_ease']
    if ease > 90:
        level = "very easy to read, easily understood by an average 11-year-old student."
    elif ease > 80:
        level = "easy to read."
    elif ease > 70:
        level = "fairly easy to read."
    elif ease > 60:
        level = "plain English, easily understood by 13- to 15-year-old students."
    elif ease > 50:
        level = "fairly difficult to read."
    elif ease > 30:
        level = "difficult to read."
    else:
        level = "very difficult to read, best understood by university graduates."
    return f"The document has a Flesch Reading Ease score of {ease:.2f}, which means it is {level} Other metrics: Flesch-Kincaid Grade: {scores['flesch_kincaid_grade']:.2f}, SMOG Index: {scores['smog_index']:.2f}, Dale-Chall: {scores['dale_chall']:.2f}."

def extract_entities(text):
    doc = nlp(text)
    return set((ent.text.lower(), ent.label_) for ent in doc.ents)

def analyze_sentiment(paragraphs):
    sentiments = [sia.polarity_scores(para)['compound'] for para in paragraphs if para.strip()]
    avg = sum(sentiments) / len(sentiments) if sentiments else 0
    return avg, sentiments

def generate_sentiment_description(avg_sent, pos, neg, filename):
    desc = f"The overall sentiment of {filename} is {avg_sent:.2f}, suggesting a "
    if avg_sent > 0.3:
        desc += "generally positive tone, indicating optimism, approval, or favorable descriptions throughout the document. "
    elif avg_sent < -0.3:
        desc += "generally negative tone, reflecting criticism, concerns, or unfavorable aspects. "
    else:
        desc += "neutral or balanced tone, with a mix of positive and negative elements. "
    if pos:
        desc += f"There are {len(pos)} notably positive paragraphs, often highlighting successes, benefits, or positive developments. For example, one positive paragraph discusses... "
    if neg:
        desc += f"There are {len(neg)} notably negative paragraphs, which may point out problems, risks, or criticisms. For instance, a negative paragraph addresses... "
    return desc

def get_topic_modeling(texts):
    try:
        topic_model.fit(texts)
        topic_info = topic_model.get_topic_info()
        topics = []
        for topic_id in topic_info.Topic.unique():
            if topic_id != -1:
                words = topic_model.get_topic(topic_id)
                topics.append([w[0] for w in words[:10]])
        doc_topics = topic_model.get_document_info(texts)
        topics1 = doc_topics[doc_topics.Document == texts[0]].Topic.iloc[0]
        topics2 = doc_topics[doc_topics.Document == texts[1]].Topic.iloc[0]
        return topics, topics1, topics2, topic_info
    except:
        return None, None, None, None

def generate_topic_description(topics, topics1, topics2, filename1, filename2):
    if not topics:
        return "Unable to generate topic modeling."
    desc = "Topic modeling reveals key themes in the documents. "
    for i, words in enumerate(topics):
        desc += f"Topic {i+1}: {', '.join(words)}. This topic has prominence in {filename1} with label {topics1} and in {filename2} with {topics2}. "
    desc += "Overall, the documents share themes like..., but differ in emphasis on..."
    return desc

def highlight_entities(text):
    doc = nlp(text)
    highlighted = text
    for ent in reversed(doc.ents):
        highlighted = highlighted[:ent.start_char] + f'<span class="entity entity-{ent.label_}" title="{ent.label_}">{ent.text}</span>' + highlighted[ent.end_char:]
    return highlighted

def generate_diff_summary(para1, para2):
    prompt = f"Summarize the key differences and similarities between these two paragraphs in detail: Paragraph 1: {para1[:1500]}\nParagraph 2: {para2[:1500]}"
    try:
        summary = summarizer(prompt, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
        return summary
    except:
        return "Unable to generate detailed diff summary."

def generate_overall_report(filename1, filename2, ab_sum1, ab_sum2, common_ents, unique_ents1, unique_ents2, keyphrases_common, keyphrases_unique1, keyphrases_unique2, avg_sent1, avg_sent2, topic_desc, readability_desc1, readability_desc2, similar_pairs, unique1, unique2, pos1, neg1, pos2, neg2):
    report = f"""
    Comprehensive Document Comparison Report: {filename1} vs {filename2}
    
    Executive Summary:
    {filename1} can be abstracted as: {ab_sum1}
    {filename2} can be abstracted as: {ab_sum2}
    
    The documents share common entities such as {', '.join([e[0] for e in common_ents[:5]])}, indicating overlapping subjects like persons, organizations, or locations. However, {filename1} uniquely mentions {', '.join([e[0] for e in unique_ents1[:5]])}, while {filename2} has unique entities like {', '.join([e[0] for e in unique_ents2[:5]])} , suggesting divergent focuses.
    
    Keyphrases Analysis:
    Common key phrases include {', '.join(list(keyphrases_common)[:10])}, highlighting shared themes and terminology.
    Unique keyphrases in {filename1}: {', '.join(list(keyphrases_unique1)[:10])}
    Unique to {filename2}: {', '.join(list(keyphrases_unique2)[:10])}
    This indicates that {filename1} emphasizes more on aspects like..., whereas {filename2} focuses on...
    
    Sentiment Insights:
    {generate_sentiment_description(avg_sent1, pos1, neg1, filename1)}
    {generate_sentiment_description(avg_sent2, pos2, neg2, filename2)}
    Comparatively, {filename1} appears more/less positive than {filename2}, which may reflect different authorial intentions or contextual differences.
    
    Topic Modeling Overview:
    {topic_desc}
    
    Readability Comparison:
    {readability_desc1}
    {readability_desc2}
    This suggests that {filename1} is more accessible to a general audience compared to {filename2}, or vice versa.
    
    Similarity and Differences:
    There are {len(similar_pairs)} similar sections, where content overlaps significantly. For example, one pair shows similarity in..., with differences in...
    Unique content in {filename1} includes paragraphs that introduce novel ideas such as..., contributing to its distinct perspective.
    Similarly, {filename2} has unique elements that add... 
    
    In conclusion, the two documents align on core ideas but diverge in depth, tone, and specific details, making them complementary in... 
    """
    # Use summarizer to refine the report
    refined_report = summarizer(report, max_length=800, min_length=400, do_sample=False)[0]['summary_text']
    return refined_report

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
        
        summary_sentences = int(request.form.get('summary_sentences', 50))  # Increased default for more detail
        similarity_threshold = float(request.form.get('similarity_threshold', 0.5))  # Lowered for more pairs
        try:
            text1 = extract_text_from_file(path1)
            text2 = extract_text_from_file(path2)
            
            paras1 = [p.strip() for p in text1.split('\n\n') if len(p.strip()) > 50]  # Filter short paras
            paras2 = [p.strip() for p in text2.split('\n\n') if len(p.strip()) > 50]
            
            # Summaries
            extract_sum1 = generate_extractive_summary(text1, summary_sentences)
            ab_sum1 = generate_abstractive_summary(text1)
            extract_sum2 = generate_extractive_summary(text2, summary_sentences)
            ab_sum2 = generate_abstractive_summary(text2)
            
            sum1_data = []
            for s in extract_sum1:
                source = find_paragraph_source(s, paras1)
                entities = ', '.join([f"{ent.text} ({ent.label_})" for ent in nlp(s).ents]) or 'None'
                highlighted = highlight_entities(s)
                sum1_data.append({'text': s, 'highlighted': highlighted, 'source': source, 'entities': entities})
            
            sum2_data = []
            for s in extract_sum2:
                source = find_paragraph_source(s, paras2)
                entities = ', '.join([f"{ent.text} ({ent.label_})" for ent in nlp(s).ents]) or 'None'
                highlighted = highlight_entities(s)
                sum2_data.append({'text': s, 'highlighted': highlighted, 'source': source, 'entities': entities})
            
            # Entities
            ents1 = extract_entities(text1)
            ents2 = extract_entities(text2)
            common_ents = sorted(ents1.intersection(ents2))
            unique_ents1 = sorted(ents1 - ents2)
            unique_ents2 = sorted(ents2 - ents1)
            
            # Keyphrases
            keyphrases1 = extract_keyphrases(text1)
            keyphrases2 = extract_keyphrases(text2)
            common_keyphrases = sorted(keyphrases1.intersection(keyphrases2))
            unique_keyphrases1 = sorted(keyphrases1 - keyphrases2)
            unique_keyphrases2 = sorted(keyphrases2 - keyphrases1)
            
            # Readability
            readability1 = calculate_readability(text1)
            readability2 = calculate_readability(text2)
            readability_desc1 = interpret_readability(readability1)
            readability_desc2 = interpret_readability(readability2)
            
            # Sentiment
            avg_sent1, sent1 = analyze_sentiment(paras1)
            avg_sent2, sent2 = analyze_sentiment(paras2)
            pos1 = [(i+1, paras1[i][:300], sent1[i]) for i in range(len(sent1)) if sent1[i] > 0.3]  # Adjusted threshold
            neg1 = [(i+1, paras1[i][:300], sent1[i]) for i in range(len(sent1)) if sent1[i] < -0.3]
            pos2 = [(i+1, paras2[i][:300], sent2[i]) for i in range(len(sent2)) if sent2[i] > 0.3]
            neg2 = [(i+1, paras2[i][:300], sent2[i]) for i in range(len(sent2)) if sent2[i] < -0.3]
            sent_desc1 = generate_sentiment_description(avg_sent1, pos1, neg1, filename1)
            sent_desc2 = generate_sentiment_description(avg_sent2, pos2, neg2, filename2)
            
            # Topic Modeling
            texts = [text1, text2]
            topics, topics1, topics2, topic_info = get_topic_modeling(texts)
            topic_data = []
            if topics:
                for i, words in enumerate(topics):
                    topic_data.append({'words': ', '.join(words), 'weight1': topic_model.get_topics()[i+1][0][1] if i+1 in topic_model.get_topics() else 0, 'weight2': topic_model.get_topics()[i+1][1][1] if i+1 in topic_model.get_topics() else 0})
            topic_desc = generate_topic_description(topics, topics1, topics2, filename1, filename2)
            
            # Similarity with semantic embeddings
            emb1 = sentence_model.encode(paras1, convert_to_tensor=True)
            emb2 = sentence_model.encode(paras2, convert_to_tensor=True)
            sim_matrix = util.cos_sim(emb1, emb2).cpu().numpy()
            
            similar_pairs = []
            html_differ = difflib.HtmlDiff()
            for i in range(sim_matrix.shape[0]):
                for j in range(sim_matrix.shape[1]):
                    score = sim_matrix[i, j]
                    if score >= similarity_threshold:
                        para1_ents = {ent.text.lower() for ent in nlp(paras1[i]).ents}
                        para2_ents = {ent.text.lower() for ent in nlp(paras2[j]).ents}
                        overlap = ', '.join(para1_ents.intersection(para2_ents)) or 'None'
                        diff_table = html_differ.make_table(
                            paras1[i].splitlines(keepends=True),
                            paras2[j].splitlines(keepends=True),
                            fromdesc=f"{filename1} Para {i+1}",
                            todesc=f"{filename2} Para {j+1}"
                        )
                        diff_summary = generate_diff_summary(paras1[i], paras2[j])
                        similar_pairs.append({
                            'para1': i+1,
                            'text1': paras1[i],
                            'para2': j+1,
                            'text2': paras2[j],
                            'score': score,
                            'overlap': overlap,
                            'diff_table': diff_table,
                            'diff_summary': diff_summary
                        })
            
            unique1 = []
            for i in range(len(paras1)):
                if max(sim_matrix[i]) < similarity_threshold:
                    highlighted = highlight_entities(paras1[i])
                    unique_summary = summarizer(paras1[i], max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                    unique1.append({'para': i+1, 'text': paras1[i], 'highlighted': highlighted, 'sent': sent1[i], 'summary': unique_summary})
            
            unique2 = []
            for j in range(len(paras2)):
                if max(sim_matrix[:, j]) < similarity_threshold:
                    highlighted = highlight_entities(paras2[j])
                    unique_summary = summarizer(paras2[j], max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                    unique2.append({'para': j+1, 'text': paras2[j], 'highlighted': highlighted, 'sent': sent2[j], 'summary': unique_summary})
            
            # Overall report
            overall_report = generate_overall_report(filename1, filename2, ab_sum1, ab_sum2, common_ents, unique_ents1, unique_ents2, common_keyphrases, unique_keyphrases1, unique_keyphrases2, avg_sent1, avg_sent2, topic_desc, readability_desc1, readability_desc2, similar_pairs, unique1, unique2, pos1, neg1, pos2, neg2)
            
            # Clean up
            os.remove(path1)
            os.remove(path2)
            
            return render_template('result.html',
                                   filename1=filename1, filename2=filename2,
                                   ab_sum1=ab_sum1, ab_sum2=ab_sum2,
                                   sum1_data=sum1_data, sum2_data=sum2_data,
                                   common_ents=common_ents, unique_ents1=unique_ents1, unique_ents2=unique_ents2,
                                   common_keyphrases=common_keyphrases, unique_keyphrases1=unique_keyphrases1, unique_keyphrases2=unique_keyphrases2,
                                   readability_desc1=readability_desc1, readability_desc2=readability_desc2,
                                   avg_sent1=avg_sent1, avg_sent2=avg_sent2, sent_desc1=sent_desc1, sent_desc2=sent_desc2,
                                   topic_data=topic_data, topic_desc=topic_desc,
                                   similar_pairs=similar_pairs,
                                   unique1=unique1, unique2=unique2,
                                   overall_report=overall_report)
        except Exception as e:
            logging.error(f"Processing error: {e}")
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
