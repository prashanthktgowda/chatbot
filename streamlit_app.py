import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import torch
from urllib.parse import urljoin
from time import sleep

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Documentation links for each CDP
docs_links = {
    "Segment": "https://segment.com/docs",
    "mParticle": "https://docs.mparticle.com",
    "Lytics": "https://docs.lytics.com",
    "Zeotap": "https://docs.zeotap.com/home/en-us/"
}

# Preprocess text for better matching
def preprocess_text(text):
    return ' '.join(text.lower().strip().split())

# Scrape all links from a documentation page
def scrape_links(base_url, query):
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract and filter links based on query similarity
        links = {}
        query_embedding = model.encode(query, convert_to_tensor=True)
        for a in soup.find_all('a', href=True):
            link_text = preprocess_text(a.get_text())
            if len(link_text) > 3:  # Avoid empty or short links
                text_embedding = model.encode(link_text, convert_to_tensor=True)
                score = util.pytorch_cos_sim(query_embedding, text_embedding).item()
                if score > 0.5:  # Adjust the threshold for relevance
                    links[link_text] = urljoin(base_url, a['href'])
        return links
    except Exception as e:
        st.error(f"Error scraping links from {base_url}: {e}")
        return {}

# Scrape content from a specific URL
def scrape_content(url, query):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Collect potential content (code, scripts, paragraphs)
        content = []
        for tag in ['code', 'pre', 'script', 'p', 'li', 'h2', 'h3']:
            for element in soup.find_all(tag):
                text = element.get_text().strip()
                if text:
                    content.append(text)

        # Match the content to the query
        query_embedding = model.encode(query, convert_to_tensor=True)
        content_embeddings = model.encode(content, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, content_embeddings)[0]
        top_indices = torch.topk(scores, k=min(10, len(content))).indices.tolist()
        relevant_content = [content[idx] for idx in top_indices if len(content[idx]) > 20]

        # Remove duplicate points and sequence results
        return sorted(set(relevant_content), key=relevant_content.index)
    except Exception as e:
        st.warning(f"Error scraping content from {url}: {e}")
        return []

# Format answers with step-by-step instructions
def format_answer(content):
    if not content:
        return "- No relevant content found for your query."

    steps = []
    for i, point in enumerate(content, start=1):
        steps.append(f"{i}. {point}")
    return "\n".join(steps)

# Recursive function to find the most relevant information
def find_relevant_information(query, base_url, max_depth=3, current_depth=0):
    if current_depth > max_depth:
        return None

    # Step 1: Scrape links with query relevance
    links = scrape_links(base_url, query)
    if not links:
        return None

    # Step 2: Find the most relevant link
    query_embedding = model.encode(query, convert_to_tensor=True)
    best_link, best_score = None, -1

    for text, link in links.items():
        text_embedding = model.encode(text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, text_embedding).item()
        if score > best_score:
            best_score = score
            best_link = (text, link)

    if best_link:
        section_name, section_url = best_link
        st.info(f"Exploring: {section_name} ({section_url})")

        # Step 3: Fetch content if it's a final article link
        content = scrape_content(section_url, query)
        if content:
            return content

        # Step 4: Recurse if no relevant content is found
        sleep(1)  # Avoid aggressive scraping
        return find_relevant_information(query, section_url, max_depth, current_depth + 1)

    return None

# Fetch relevant information for a query
def fetch_relevant_information(query, cdp):
    base_url = docs_links.get(cdp)
    if not base_url:
        return [f"No documentation available for {cdp}."]

    relevant_content = find_relevant_information(query, base_url)
    if relevant_content:
        return format_answer(relevant_content)

    return "- No relevant content found for your query."

# Cross-CDP comparisons
def compare_cdps(query, cdp1, cdp2):
    info_cdp1 = fetch_relevant_information(query, cdp1)
    info_cdp2 = fetch_relevant_information(query, cdp2)

    comparison = [
        f"### Comparison: {cdp1} vs. {cdp2}",
        f"**{cdp1}:**\n{info_cdp1}",
        "",
        f"**{cdp2}:**\n{info_cdp2}",
    ]
    return "\n".join(comparison)

# Streamlit Chatbot UI
st.set_page_config(page_title="CDP Support Chatbot", page_icon="💬", layout="wide")

# Header Section
st.title("💬 CDP Support Chatbot")
st.markdown(
    "This chatbot helps you with 'how-to' questions for Segment, mParticle, Lytics, and Zeotap."
)

# Input Section
query = st.text_input("Ask your question:", placeholder="e.g., How do I create an audience in Lytics?")
cdp_choice = st.selectbox("Select a CDP to search:", list(docs_links.keys()), index=0)
comparison_mode = st.checkbox("Compare with another CDP")
if comparison_mode:
    compare_cdp_choice = st.selectbox("Select another CDP for comparison:", list(docs_links.keys()), index=1)

submit = st.button("Submit Query")

# Process User Query
if submit and query:
    if comparison_mode:
        st.markdown(f"**Comparing {cdp_choice} and {compare_cdp_choice}...**")
        results = compare_cdps(query, cdp_choice, compare_cdp_choice)
    else:
        st.markdown(f"**Fetching information from {cdp_choice} documentation...**")
        results = fetch_relevant_information(query, cdp_choice)

    st.markdown(results)

# Maintain Chat History
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if submit and query:
    st.session_state.chat_history.append((query, results))

# Display Chat History
st.markdown("---")
st.write("**Previous Queries & Answers**")
for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
