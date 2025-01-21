import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import torch
from urllib.parse import urljoin
from huggingface_hub import hf_hub_download

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define documentation links for each CDP
docs_links = {
    "Segment": "https://segment.com/docs",
    "mParticle": "https://docs.mparticle.com",
    "Lytics": "https://docs.lytics.com",
    "Zeotap": "https://docs.zeotap.com/home/en-us"
}

# Handle forbidden errors gracefully
def is_accessible(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

# Function to preprocess text
def preprocess_text(text):
    return ' '.join(text.lower().split())

# Scrape all links from a documentation page
def scrape_links(base_url):
    try:
        if not is_accessible(base_url):
            return {}
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = {}
        for a_tag in soup.find_all('a', href=True):
            text = preprocess_text(a_tag.get_text())
            href = urljoin(base_url, a_tag['href'])
            links[text] = href
        return links
    except Exception as e:
        st.error(f"Error fetching links from {base_url}: {e}")
        return {}

# Scrape content from a specific URL
def scrape_content(url):
    try:
        if not is_accessible(url):
            return [f"Cannot access {url}. The page might be restricted."]
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content = [p.get_text().strip() for p in soup.find_all(['p', 'li'])]
        return content
    except Exception as e:
        return [f"Error scraping content from {url}: {e}"]

# Find the most relevant link for a query
def find_relevant_link(query, links):
    query_embedding = model.encode(query, convert_to_tensor=True)
    best_link = None
    best_score = -1
    for text, link in links.items():
        text_embedding = model.encode(text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, text_embedding).item()
        if score > best_score:
            best_score = score
            best_link = (text, link)
    return best_link

# Fetch relevant information for the query
def fetch_relevant_information(query, cdp):
    base_url = docs_links.get(cdp)
    if not base_url:
        return [f"No documentation available for {cdp}."]

    # Step 1: Scrape links from the main documentation page
    links = scrape_links(base_url)
    if not links:
        return [f"No links found in {cdp}'s documentation."]

    # Step 2: Identify the most relevant link
    relevant_link = find_relevant_link(query, links)
    if not relevant_link:
        return [f"No relevant section found in {cdp}'s documentation."]

    section_name, section_url = relevant_link
    st.info(f"Exploring section: {section_name} ({section_url})")

    # Step 3: Scrape full content from the relevant link
    content = scrape_content(section_url)
    if not content:
        return [f"No content found in the section {section_name}."]

    # Step 4: Filter and format content appropriately
    query_embedding = model.encode(query, convert_to_tensor=True)
    content_embeddings = model.encode(content, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, content_embeddings)[0]
    top_indices = torch.topk(scores, k=min(5, len(content))).indices.tolist()

    if len(content) > 1:
        output = [f"### {section_name.capitalize()}"]
        for idx in top_indices:
            if len(content[idx]) > 20:
                output.append(f"- {content[idx]}")
    else:
        output = [content[0]]
    return output

# Handle cross-CDP comparisons
def compare_cdps(query, cdp1, cdp2):
    info_cdp1 = fetch_relevant_information(query, cdp1)
    info_cdp2 = fetch_relevant_information(query, cdp2)

    comparison = [
        f"### Comparison: {cdp1} vs. {cdp2}",
        f"**{cdp1}:**",
        *info_cdp1,
        "",
        f"**{cdp2}:**",
        *info_cdp2,
    ]
    return comparison

# Streamlit Chatbot UI
st.set_page_config(page_title="CDP Chatbot", page_icon="💬", layout="wide")

# Header Section
st.title("💬 CDP Query Assistant")
st.markdown(
    "This chatbot provides information and comparisons for Segment, mParticle, Lytics, and Zeotap. "
    "Ask a question or request comparisons between CDPs."
)

# Input Section
query = st.text_input("Ask your question:", placeholder="e.g., How does Segment's audience creation compare to Lytics'?")
cdp_choice = st.selectbox("Select a CDP to search:", list(docs_links.keys()), index=0)
comparison_mode = st.checkbox("Enable Cross-CDP Comparison")
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
    for result in results:
        st.write(result)

# Chat History
st.markdown("---")
st.write("**Previous Queries & Answers**")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if submit and query:
    st.session_state.chat_history.append((query, results))

# Display Chat History
for q, a in st.session_state.chat_history:
    st.markdown(f"**You:** {q}")
    for ans in a:
        st.markdown(f"**Bot:** {ans}")
