import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import nltk
import requests
from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import numpy as np
import time

nltk.download('punkt')

logger = logging.getLogger(__name__)

# Initialize Qdrant Client
qdrant_client = QdrantClient(host='localhost', port=6333)


class FAQScraper:
    def __init__(self, base_url, max_pages=5):
        self.base_url = base_url.rstrip('/')
        self.max_pages = max_pages
        self.visited = set()
        self.to_visit = [self.base_url]
        self.data = []
        logger.debug(f"Initialized FAQScraper with base_url: {self.base_url} and max_pages: {self.max_pages}")

        # Set up Selenium WebDriver using webdriver-manager
        options = Options()
        options.add_argument('--headless')  # Run in headless mode
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--disable-dev-shm-usage')

        self.driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=options
        )
        self.driver.set_page_load_timeout(30)  # Set timeout to 30 seconds

    def scrape(self):
        logger.debug(f"Starting scraping from base URL: {self.base_url}")
        try:
            while self.to_visit and len(self.visited) < self.max_pages:
                url = self.to_visit.pop(0)
                if url in self.visited:
                    continue
                logger.debug(f"Scraping URL: {url}")
                try:
                    self.driver.get(url)
                    time.sleep(2)  # Wait for JavaScript to load content
                    html_content = self.driver.page_source
                    logger.debug(f"HTML content of length {len(html_content)} retrieved from URL: {url}")

                    soup = BeautifulSoup(html_content, 'html.parser')

                    # Check if the page is an article (FAQ)
                    if self.is_article_page(url):
                        # Extract the question and answer
                        question, answer = self.extract_faq(soup)
                        if question and answer:
                            self.data.append({'question': question, 'answer': answer})
                            logger.debug(f"Extracted FAQ from URL: {url}")
                    else:
                        # Enqueue sub-links
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            full_url = urljoin(url, href)
                            full_url = full_url.rstrip('/')
                            if self.is_valid_url(
                                    full_url) and full_url not in self.visited and full_url not in self.to_visit:
                                self.to_visit.append(full_url)
                    self.visited.add(url)
                except (WebDriverException, TimeoutException) as e:
                    logger.error(f"Error scraping URL {url}: {e}")
                    continue
            logger.debug(f"Finished scraping. Visited {len(self.visited)} URLs. Collected {len(self.data)} FAQs.")
        finally:
            self.driver.quit()

    def is_article_page(self, url):
        # Check if the URL matches the pattern of an article page
        return re.match(r'.*/solutions/articles/\d+.*', url)

    def extract_faq(self, soup):
        # Extract the title (question)
        question_tag = soup.find('title')

        # Get all divs with class 'article-body' if more than one section
        answer_tag = soup.find(id='article-body')
        question = question_tag.get_text(strip=True) if question_tag else None

        # Concatenate all div contents if there are multiple divs
        if answer_tag:
            answer = '\n'.join([div.get_text(separator='\n', strip=True) for div in answer_tag])
        else:
            answer = None

        return question, answer

    def is_valid_url(self, url):
        # Only follow links within the same domain and under /support/solutions
        parsed_base = urlparse(self.base_url)
        parsed_url = urlparse(url)
        is_same_netloc = parsed_url.netloc == parsed_base.netloc
        is_valid_path = parsed_url.path.startswith('/support/solutions')
        return is_same_netloc and is_valid_path


def prepare_data(faq_data):
    logger.debug(f"Preparing data from {len(faq_data)} FAQs")
    prepared_data = []
    for item in faq_data:
        question = item['question']
        answer = item['answer']
        combined_text = f"Q: {question}\nA: {answer}"
        # Split the combined text into chunks if necessary
        prepared_data.append({'text': combined_text})
    logger.debug(f"Total prepared data chunks: {len(prepared_data)}")
    if not prepared_data:
        logger.error("No FAQs were prepared from the scraped data.")
        raise ValueError("No FAQs were prepared from the scraped data.")
    return prepared_data


def get_embeddings(texts):
    embeddings = []
    batch_size = 20  # Adjust as needed
    logger.debug(f"Generating embeddings for {len(texts)} texts")

    url = "https://api.openai.com/v1/embeddings"  # OpenAI's REST API endpoint for embeddings
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.debug(f"Processing batch {i // batch_size + 1}: size {len(batch)}")
        try:
            data = {
                "model": "text-embedding-3-small",
                "input": batch
            }
            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200:
                response_data = response.json()
                logger.debug(f"Received response with {len(response_data['data'])} embeddings")
                batch_embeddings = [e['embedding'] for e in response_data['data']]
                embeddings.extend(batch_embeddings)
            else:
                logger.error(f"Error generating embeddings: {response.status_code} - {response.text}")
                raise Exception(f"Error generating embeddings: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    logger.debug(f"Total embeddings generated: {len(embeddings)}")
    if not embeddings:
        logger.error("No embeddings were generated.")
        raise ValueError("No embeddings were generated.")
    return embeddings


def store_embeddings(team_id, embeddings, texts):
    if not embeddings or not texts:
        logger.error("Embeddings or texts are empty. Cannot store in Qdrant.")
        raise ValueError("Embeddings or texts are empty.")
    # Create a collection per team if not exists
    collection_name = f"team_{team_id}"
    logger.debug(f"Storing embeddings in Qdrant collection: {collection_name}")
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=len(embeddings[0]), distance="Cosine"),
    )
    # Prepare points to upload
    payload = [{"text": text} for text in texts]
    points = qmodels.Batch(
        ids=list(range(len(embeddings))),
        vectors=embeddings,
        payloads=payload,
    )
    # Upload to Qdrant
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
    )
    logger.debug(f"Stored {len(embeddings)} embeddings in collection {collection_name}")


def get_query_embedding(query):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "text-embedding-3-small",
        "input": query
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        response_data = response.json()
        query_embedding = response_data['data'][0]['embedding']
        return query_embedding
    else:
        logger.error(f"Error generating query embedding: {response.status_code} - {response.text}")
        raise Exception(f"Error generating query embedding: {response.status_code} - {response.text}")


def search_embeddings(team_id, query_embedding, top_k=5):
    collection_name = f"team_{team_id}"

    logger.debug(f"Searching embeddings in collection: {collection_name}")
    try:
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        # Extract texts from payload
        texts = [hit.payload['text'] for hit in search_result]
        logger.debug(f"Retrieved {len(texts)} relevant texts")
        return texts
    except Exception as e:
        logger.error(f"Error searching embeddings: {e}")
        raise


def clear_qdrant_collection(team_id):
    collection_name = f"team_{team_id}"
    try:
        if qdrant_client.get_collection(collection_name):
            qdrant_client.delete_collection(collection_name=collection_name)
            logger.info(f"Successfully deleted collection '{collection_name}' from Qdrant.")
            print(f"Successfully deleted collection '{collection_name}' from Qdrant.")
        else:
            logger.warning(f"Collection '{collection_name}' does not exist in Qdrant.")
            print(f"Collection '{collection_name}' does not exist in Qdrant.")
    except Exception as e:
        logger.error(f"An error occurred while deleting collection '{collection_name}': {e}")
        print(f"An error occurred while deleting collection '{collection_name}': {e}")
