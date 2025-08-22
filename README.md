# Context-Based-Academic-Search-Engine 📚

This Django application is a powerful, context-based search engine designed for academic documents. It uses a **hybrid search** approach to provide highly relevant and accurate results by combining traditional keyword matching with modern semantic search capabilities.

---

## Features

* **Intelligent PDF Processing**: Extracts key information such as author, title, abstract, and full text directly from PDF files.
* **Vector Embeddings**: Leverages pre-trained `SentenceTransformer` models to create **vector embeddings** for each document, capturing their semantic meaning.
* **Hybrid Search Algorithm**: Combines the efficiency of **BM25** (Best Match 25) for keyword-based ranking with the conceptual power of **semantic similarity** using vector embeddings.
* **Author-Based Boosting**: Provides a significant score boost to documents where the author's name matches or is closely related to the query, enhancing discoverability of specific works.
* **Query Enhancement**: Automatically corrects spelling errors using `difflib` and expands queries with relevant synonyms from `WordNet` to ensure comprehensive search results.
* **Robust Indexing**: Processes and indexes documents to create a searchable database, including a rich vocabulary for spell correction.

---

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd Context-Based-Academic-Search-Engine
    ```
2.  **Set Up Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    (Note: You'll need to create a `requirements.txt` file containing `Django`, `pdfplumber`, `sentence-transformers`, `nltk`, `rank-bm25`, and `numpy`).

4.  **Download NLTK Resources**:
    The provided Python script automatically downloads required NLTK data (`punkt`, `stopwords`, `wordnet`) if they are not already present.

---

## Usage

1.  **Integrate with Django**: Ensure the provided Python script (or its functions) is properly integrated into your Django project's views or management commands to process PDFs and execute searches.
2.  **Process Documents**: Use a Django management command or an administrative interface to upload and process your academic PDF files. The `process_pdf` function will extract data and generate embeddings for each document.
3.  **Run the Django App**:
    ```bash
    python manage.py runserver
    ```
4.  **Search**: Access your Django application's search interface in a web browser. Enter your academic queries, and the system will leverage its hybrid algorithm to return the most relevant papers, ranked by a combined score.

---

## Code Structure

* **`engine/`**: (Assuming this is where your provided Python code resides or is integrated).
    * `__init__.py`: Handles initial setup, including NLTK resource downloads.
    * `process_pdf(file_path)`: Extracts text, metadata, and generates embeddings for a given PDF file.
    * `tokenize(text)`: Normalizes text, removes stopwords, and lemmatizes tokens.
    * `get_model()`: Lazy-loads the `SentenceTransformer` model.
    * `correct_spell(query, corpus_words)`: Corrects spelling in the query.
    * `expand_query(query)`: Expands the query with synonyms.
    * `to_numpy(vec)`: Converts embeddings to NumPy arrays.
    * `_build_bm25_corpus(docs, field_boosts)`: Creates a BM25 corpus with field boosting.
    * `_author_overlap_bonus(query_tokens, author_text)`: Calculates bonus for author overlap.
    * `hybrid_search(query, docs, ...)`: The main function that orchestrates the hybrid search process, combining BM25, semantic similarity, and author boosting.
* **`myproject/`** (or your Django project's root):
    * `settings.py`: Django project settings.
    * `urls.py`: URL routing for the project.
* **`myapp/`** (or your specific Django app, e.g., `search_app`):
    * `models.py`: Defines Django models for storing processed document data (e.g., author, title, summary, embedding).
    * `views.py`: Contains the logic for handling web requests and rendering search results.
    * `templates/`: HTML files for the user interface.
    * `static/`: CSS, JavaScript, and other static assets.

---

## Dependencies

* **Django**: A high-level Python Web framework that encourages rapid development and clean, pragmatic design.
* **pdfplumber**: A library for extracting text and data from PDFs.
* **nltk**: The Natural Language Toolkit, used for text processing tasks like tokenization, stopword removal, lemmatization, and synonym expansion.
* **sentence-transformers**: A Python framework for state-of-the-art sentence, text and image embeddings.
* **rank-bm25**: A simple, cython-optimized library for BM25 ranking.
* **numpy**: A fundamental package for scientific computing with Python.
* **difflib**: A Python module that provides classes and functions for comparing sequences.
