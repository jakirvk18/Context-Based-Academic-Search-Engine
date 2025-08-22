# Context-Based-Academic-Search-Engine 📚

This Django application is a powerful, context-based search engine designed for academic documents. It uses a **hybrid search** approach to provide highly relevant and accurate results by combining traditional keyword matching with modern semantic search capabilities.

---

## Features

* **Intelligent PDF Processing**: Extracts key information such as author, title, abstract, and full text directly from PDF files. This data is then used to enrich the search index.
* **Vector Embeddings**: Leverages pre-trained `SentenceTransformer` models to create **vector embeddings** for each document. These numerical representations capture the semantic meaning, which is crucial for finding related documents even if they don't share the same keywords.
* **Hybrid Search Algorithm**: The core of the application, it blends two powerful search techniques:
    1.  **BM25 (Best Match 25)**: A classic keyword-matching algorithm that ranks documents based on term frequency and document length. The implementation boosts important fields like author and title, giving them more weight.
    2.  **Semantic Similarity**: Uses the generated vector embeddings to find documents that are conceptually similar to the user's query, providing more intelligent search results.
* **Author-Based Boosting**: A significant score boost is applied to documents where the author's name matches or is closely related to the query, greatly enhancing the discoverability of specific works.
* **Query Enhancement**: Automatically refines user queries by:
    * **Spell Correction**: Using `difflib` to correct spelling errors.
    * **Synonym Expansion**: Expanding queries with relevant synonyms from `WordNet` to ensure comprehensive coverage.
* **Robust Indexing**: Processes and indexes academic PDF documents to create a searchable database. This includes building a rich vocabulary used for spell correction and semantic matching.

---

## Folder Structure
```bash
Markdown

# Context-Based-Academic-Search-Engine 📚

This Django application is a powerful, context-based search engine designed for academic documents. It uses a **hybrid search** approach to provide highly relevant and accurate results by combining traditional keyword matching with modern semantic search capabilities.

---

## Features

* **Intelligent PDF Processing**: Extracts key information such as author, title, abstract, and full text directly from PDF files. This data is then used to enrich the search index.
* **Vector Embeddings**: Leverages pre-trained `SentenceTransformer` models to create **vector embeddings** for each document. These numerical representations capture the semantic meaning, which is crucial for finding related documents even if they don't share the same keywords.
* **Hybrid Search Algorithm**: The core of the application, it blends two powerful search techniques:
    1.  **BM25 (Best Match 25)**: A classic keyword-matching algorithm that ranks documents based on term frequency and document length. The implementation boosts important fields like author and title, giving them more weight.
    2.  **Semantic Similarity**: Uses the generated vector embeddings to find documents that are conceptually similar to the user's query, providing more intelligent search results.
* **Author-Based Boosting**: A significant score boost is applied to documents where the author's name matches or is closely related to the query, greatly enhancing the discoverability of specific works.
* **Query Enhancement**: Automatically refines user queries by:
    * **Spell Correction**: Using `difflib` to correct spelling errors.
    * **Synonym Expansion**: Expanding queries with relevant synonyms from `WordNet` to ensure comprehensive coverage.
* **Robust Indexing**: Processes and indexes academic PDF documents to create a searchable database. This includes building a rich vocabulary used for spell correction and semantic matching.

---

## Folder Structure

searchengine/  
├── accounts/                  # Django app for user authentication, registration, profiles
│   ├── pycache/
│   ├── migrations/
│   ├── templates/
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   └── urls.py
├── home/                      # Core Django app for search engine UI and logic
│   ├── pycache/
│   ├── management/commands/   # Custom commands (e.g., for PDF processing)
│   ├── migrations/
│   ├── templates/home/        # HTML templates for home app (e.g., search page)
│   ├── init.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py              # Database models for processed documents (metadata, embeddings)
│   ├── tests.py
│   ├── urls.py                # URL patterns for the home app
│   ├── utils.py               # Utility functions, including PDF processing and search logic
│   └── views.py               # Handles web requests and renders search results
├── media/                     # Stores user-uploaded files (e.g., original PDF documents)
├── searchengine/              # Django project configuration
│   ├── init.py
│   ├── asgi.py
│   ├── settings.py            # Project settings
│   ├── urls.py                # Project-level URL dispatcher
│   └── wsgi.py
├── static/                    # Static files (CSS, JS, images, etc.) for the app
├── staticfiles/               # Collected static files (used in deployment)
├── db.sqlite3                 # Default SQLite database file
├── manage.py                  # Django's command-line utility
└── requirements.txt           # List of Python dependencies
```

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
    Create a `requirements.txt` file in your project root with the following content:
    ```
    Django
    pdfplumber
    sentence-transformers
    nltk
    rank-bm25
    numpy
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download NLTK Resources**:
    The provided Python script (likely in `home/utils.py` or similar) automatically downloads required NLTK data (`punkt`, `stopwords`, `wordnet`) if they are not already present.

5.  **Run Migrations**:
    ```bash
    python manage.py makemigrations home accounts
    python manage.py migrate
    ```

---

## Usage

1.  **Process Documents**:
    You will need a mechanism (e.g., a Django management command or an admin interface) to upload PDF files to the `media/` directory. Once uploaded, trigger the `process_pdf` function (likely from `home/utils.py`) to extract content and generate embeddings, storing this data in your Django models (e.g., defined in `home/models.py`).

2.  **Run the Django Development Server**:
    ```bash
    python manage.py runserver
    ```
3.  **Access the Search Engine**:
    Open your web browser and navigate to the appropriate URL (e.g., `http://127.0.0.1:8000/search/` if configured in `searchengine/urls.py` and `home/urls.py`). You can then enter your academic queries into the search interface. The system will process your query, perform a hybrid search using the indexed documents, and display the most relevant results.

---

## Core Components (within `home/utils.py` or similar)

* **`__init__.py`**: Ensures NLTK resources are available.
* **`process_pdf(file_path)`**: The function responsible for:
    * Opening and extracting text/metadata from PDF files using `pdfplumber`.
    * Tokenizing, removing stopwords, and lemmatizing text with `nltk`.
    * Generating a normalized vector `embedding` using `SentenceTransformer`.
    * Identifying common `keywords` from the document.
    * Returning a dictionary of extracted information and the embedding.
* **`tokenize(text)`**: Preprocesses text for search, converting to lowercase, removing non-alphanumeric characters, stemming/lemmatizing, and removing common stopwords.
* **`get_model()`**: Lazily loads the `all-MiniLM-L6-v2` `SentenceTransformer` model to conserve memory and optimize performance.
* **`correct_spell(query, corpus_words)`**: Enhances query quality by correcting common spelling mistakes against a defined vocabulary.
* **`expand_query(query)`**: Further enriches the query by adding synonyms using `nltk.corpus.wordnet`, broadening the search scope.
* **`hybrid_search(query, docs, ...)`**: The main search orchestrator that:
    * Preprocesses the user `query` (spell correction, expansion).
    * Calculates **BM25 scores** based on boosted document fields (author, title, keywords, abstract, summary).
    * Calculates **semantic similarity scores** between the query embedding and document embeddings.
    * Combines these scores using configurable `bm25_weight` and `sem_weight`.
    * Applies an **author overlap bonus** and a strong boost for exact/contained author name matches.
    * Ranks documents by the final combined score and returns the top `limit` results.

---
# **🤖 Author**
Jakir Hussain
