import json
import sqlite3
from langchain.docstore.document import Document
import logging

from config import DATABASE_PATH


# Create a connection to the SQLite database
def create_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT  NOT NULL,
        link TEXT NOT NULL,
        metadata TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """
    )
    conn.commit()
    conn.close()


def is_doc_exist(doc):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    link = doc.metadata["link"]
    # Check if the article already exists by its link (to avoid duplicates)
    c.execute("SELECT * FROM documents WHERE link = ?", (link,))
    result = c.fetchone()
    conn.close()
    if result:
        logging.debug(f"Article '{link}' already exists in the database.")
    return result


# Function to insert a document into the database
def insert_doc(doc):
    link = doc.metadata["link"]
    # Insert the article into the database
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO documents (title, link, metadata, content)
                 VALUES (?, ?, ?, ?)""",
        (
            doc.metadata["title"],
            link,
            json.dumps(doc.metadata),
            json.dumps(doc.page_content),
        ),
    )
    conn.commit()
    conn.close()

    logging.info(f"Article '{doc.metadata['title']}' inserted into the database.")
    return True


def load_documents_from_db():
    # Connect to SQLite database
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    # Retrieve metadata and content from the database
    c.execute("SELECT metadata, content FROM documents")
    rows = c.fetchall()

    # List to store LangChain documents
    documents = []

    for row in rows:
        # Parse the metadata and content from JSON text
        metadata = json.loads(row[0])  # Assuming metadata is stored as a JSON string
        content = json.loads(row[1])  # Assuming content is stored as a JSON string

        # Create a LangChain Document object
        doc = Document(
            metadata=metadata,  # The metadata (typically a dictionary)
            page_content=content,  # The content (typically a string or list)
        )

        # Add the document to the list
        documents.append(doc)

    logging.info(f"Loaded {len(documents)} from database")

    # Close the database connection
    conn.close()

    return documents
