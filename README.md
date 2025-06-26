# RAG-Supported-QA-Engine

# Multi-PDF Question Answering Application

This project is a Streamlit application that allows users to ask questions based on uploaded PDF documents. It leverages Langchain and Groq LLM to generate answers by extracting contextual information from the relevant PDF documents.

## Features

*   **Multi-PDF Support:** Users can upload multiple PDF files.
*   **Question Answering:** Generates answers to questions based on the uploaded PDFs.
*   **Langchain Integration:** The Langchain library is used for LLM orchestration and document processing.
*   **Groq LLM:** Utilizes Groq's fast LLMs for generating answers.
*   **Streamlit Interface:** Provides a user-friendly web interface.
*   **Vector Database:** Employs ChromaDB to store vector representations of documents for efficient retrieval of relevant chunks.
*   **Session Management:** Streamlit session management ensures the vector database is rebuilt only when PDF files change, improving performance.

## Prerequisites

*   Python 3.7 or higher
*   [Groq API key](https://console.groq.com/keys)

## Installation

1.  Clone this repository:

    ```bash
    git clone <https://github.com/hitaskomur/RAG-Supported-QA-Engine>
    cd <project_directory>
    ```

2.  **Recommended:** Activate the virtual environment:

    ```bash
    # Linux/macOS
    source env/bin/activate

    # Windows
    .\env\Scripts\activate
    ```

3.  Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Set your Groq API key as an environment variable or enter it directly into the Streamlit interface.

2.  Run the Streamlit application:

    ```bash
    streamlit run main.py
    ```

3.  The application will open in a web browser. Upload PDF files, enter your question, and click the "Answer" button.

## File Structure
.
├── main.py # Main Streamlit application file
├── env/ # Virtual environment directory
├── requirements.txt # List of dependencies
├── README.md # This file
└── chroma_langchain_db/ # ChromaDB database files (created when used)







## Dependencies

The project's dependencies are listed in the `requirements.txt` file. You can easily install the required libraries using this file.

## Environment Variables

*   `GROQ_API_KEY`: Required to access the Groq API.

## Notes

*   The initial creation of the vector database, or its recreation when PDF files change, may take some time. Please be patient.
*   The `chroma_langchain_db` directory contains the vector database. Do not delete this directory unless you want to rebuild the database.
*   Note that the Groq API might have associated costs. Please check Groq's pricing policies.

## Contributing

Contributions are welcome! Please discuss issues and proposed enhancements before submitting a pull request.


## Author

[Halil İbrahim Taşkömür/hitaskomur]



