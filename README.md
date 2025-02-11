# Document Chatbot Streamlit App

This Streamlit application allows you to upload a PDF document and engage in a conversation with a chatbot about its contents. The chatbot uses Hugging Face Transformers for text summarization and question answering, leveraging either TensorFlow as a backend.

## Features

*   **PDF Upload:** Upload a PDF document for analysis.
*   **Text Summarization:** Automatically summarizes the document content for efficient processing (optional, but recommended for large documents).
*   **Question Answering:** Ask questions about the document and receive concise answers from the chatbot, which uses the document summary and conversation history as context.
*   **Conversation Memory:** Maintains a conversation history to provide contextual awareness for follow-up questions.
*   **Easy Setup:** Simple installation and configuration with `requirements.txt`.

## Prerequisites

*   Python 3.7 or higher
*   Pip package manager

## Installation

1.  **Clone the repository:**

    ```
    git clone git@github.com:RoystonDAlmeida/huggingface-pdf-chatbot.git
    cd huggingface-pdf-chatbot/
    ```

2.  **Create a virtual environment (recommended):**

    ```
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate.bat  # On Windows
    ```

3.  **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

## Configuration

1.  **Hugging Face Model:**

    *   The application uses `facebook/bart-large-cnn` for text summarization by default. You can change this by modifying the `MODEL_NAME` variable in `app.py`:

        ```
        MODEL_NAME = "facebook/bart-large-cnn"  # A summarization model
        ```

    *   Ensure that the chosen model has a TensorFlow implementation if you intend to use TensorFlow as the backend.
    *   The text generation pipeline is set to `google/flan-t5-base`.

## Usage

1.  **Run the Streamlit app:**

    ```
    streamlit run app.py
    ```

2.  **Open in browser:**

    Open your web browser and navigate to the URL displayed in the terminal (usually `http://localhost:8501`).

3.  **Upload a PDF:**

    Click the "Browse files" button to upload a PDF document.

4.  **Ask questions:**

    Enter your questions in the text input field and press Enter. The chatbot will provide answers based on the document content and conversation history.

5.  **View Conversation History:**

    Expand the "Show Conversation History" section to see the full conversation.

## requirements.txt

The `requirements.txt` file lists the project's dependencies:

```
streamlit
transformers
PyPDF2
langchain
tensorflow
```