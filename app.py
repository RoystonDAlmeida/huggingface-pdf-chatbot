import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader  # For reading PDF documents
from langchain.memory import ConversationBufferMemory #memory
import io

# -----------------------  Configuration  -----------------------

# Hugging Face Model (choose a suitable one)
MODEL_NAME = "facebook/bart-large-cnn" # A summarization model

# ----------------------- Helper Functions -----------------------

def read_pdf(uploaded_file):
    """Reads text from a PDF file."""
    text = ""
    try:
        # Use BytesIO to handle the file-like object
        pdf_file = io.BytesIO(uploaded_file.read())
        reader = PdfReader(pdf_file) # Pass the BytesIO object to PdfReader
        for page in reader.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

    return text

def initialize_pipeline(model_name):
    """Initializes the Hugging Face pipeline."""
    try:
        summarizer = pipeline("summarization", model=model_name)
        return summarizer
    except Exception as e:
        st.error(f"Error initializing pipeline: {e}")
        return None

# ----------------------- Chatbot Logic -----------------------

def chatbot(document_text, summarizer, conversation_memory):
    """Handles the chatbot interaction."""
    if summarizer is None:
        st.write("Summarization pipeline not initialized.  Check configuration.")
        return

    user_input = st.text_input("Ask a question about the document:")

    if user_input:
        # 1. Summarize Document (optional, but helpful for long documents)
        try:
            summary = summarizer(document_text, max_length=130, min_length=30, do_sample=False)
            summary_text = summary[0]['summary_text']
            #st.write("Document Summary:", summary_text) #commented out to not show the summary
        except Exception as e:
            st.error(f"Error during summarization: {e}")
            summary_text = document_text  # Fallback: Use the full text
            #st.write("Document Summary: (Using full text due to summarization error)") #commented

        # 2. Augment User Input with Context (Memory)
        # Load existing memory
        if "history" in st.session_state:
            conversation_memory.chat_memory.messages = st.session_state.history
        else:
            st.session_state.history = conversation_memory.chat_memory.messages

        # 3. Add the user's question to conversation history
        conversation_memory.save_context({"input": user_input}, {"output": "..."})

        #4. Build the Prompt
        context = "\n".join([str(msg.content) for msg in conversation_memory.chat_memory.messages])
        prompt = f"""You are a chatbot assisting users with questions about a document.
        Here is the document summary: {summary_text}
        Here is the conversation history: {context}
        Answer the user's question concisely, using the information from the document and conversation history only.  If the answer is not in the document, respond with "I am sorry, I cannot answer the question based on the document.".
        User question: {user_input}
        """

        # 5. Inference
        qa_pipeline = pipeline("text-generation", model = "google/flan-t5-base") #chose text-generation as it can take long prompts

        try:
            answer = qa_pipeline(prompt, max_length=200, do_sample=True)
            answer_text = answer[0]['generated_text']

        except Exception as e:
            st.error(f"Error during question answering: {e}")
            answer_text = "I encountered an error processing your question."

        #6. Store the response
        conversation_memory.save_context({"input": user_input}, {"output": answer_text})
        st.session_state.history = conversation_memory.chat_memory.messages # Save current memory

        st.write("Chatbot's Answer:", answer_text) #display response

        # Display conversation history
        with st.expander("Show Conversation History"): #collapsible box
             for i, message in enumerate(st.session_state.history):
                st.write(f"{message.type.capitalize()}: {message.content}")

# ----------------------- Main Application -----------------------

def main():
    st.set_page_config(page_title="PDF Document Chatbot")
    st.title("PDF Document Chatbot")

    # 1. File Upload
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file is not None:
        # 2. Read the PDF
        document_text = read_pdf(uploaded_file)

        if document_text:
            # 3. Initialize Pipeline
            summarizer = initialize_pipeline(MODEL_NAME)

            # 4. Initialize Conversation Memory
            if 'conversation_memory' not in st.session_state:
               st.session_state.conversation_memory = ConversationBufferMemory(return_messages=True)

            # 5. Run Chatbot
            chatbot(document_text, summarizer, st.session_state.conversation_memory)

if __name__ == "__main__":
    main()