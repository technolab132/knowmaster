import streamlit as st
import PyPDF2
import json
import os
from dotenv import load_dotenv
from io import BytesIO
import openai
import time

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="JusticeMinds Knowledge Base",
    page_icon="📚",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
button[title="View fullscreen"] {visibility: hidden;}
img {cursor: default !important;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.image("https://www.justice-minds.com/logomain.png", width=150 ,use_column_width=False)


# OpenRouter API configuration
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = os.getenv('OPENROUTER_API_KEY')

# Available models from OpenRouter with proper model IDs
MODELS = {
    "openai/gpt-4-turbo-preview": "Most capable GPT-4 model",
    "openai/gpt-3.5-turbo": "Efficient GPT-3.5 model",
    "anthropic/claude-3-opus": "Most capable Claude model",
    "anthropic/claude-3-sonnet": "Balanced Claude model",
    "google/gemini-pro": "Google's most capable model",
    "meta-llama/llama-2-70b-chat": "Meta's largest model",
    "mistral/mistral-medium": "Balanced Mistral model"
}

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_content' not in st.session_state:
    st.session_state.document_content = ""

def extract_text_from_pdf(pdf_file):
    """Extract text content from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=2000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

def show_preparation_steps(placeholder):
    """Show preparation steps with animated ellipsis"""
    steps = [
        "🔍 Analyzing document context",
        "📚 Processing content chunks",
        "🧠 Preparing AI model",
        "✨ Initializing response generation"
    ]
    
    for step in steps:
        for i in range(4):  # Show animation for each step
            dots = "." * i
            placeholder.markdown(f"**{step}{dots}**")
            time.sleep(0.3)
    
    placeholder.markdown("**🚀 Starting response generation...**")
    time.sleep(0.5)
    placeholder.empty()

def get_ai_response_stream(prompt, response_placeholder, model="openai/gpt-4-turbo-preview"):
    """Get streaming response from OpenRouter API using OpenAI SDK"""
    try:
        # Show preparation steps
        prep_placeholder = st.empty()
        show_preparation_steps(prep_placeholder)
        
        # Split document into chunks with overlap
        chunks = chunk_text(st.session_state.document_content)
        
        # Create a comprehensive system message
        system_message = """You are a highly knowledgeable assistant analyzing documents and providing detailed, accurate responses. 
        When answering:
        1. Use specific quotes and references from the document when relevant
        2. Provide comprehensive, well-structured responses
        3. If information is not in the document, clearly state that
        4. Maintain accuracy while being thorough"""
        
        # Prepare messages with context
        messages = [
            {"role": "system", "content": system_message}
        ]
        
        # Add context from chunks
        context_message = "Document contents:\n\n"
        for i, chunk in enumerate(chunks):
            context_message += f"Section {i+1}:\n{chunk}\n\n"
        
        messages.append({
            "role": "user",
            "content": f"{context_message}\nBased on the above document, please answer this question: {prompt}"
        })

        # Create streaming chat completion
        stream = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=4000,
            headers={
                "HTTP-Referer": "https://localhost:8501",
                "X-Title": "Knowledge Base Chat"
            },
            stream=True  # Enable streaming
        )
        
        # Initialize response collection
        collected_messages = []
        full_response = ""
        
        # Stream the response
        for chunk in stream:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    # Update the placeholder with the accumulated response
                    response_placeholder.markdown(f"🤖 **Assistant:** {full_response}▌")
        
        # Return the complete response
        return full_response

    except openai.error.AuthenticationError:
        return "Error: Invalid API key. Please check your OpenRouter API key configuration."
    except openai.error.Timeout:
        return "Error: Request timed out. Please try again."
    except openai.error.APIError as e:
        return f"OpenRouter API error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Streamlit UI
st.title("📚 JusticeMinds Knowledge Base")
st.write("Upload a PDF document and chat with its contents using various AI models!")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    with st.spinner("Processing document..."):
        st.session_state.document_content = extract_text_from_pdf(uploaded_file)
        st.success("Document uploaded and processed successfully!")
        # Show first 500 characters of processed text
        st.write("Preview of processed text:")
        st.write(st.session_state.document_content[:500] + "...")

# Model selection
selected_model = st.selectbox(
    "Select AI Model",
    options=list(MODELS.keys()),
    format_func=lambda x: f"{x.split('/')[-1]} - {MODELS[x]}"
)

# Chat interface
if st.session_state.document_content:
    # Input field for user question
    user_input = st.text_input("Ask a question about your document:")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Create a placeholder for the streaming response
        response_placeholder = st.empty()
        
        # Get streaming AI response
        full_response = get_ai_response_stream(user_input, response_placeholder, selected_model)
        
        # Add the complete response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Clear the streaming placeholder
        response_placeholder.empty()

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"🧑 **You:** {message['content']}")
        else:
            st.markdown(f"🤖 **Assistant:** {message['content']}")
        st.markdown("---")
else:
    st.info("Please upload a PDF document to start chatting!")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.write("""
    This application allows you to:
    1. Upload PDF documents
    2. Choose from various AI models
    3. Chat with your documents using advanced AI
    
    The available models are powered by OpenRouter API, giving you access to the latest AI technologies.
    """)
    
    st.header("Model Information")
    for model, description in MODELS.items():
        st.write(f"**{model.split('/')[-1]}**")
        st.write(description)
        st.write("---")
    
    if openai.api_key:
        st.success("OpenRouter API Key is configured")
    else:
        st.error("OpenRouter API Key is not configured. Please add it to your .env file")
