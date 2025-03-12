import streamlit as st
import pandas as pd
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
import speech_recognition as sr
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from docx import Document as Dip
import fitz
from dotenv import load_dotenv
import pyttsx3
import time
import os
import base64
import tempfile

# Custom CSS for styling
custom_css = """
<style>
    .main-title {
        font-size: 3rem !important;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #AED6F1;
    }

    .stButton > button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background-color: #1A5276;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .chat-container {
        border-radius: 10px;
        border: 1px solid #E5E8E8;
        padding: 1rem;
        background-color: #F8F9F9;
    }

    .sidebar-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E86C1;
        margin-bottom: 1rem;
    }

    .chat-history-btn {
        margin-bottom: 0.5rem;
    }

    .file-upload-section {
        background-color: #EBF5FB;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px dashed #2E86C1;
    }

    .user-message {
        background-color: #D4E6F1;
        border-radius: 15px 15px 15px 0;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .assistant-message {
        background-color: #EBF5FB;
        border-radius: 15px 15px 0 15px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .export-btn {
        float: right;
    }

    .voice-toggle {
        background-color: #F8F9F9;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #E5E8E8;
    }

    .progress-bar-container {
        margin: 1rem 0;
    }

    /* Animation for the spinner */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .spinner {
        border: 4px solid rgba(0, 0, 0, 0.1);
        width: 36px;
        height: 36px;
        border-radius: 50%;
        border-left-color: #2E86C1;
        animation: spin 1s linear infinite;
    }
</style>
"""

# Set page configuration
st.set_page_config(page_title="File_analyzer", layout="wide")
st.markdown(custom_css, unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'general_chat_history' not in st.session_state:
    st.session_state.general_chat_history = {}
if 'file_chat_history' not in st.session_state:
    st.session_state.file_chat_history = {}
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
if 'chat_type' not in st.session_state:
    st.session_state.chat_type = "general"
if 'current_chat_name' not in st.session_state:
    st.session_state.current_chat_name = "New Chat"
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'file_data' not in st.session_state:
    st.session_state.file_data = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'voice_mode' not in st.session_state:
    st.session_state.voice_mode = False
if 'messages' not in st.session_state:
    st.session_state.messages = []


# Function to generate a unique chat ID
def generate_chat_id():
    return str(int(time.time()))


# Function to extract text from a PDF
def pdf_text_extract(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text.strip()


# Function to read different file types
def read_txt(file):
    return file.getvalue().decode("utf-8")


def read_docx(file):
    doc = Dip(file)
    return "\n".join(para.text for para in doc.paragraphs)


def read_csv(file):
    return pd.read_csv(file)


def read_excel(file):
    return pd.read_excel(file)


def read_json(file):
    return pd.read_json(file)


# Function to extract speech input
def extract_from_voice():
    st.markdown("<h4 style='color:#2E86C1'>🎤 Listening... (Speak now)</h4>", unsafe_allow_html=True)
    status_placeholder = st.empty()
    status_placeholder.info("Listening...")

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_text = recognizer.listen(source, timeout=5)
            status_placeholder.info("Processing speech...")
            vmsg = recognizer.recognize_google(audio_text)
            status_placeholder.success(f"You said: {vmsg}")
            return vmsg
    except sr.UnknownValueError:
        status_placeholder.error("Sorry, I did not understand that.")
        return None
    except sr.RequestError:
        status_placeholder.error("Could not request results from Google Speech Recognition service.")
        return None
    except Exception as e:
        status_placeholder.error(f"Error: {str(e)}")
        return None


# Function to process text-based files and create a vector database
def process_text_data(data, llm):
    with st.spinner("Processing text data..."):
        st.markdown("""
        <div class="progress-bar-container">
            <h4 style='color:#2E86C1'>Preparing document for analysis</h4>
        </div>
        """, unsafe_allow_html=True)

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
        chunks = splitter.split_text(data)
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Show progress
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

        vector_db = FAISS.from_documents(documents, OllamaEmbeddings(model="all-minilm"))

        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            You are an AI language model assistant. Your task is to generate five different versions of the given 
            user question to retrieve relevant documents from a vector database. Your goal is to help the user by 
            providing alternate questions to solve the problem of distance-based similarity search.
            Original Question: {question}
            """
        )
        retriever = MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=query_prompt)

        prompt_template = ChatPromptTemplate.from_template("""
            content:
            {context}
            Question: {question}
            """)

        chain = (
                {"context": retriever, "question": RunnablePassthrough()} |
                prompt_template |
                llm |
                StrOutputParser()
        )
        return chain


# Function to initialize the chatbot
def initialize_bot():
    memory = ConversationBufferMemory()
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""
            You are an AI assistant. Respond concisely and only with the information requested.
            Conversation history:
            {history}
            User: {input}
            AI:
            """
    )
    # Initialize conversation chain
    conversation = ConversationChain(llm=st.session_state.llm, memory=memory, prompt=prompt)
    return conversation


# Function to save chat history
def save_chat(chat_name, chat_id, chat_type):
    if chat_type == "general":
        st.session_state.general_chat_history[chat_id] = {
            "name": chat_name,
            "messages": st.session_state.messages
        }
    else:
        st.session_state.file_chat_history[chat_id] = {
            "name": chat_name,
            "messages": st.session_state.messages,
            "file_type": st.session_state.file_type
        }


# Function to load chat history
def load_chat(chat_id, chat_type):
    if chat_type == "general":
        if chat_id in st.session_state.general_chat_history:
            st.session_state.messages = st.session_state.general_chat_history[chat_id]["messages"]
            st.session_state.current_chat_name = st.session_state.general_chat_history[chat_id]["name"]
    else:
        if chat_id in st.session_state.file_chat_history:
            st.session_state.messages = st.session_state.file_chat_history[chat_id]["messages"]
            st.session_state.current_chat_name = st.session_state.file_chat_history[chat_id]["name"]
            st.session_state.file_type = st.session_state.file_chat_history[chat_id]["file_type"]


# Function to create a new chat
def new_chat():
    st.session_state.current_chat_id = generate_chat_id()
    st.session_state.messages = []
    st.session_state.current_chat_name = "New Chat"


# Function to export chat as a text file
def export_chat():
    if not st.session_state.messages:
        st.warning("No chat to export.")
        return

    chat_content = f"# {st.session_state.current_chat_name}\n\n"
    for message in st.session_state.messages:
        role = "User" if message["role"] == "user" else "AI"
        chat_content += f"**{role}**: {message['content']}\n\n"

    # Create a download link
    b64 = base64.b64encode(chat_content.encode()).decode()
    filename = f"{st.session_state.current_chat_name.replace(' ', '_')}.txt"
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" style="text-decoration:none;"><button style="background-color:#2E86C1;color:white;border:none;padding:0.5rem 1rem;border-radius:5px;cursor:pointer;">Download Chat</button></a>'
    return href


# Initialize LLM
def initialize_llm():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API")
    genai.configure(api_key=api_key)
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash")


# Main app
def main():
    # Set title
    st.markdown("<h1 class='main-title'>📊 Advanced File Analyzer</h1>", unsafe_allow_html=True)

    # Initialize LLM if not already initialized
    if 'llm' not in st.session_state:
        st.session_state.llm = initialize_llm()

    # Sidebar
    with st.sidebar:
        st.markdown("<div class='sidebar-title'>🔧 Control Panel</div>", unsafe_allow_html=True)

        # Chat type selection
        chat_type = st.radio("Select Chat Type", ["General Chat", "File Chat"],
                             index=0 if st.session_state.chat_type == "general" else 1)

        st.session_state.chat_type = "general" if chat_type == "General Chat" else "file"

        # New chat button
        if st.button("🆕 New Chat"):
            new_chat()

        # Chat history
        st.markdown("<div class='sidebar-title'>💬 Chat History</div>", unsafe_allow_html=True)

        # Display general chat history
        if st.session_state.chat_type == "general":
            if st.session_state.general_chat_history:
                for chat_id, chat_info in st.session_state.general_chat_history.items():
                    if st.button(f"📝 {chat_info['name']}", key=f"general_{chat_id}",
                                 help=f"Load chat: {chat_info['name']}"):
                        st.session_state.current_chat_id = chat_id
                        load_chat(chat_id, "general")
                        st.rerun()
            else:
                st.info("No general chats yet. Start a new conversation!")

        # Display file chat history
        else:
            if st.session_state.file_chat_history:
                for chat_id, chat_info in st.session_state.file_chat_history.items():
                    if st.button(f"📄 {chat_info['name']}", key=f"file_{chat_id}",
                                 help=f"Load chat: {chat_info['name']}"):
                        st.session_state.current_chat_id = chat_id
                        load_chat(chat_id, "file")
                        st.rerun()
            else:
                st.info("No file chats yet. Upload a file to start!")

    # Export chat option in top right
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📤 Export Chat", help="Export current conversation"):
            href = export_chat()
            if href:
                st.markdown(href, unsafe_allow_html=True)

    # Chat name input
    with col1:
        st.session_state.current_chat_name = st.text_input("💬 Chat Name", value=st.session_state.current_chat_name,
                                                           help="Enter a name for your conversation")

    # Handle file upload for file chat
    if st.session_state.chat_type == "file":
        st.markdown("<div class='file-upload-section'>", unsafe_allow_html=True)
        file = st.file_uploader("📂 Upload a file for analysis",
                                type=["pdf", "txt", "docx", "csv", "xlsx", "json"],
                                help="Supported file types: PDF, TXT, DOCX, CSV, XLSX, JSON")
        st.markdown("</div>", unsafe_allow_html=True)

        if file:
            file_type = file.name.split(".")[-1]
            st.session_state.file_type = file_type

            # Process the file based on its type
            file_readers = {
                "csv": read_csv,
                "xlsx": read_excel,
                "json": read_json,
                "pdf": pdf_text_extract,
                "txt": read_txt,
                "docx": read_docx,
            }

            if file_type in file_readers:
                try:
                    st.session_state.file_data = file_readers[file_type](file)
                    st.success(f"✅ File processed successfully: {file.name}")

                    # Create the chain for text-based files
                    if file_type in ["pdf", "txt", "docx"]:
                        st.session_state.chain = process_text_data(str(st.session_state.file_data),
                                                                   st.session_state.llm)

                    # Create a new chat ID for this file
                    if st.session_state.current_chat_id is None:
                        st.session_state.current_chat_id = generate_chat_id()

                    # Save the file name as chat name if it's a new chat
                    if st.session_state.current_chat_name == "New Chat":
                        st.session_state.current_chat_name = file.name
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
            else:
                st.error("❌ Unsupported file type")

    # Voice assistance toggle
    st.markdown("<div class='voice-toggle'>", unsafe_allow_html=True)
    st.session_state.voice_mode = st.checkbox("🎤 Enable Voice Assistant", value=st.session_state.voice_mode)
    st.markdown("</div>", unsafe_allow_html=True)

    # Initialize conversation for general chat if needed
    if st.session_state.chat_type == "general" and st.session_state.conversation is None:
        st.session_state.conversation = initialize_bot()

    # Display chat messages
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center;padding:2rem;">
                <h3 style="color:#7FB3D5">Welcome to Advanced File Analyzer!</h3>
                <p>Start a conversation or upload a file to begin.</p>
            </div>
            """, unsafe_allow_html=True)

        for message in st.session_state.messages:
            role_class = "user-message" if message["role"] == "user" else "assistant-message"
            role_icon = "👤" if message["role"] == "user" else "🤖"

            st.markdown(f"""
            <div class="{role_class}">
                <strong>{role_icon} {message["role"].title()}</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    if st.session_state.voice_mode:
        if st.button("🎤 Start Voice Input", help="Speak your question"):
            user_input = extract_from_voice()
            if user_input:
                process_input(user_input)

    user_input = st.chat_input("Type your message here...")
    if user_input:
        process_input(user_input)

    # Save chat if there are messages
    if st.session_state.messages and st.session_state.current_chat_id:
        save_chat(st.session_state.current_chat_name, st.session_state.current_chat_id, st.session_state.chat_type)


def process_input(user_input):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display thinking indicator
    with st.spinner("🧠 Thinking..."):
        if st.session_state.chat_type == "general":
            response = st.session_state.conversation.run(user_input)
        else:
            if st.session_state.file_data is not None:
                if st.session_state.file_type in ["pdf", "txt", "docx"]:
                    response = st.session_state.chain.invoke(user_input)
                else:
                    response = st.session_state.llm.invoke(
                        f"This is the question: {user_input}\nYou have content: {st.session_state.file_data.to_string()}"
                    ).content
            else:
                response = "Please upload a file first."

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Speak the response if voice mode is enabled
    if st.session_state.voice_mode:
        speak(response)

    # Force a rerun to update the UI with new messages
    st.rerun()


# Function to speak text
def speak(text):
    try:
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            temp_filename = f.name

        # Use pyttsx3 to convert text to speech
        engine = pyttsx3.init()
        engine.setProperty('rate', 185)
        engine.save_to_file(text, temp_filename)
        engine.runAndWait()

        # Play the audio using streamlit's audio
        audio_file = open(temp_filename, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

        # Clean up the temporary file
        os.unlink(temp_filename)
    except Exception as e:
        st.error(f"Error during text-to-speech: {str(e)}")


if __name__ == "__main__":
    main()
