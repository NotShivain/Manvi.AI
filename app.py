import streamlit as st
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


# -----------------------------
# PDF Text Extraction
# -----------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# -----------------------------
# Web Article Extraction
# -----------------------------
def extract_text_from_link(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract main readable content
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text = "\n".join(paragraphs)
        return text.strip()

    except Exception as e:
        st.error(f"❌ Error fetching article: {e}")
        return ""


# -----------------------------
# Split Text into Chunks
# -----------------------------
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# -----------------------------
# Create Vectorstore
# -----------------------------
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# -----------------------------
# Build Conversational Chain
# -----------------------------
def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# -----------------------------
# Handle User Input
# -----------------------------
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("⚠️ Please upload or process your source first!")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True
            )


# -----------------------------
# Main Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Chat with PDFs or Articles", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("🤖 Manvi.AI — Your Personal Research Paper Yapper")

    # User Question Input
    user_question = st.text_input("Ask a question about your documents or articles:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar for Upload / Link
    with st.sidebar:
        st.subheader("📄 Source Options")
        option = st.radio("Choose input type:", ("Upload PDFs", "Add Article Link"))

        if option == "Upload PDFs":
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'",
                accept_multiple_files=True
            )

            if st.button("Process PDFs"):
                if not pdf_docs:
                    st.warning("Please upload at least one PDF before processing.")
                else:
                    with st.spinner("Processing your documents... ⏳"):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("✅ PDFs processed successfully!")

        elif option == "Add Article Link":
            article_url = st.text_input("Enter the article URL:")
            if st.button("Process Article"):
                if not article_url:
                    st.warning("Please enter a valid article link.")
                else:
                    with st.spinner("Fetching and processing article... 🌐"):
                        article_text = extract_text_from_link(article_url)
                        if article_text:
                            text_chunks = get_text_chunks(article_text)
                            vectorstore = get_vectorstore(text_chunks)
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.success("✅ Article processed successfully!")

        # Watermark
        st.markdown(
            """
            <div style="position: fixed;
                        bottom: 10px;
                        left: 20px;
                        font-size: 12px;
                        color: #999999;
                        opacity: 0.7;">
                Made with ❤️ by <b>Shivain</b>
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------
# Run the App
# -----------------------------
if __name__ == '__main__':
    main()
