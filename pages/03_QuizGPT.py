import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader

st.set_page_config(
    page_title="QuizGPT | Leah",
    page_icon="❓",
)

st.title("QuizGPT")


@st.cache_resource(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose one of the options:",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt, or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
            st.write(docs)
    else:
        topic = st.text_input("Search Wikipedia here...")
        if topic:
            retriever = WikipediaRetriever(top_k_results=3)
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)

if not docs:
    st.markdown(
        """
        <span style="font-size: 23px;" >Welcome to QuizGPT! Leah will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.</span>
        
        Get started by uploading a file or searching on Wikipedia in the sidebar. </span>
        """,
        unsafe_allow_html=True,
    )
else:
    st.write(docs)
