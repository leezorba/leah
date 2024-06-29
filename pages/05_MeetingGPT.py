import streamlit as st
from operator import itemgetter
from pydub import AudioSegment
import math
import subprocess
import glob
import openai
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self, *args, **kwargs):
        self.message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
)

streaming_llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=50,
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource(show_spinner="Embedding the file...")
def embed_file(file_name):
    file_path = f"./.cache/meeting_files/{file_name}"
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_name}")
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        OpenAIEmbeddings(), cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()


@st.cache_resource()
def extract_audio_from_video(video_path):
    audio_path = video_path.replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)


@st.cache_resource()
def cut_audio_in_chunks(video_name, audio_path, chunk_minutes, chunks_folder):
    if os.path.exists(f"./.cache/chunks/{video_name}/00_chunk.mp3"):
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_seconds = chunk_minutes * 60 * 1000
    chunks = math.ceil(len(track) / chunk_seconds)
    for i in range(chunks):
        start_time = i * chunk_seconds
        end_time = (i + 1) * chunk_seconds
        chunk = track[start_time:end_time]
        chunk.export(
            f"{chunks_folder}/{str(i).zfill(2)}_chunk.mp3",
            format="mp3",
        )


@st.cache_resource()
def transcribe_chunks(chunks_folder, destination_file):
    if os.path.exists(destination_file):
        return
    files = sorted(glob.glob(f"{chunks_folder}/*.mp3"))
    for file in files:
        with open(file, "rb") as audio_file, open(destination_file, "a") as text_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
            text_file.write(transcript.text)


def process_video(video):
    st.write("Loading video...")
    video_content = video.read()
    video_path = f"./.cache/meeting_files/{video.name}"
    audio_path = (
        video_path.replace(".mp4", ".mp3")
        .replace(".avi", ".mp3")
        .replace(".mkv", ".mp3")
        .replace(".mov", ".mp3")
    )
    transcript_path = (
        video_path.replace(".mp4", ".txt")
        .replace(".avi", ".txt")
        .replace(".mkv", ".txt")
        .replace(".mov", ".txt")
    )

    with open(video_path, "wb") as f:
        f.write(video_content)

    st.write("Extracting audio...")
    extract_audio_from_video(video_path)
    st.write("Splitting audio...")
    cut_audio_in_chunks(video.name, audio_path, 10, chunks_folder)
    st.write("Transcribing audio...")
    transcribe_chunks(
        chunks_folder,
        transcript_path,
    )
    return transcript_path


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def save_memory(input, output):
    st.session_state["memory"].save_context({"input": input}, {"output": output})


# def invoke_chain(question):
#     result = chain.invoke(question)
#     save_memory(question, result.content)


@st.cache_resource()
def generate_summary(transcript_path):
    loader = TextLoader(transcript_path)
    docs = loader.load_and_split(text_splitter=splitter)

    first_summary_prompt = ChatPromptTemplate.from_template(
        """
        Write a concise summary of the following:
        "{text}"
        Concise Summary:
        """
    )

    progress_text = "Generating summary..."

    first_summary_chain = first_summary_prompt | llm | StrOutputParser()

    my_bar = st.progress(0, text=f"{progress_text} (0/{len(docs)})")

    summary = first_summary_chain.invoke({"text": docs[0].page_content})

    my_bar.progress(1 / len(docs), text=f"{progress_text} (1/{len(docs)})")

    refine_prompt = ChatPromptTemplate.from_template(
        """
        Your job is to produce a final summary. We have provided an existing summary upto a certain point: {existing_summary}.
        
        We have the opportunity to refine the existing summary (only if needed) with some more context below.
        
        ----------
        {context}
        ----------
        
        Given the new context, refine the previously existing summary. If the new given context isn't useful, RETURN the previously existing summary right before the iteration as it is. DO NOT just give something like "The original summary is sufficient and does not need to be refined with the additional context provided." as a summary. Do not start your final summary with a phrase like "the new context provided..." Keep the final summary under 500 words. 
        """
    )

    refine_chain = refine_prompt | llm | StrOutputParser()
    for i, doc in enumerate(docs[1:]):
        summary = refine_chain.invoke(
            {
                "existing_summary": summary,
                "context": doc.page_content,
            }
        )
        my_bar.progress((i + 2) / len(docs), f"{progress_text} ({i+2}/len(docs))")
    return summary


st.set_page_config(page_title="MtgGPT", page_icon="🕑")

st.title("MeetingGPT")

st.markdown(
    """       
    <span style="font-size: 20px;"> Welcome to MeetingGPT, upload a video and Leah will give you a transcript, a summary and a chatbot to ask any questions about it.</span>
    
    <span style="font-size: 20px;"> Get started by uploading a video file in the sidebar.</span>          
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    chunks_folder = "./.cache/chunks/"
    with st.status("Processing the video..."):
        transcript_path = process_video(video)

    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())

    with summary_tab:
        start = st.button("Generate summary")

        if start or st.session_state["isSummaryGenerated"]:
            summary = generate_summary(transcript_path)
            st.write(summary)
            st.session_state["isSummaryGenerated"] = True

    with qa_tab:
        retriever = embed_file(transcript_path.split("/")[-1])
        query = st.chat_input("Ask any questions about the transcript.")
        paint_history()

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
             Answer the question using ONLY the following context. If you don't know the answer, just say you don't know the answer based on the given data and documents. Don't make anything up. 
             Context: {context}
             """,
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        if query:
            send_message(query, "human")
            chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough(),
                    "history": RunnableLambda(
                        st.session_state.memory.load_memory_variables
                    )
                    | itemgetter("history"),
                }
                | qa_prompt
                | streaming_llm
            )
            with st.chat_message("ai"):
                chain.invoke(query).content
                # save_memory()

else:
    st.session_state["messages"] = []
    st.session_state["memory"] = ConversationBufferMemory(
        llm=llm,
        max_token_limit=1000,
        return_messages=True,
    )
    st.session_state["isSummaryGenerated"] = False
