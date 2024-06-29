import json
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser

st.set_page_config(
    page_title="QuizGPT | Leah",
    page_icon="❓",
)

st.title("QuizGPT")


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that is role-playing as a teacher.
            
            Based ONLY on the following context, make 5 questions to test the user's knowledge about the text. 
            
            Each question should have 4 answers, three of them must be incorrect and one should be correct. 
            
            Use (o) to signal the correct answer. 
            
            Question example:
            
            Question: What is the color of ocean?
            Answers: Red | Yellow | Green | Blue(o)
            
            Question: What is the capital of Japan?
            Answers: Seoul | Busan | Tokyo(o) | Utah
            
            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o) | Painter | Actor | Model
            
            Question: When was the movie Avatar released?
            Answers: 2007 | 2009(o) | 2010 | 1998
            
            Your turn!
            
            Context: {context}
            """,
        )
    ]
)

qa_chain = qa_prompt | llm

format_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a powerful formatting algorithm.
            
            You format the given questions-and-answers into JSON format. Answers with (o) are correct ansewrs. See the example below:
            
            Example Input:
            
            Question: What is the color of ocean?
            Answers: Red | Yellow | Green | Blue(o)
            
            Question: What is the capital of Japan?
            Answers: Seoul | Busan | Tokyo(o) | Utah
            
            Question: Who was Julius Caesar?
            Answers: A Roman Emperor(o) | Painter | Actor | Model
            
            Question: When was the movie Avatar released?
            Answers: 2007 | 2009(o) | 2010 | 1998
            
            Example Output:
            
            ```json
            {{ "questions": [
                {{
                    "question": "What is the color of ocean?",
                    "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }}
                    ]
                }},
                {{
                    "question": What is the capital of Japan?
                    "answers": [
                        {{
                            "answer": "Seoul",
                            "correct": false
                        }},
                        {{
                            "answer": "Busan",
                            "correct": false
                        }},
                        {{
                            "answer": "Tokyo",
                            "correct": true
                        }},
                        {{
                            "answer": "Utah",
                            "correct": false
                        }} 
                    ]
                }},
                {{
                    "question": Who was Julius Caesar?
                    "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter ",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }}
                    ]
                }},
                {{
                    "question": "When was the movie Avatar released?",
                    "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "2010",
                            "correct": false
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }}
                    ]
                }}
            ]
            }}
            ```
            Your turn!
            
            questions-and-answers: {context}
            """,
        )
    ]
)

format_chain = format_prompt | llm


@st.cache_data(show_spinner="Loading file...")
def load_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    return UnstructuredFileLoader(file_path)


@st.cache_data(show_spinner="Generating quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": qa_chain} | format_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    topic = None
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
            docs = load_file(file)
    else:
        topic = st.text_input("Search Wikipedia here...")
        if topic:
            docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
        <span style="font-size: 23px;" >Welcome to QuizGPT! Leah will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.</span>
        
        Get started by uploading a file or searching on Wikipedia in the sidebar. </span>
        """,
        unsafe_allow_html=True,
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                correct = list(
                    filter(lambda answer: answer["correct"], question["answers"])
                )[0]["answer"]
                st.error(f"Wrong! The correct answer is: {correct}")

        button = st.form_submit_button()
