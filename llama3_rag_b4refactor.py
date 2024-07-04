### Llama3 Local RAG Resarch AI Tool

## Index

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.docstore.document import Document

# Load environment variables from .env file
load_dotenv()

local_llm = "llama3"
# compare it with gpt3.5-turbo --> local_llm = ChatOpenAI(temperature=0)

urls = [
    "https://mormonr.org/qnas/a9l1T/the_kinderhook_plates",
    "https://rsc.byu.edu/no-weapon-shall-prosper/did-joseph-smith-translate-kinderhook-plates",
    "https://www.fairlatterdaysaints.org/answers/Kinderhook_Plates",
    "https://www.churchofjesuschrist.org/study/ensign/1981/08/kinderhook-plates-brought-to-joseph-smith-appear-to-be-a-nineteenth-century-hoax",
]

docs = [
    FireCrawlLoader(
        api_key=os.getenv("FIRECRAWL_API_KEY"), url=url, mode="scrape"
    ).load()
    for url in urls
]

# Split documents
docs_list = []
for sublist in docs:
    for item in sublist:
        docs_list.append(item)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1024,
    chunk_overlap=30,
)

docs_splits = text_splitter.split_documents(docs_list)

# Filter out complex metadata and ensure proper document formatting
filtered_docs = []
for doc in docs_splits:
    # Ensure the doc is an instance of Document and has a 'metadata' attribute
    if isinstance(doc, Document) and hasattr(doc, "metadata"):
        clean_metadata = {
            k: v
            for k, v in doc.metadata.items()
            if isinstance(v, (str, int, float, bool))
        }
        filtered_docs.append(Document(doc.page_content, metadata=clean_metadata))

# Add to vectorDB. For deployment, we want to use pinecone
vectorstore = Chroma.from_documents(
    documents=filtered_docs,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)

retriever = vectorstore.as_retriever()


## Web Search With Travily

travily_api_key = os.getenv("TAVILY_API_KEY")

from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3, api_key=travily_api_key)


## Retrieval Grader
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

llm = ChatOllama(temperature=0, format="json", model=local_llm)

prompt = PromptTemplate(
    template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a grader assessing relevance of a retrieved doucment to a user question. If the document contains keywords related to the user queston, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give the binary score, 'yes' or 'no' score, to indicate whether the retrieved document is relevant to question. \n 
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: {document} \n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["document", "question"],  # do I really need this?
)

retrieval_grader = prompt | llm | JsonOutputParser()
question = (
    "Does kinderhook plates prove that Joseph Smith as a false prophet or a liar?"
)
docs = retriever.invoke(question)

doc_text = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_text}))


## Generate Answer
# Generate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# Create a more structured prompt
system_template = """You are an assistant for question-answering tasks. Your responses must follow this exact format:

Answer: [Your detailed answer here]

Sources:
[1] [URL or reference for first source]
[2] [URL or reference for second source]
[3] [URL or reference for third source]
...

Use the following pieces of retrieved context to answer the question. When you answer the question, it's MOST important that you back it up with specific citations or source links. If the documents use footnotes, be sure to track them down and use the information provided by the footnotes. Provide the URL if you are using a source link. Give as many reliable sources as you find in the documents. If there are conflicts or inconsistencies between multiple sources you found from the retrieved context, choose one based on sound logic (i.e., firsthand accounts are preferred over second-hand accounts, verified research with newer dates is preferred) and explain why you made the choice.

If you don't know the answer, just say that you don't know, but still provide the format above."""

human_template = """Question: {question}

Context: {context}

Remember to follow the exact format specified: An "Answer:" section followed by a "Sources:" section with numbered links."""

chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ]
)

# Use a stricter temperature setting
llm = ChatOpenAI(temperature=0.1, model="gpt-4")

rag_chain = chat_prompt | llm | StrOutputParser()

# Test the chain
question = "Does the Kinderhook plates incident prove that Joseph Smith was a false prophet or a liar?"
docs = retriever.invoke(question)
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)


## Hallucination Grader
llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate.from_template(
    """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a grader assessing whether a generated answer is grounded in / supported by a set of facts. Your response MUST be a simple JSON object with a single key 'score' and a value of either 'yes' or 'no'. Do not include any explanations, preambles, or additional information.

<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here are the facts:
-------
{documents}
-------

Here is the generated answer:
{generation}

Is the generated answer grounded in and supported by the given facts? Respond with ONLY a JSON object in the format {{"score": "yes"}} or {{"score": "no"}}.

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
)

hallucination_grader = (
    prompt | llm | JsonOutputParser()
)  # eventaully we should stop using parsers but use tool calling, as this is getting outdated
hallucination_grader.invoke({"documents": docs, "generation": generation})


## Answer Grader
llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate.from_template(
    """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing whether a generated answer is useful to resolve a question by a user. Your response MUST be a simple JSON object with a single key 'score' and a value of either 'yes' or 'no' to indicate whether the answer is useful to resove the question. Do not include any explanations, preambles, or additional information.

    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Here is the generated answer:
    ------------
    {generation}
    ------------
    /n/n
    Here is the question: 
    ------------
    {question}
    ------------
    Is the generated answer useful to resolve the user question? Respond with ONLY a JSON object in the format {{"score": "yes}} or {{score": "no"}}. 

    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
)

answer_grader = prompt | llm | JsonOutputParser()
answer_grader.invoke({"question": question, "generation": generation})


### Langgraph setup - states and nodes
from typing import List
from typing_extensions import TypedDict


# State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attibutes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]


# Nodes
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, "documents", which contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state):
    """
    Determines whether the retrived documents are relevant to the quetion. If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Grade each doc
    filtered_docs = []
    web_search = "No"
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT IS RELEVANT---")
            filtered_docs.append(doc)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT IS NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
    }


def generate(state):
    """
        Generate answer using RAG on retreived documents

    `   Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, "generation" , which contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
    }


def web_search(state):
    """
    Web search based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query: question"})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {
        "documents": documents,
        "question": question,
    }


# Conditional edge
def decide_to_generate(state):
    """
    Determine whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered for not being relevant
        # We will regenerate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION. INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Print documents and generation for debugging
    print(f"Documents: {documents}")
    print(f"Generation: {generation}")

    # Invoke hallucination grader
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    # Print score for debugging
    print(f"Hallucination Grader Output: {score}")

    # Check if 'score' key is present
    if "score" not in score:
        raise KeyError(
            "The 'score' key is missing in the output from the hallucination grader"
        )

    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke(
            {
                "question": question,
                "generation": generation,
            },
        )

        # Print score for debugging
        print(f"Retrieval Grader Output: {score}")

        # Check if 'score' key is present
        if "score" not in score:
            raise KeyError(
                "The 'score' key is missing in the output from the retrieval grader"
            )

        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, PLEASE RETRY---")
        return "not supported"


## Build Graphs
from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

# Set the entry point of the workflow
workflow.set_entry_point("retrieve")

# Define edges
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)

# Compile
app = workflow.compile()

# Test
from pprint import pprint

inputs = {"question": "Did Joseph Smith Translate the Kinderhook Plates?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
print(value["generation"])
