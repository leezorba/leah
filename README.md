# llama_rag.ipynb:
It implements agentic rag with LangChain and LangGraph. The model could be easily swapped between local LLMs such as llama3 and popular LLMs like gpt3.5-turbo.

Here is the workflow. 

<img width="751" alt="research gen ai_workflow" src="https://github.com/leezorba/research_gen_ai/assets/95986526/b03ea6b2-ab9f-4b0d-9400-ae1820957905">

## env file
Your .env file should have these API keys:

OPENAI_API_KEY="*****"

LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="*****"

FIRECRAWL_API_KEY="*****"

TAVILY_API_KEY="*****"

## current outcome
Here is the current outcome:

Query: Did Joseph Smith Translate the Kinderhook Plates?

Answer: The question of whether Joseph Smith translated the Kinderhook Plates is complex and has been the subject of much debate. The Kinderhook Plates were discovered in 1843 and presented to Joseph Smith, who reportedly showed interest in them. According to the History of the Church, Joseph Smith stated, “I have translated a portion of them, and find they contain the history of the person with whom they were found” [1]. However, this statement is based on a journal entry by William Clayton, which was later edited to appear as if Joseph Smith himself had written it [2].

Further complicating the matter, two eyewitnesses later confessed that the plates were a hoax, created to trap Joseph Smith into making a false translation [1]. Forensic analysis in 1980 confirmed that the plates were indeed a 19th-century forgery [1]. Despite this, some ambiguity remains about whether Joseph Smith actually attempted a translation. Some sources suggest that he may have shown interest in the plates and speculated about their content, but there is no conclusive evidence that he produced a detailed translation [3].

In summary, while there is some evidence suggesting that Joseph Smith may have attempted to translate the Kinderhook Plates, the strongest evidence indicates that the plates were a hoax and that any translation attributed to him was likely based on hearsay or misinterpretation.

Sources:
[1] https://rsc.byu.edu/no-weapon-shall-prosper/did-joseph-smith-translate-kinderhook-plates

[2] https://www.fairlatterdaysaints.org/answers/Kinderhook_Plates

[3] https://www.fairlatterdaysaints.org/answers/File:Kinderhook.plates.don.bradley.description.jpg


## update plan
As of July 10, 2024, what I am going to add in the next update is:
1. vectorcloud using pinecone
2. deployable via streamlit so user would provide their own api keys (or given api keys)


