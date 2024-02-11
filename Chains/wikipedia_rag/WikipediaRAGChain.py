import os
from operator import itemgetter
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from fastapi import FastAPI
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
from langchain_community.retrievers import WikipediaRetriever
from langserve.pydantic_v1 import BaseModel, Field

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You're a helpful research assistant. Given a user question and some Wikipedia articles,
answer the user question and provide citations. If none of the articles answer the question, say you don't know.

You must return both an answer and citations. A citation consists of a VERBATIM quote from the original
article that justifies the answer. Return a citation for every fact or quote in your answer.

Here are the Wikipedia articles:{context}""",
         ),
        ("human", "{question}"),
    ]
)


def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string.:"""
    formatted = [
        f"Article Title: {doc.metadata['title']}\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)


model = LlamaCpp(
    model_path=os.environ['MODEL'],
    n_gpu_layers=os.environ['N_GPU_LAYERS'],
    n_batch=os.environ['N_BATCH'],
    n_ctx=os.environ['N_CTX'],
    f16_kv=True
)

app = FastAPI(
    title="Basic MemoryChain Server",
    version="1.0",
    description="simple langchain/langserve app."
)


# User input
class Question(BaseModel):
    """Chat history with the bot."""
    question: str = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )


# make wikiretriever work with ChatHistory below
class HistoryWiki(WikipediaRetriever):
    def _get_relevant_documents(
        self, query: Question, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return super()._get_relevant_documents(query['question'], run_manager=run_manager)


wiki = HistoryWiki(top_k_results=4, doc_content_chars_max=2048)
formatter = itemgetter("docs") | RunnableLambda(format_docs)
# subchain for generating an answer once we've done retrieval
answer = prompt | model | StrOutputParser()
# complete chain that calls wiki -> formats docs to string -> runs answer subchain -> returns just the
# answer and retrieved docs.
chain = (
    RunnableParallel(docs=wiki, question=RunnablePassthrough())
    .assign(context=formatter)
    .assign(answer=answer)
    .pick([
           "answer",
           # "docs"
           ])
).with_types(input_type=Question)


add_routes(
    app,
    chain,
    path="/chat"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
