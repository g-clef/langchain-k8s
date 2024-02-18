import os

from fastapi import FastAPI
from langserve import add_routes

from ddg_web_research import DDGWebResearchRetriever
from duckduckgo_search_wrapper import AsyncDuckDuckGoSearchAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import ChatPromptTemplate

# LLM
llm = LlamaCpp(
    model_path=os.environ['MODEL'],
    n_gpu_layers=os.environ['N_GPU_LAYERS'],
    n_batch=os.environ['N_BATCH'],
    n_ctx=os.environ['N_CTX'],
    f16_kv=True
)


embeddings = LlamaCppEmbeddings(
    model_path=os.environ['MODEL'],
    n_gpu_layers=os.environ['N_GPU_LAYERS'],
    n_batch=os.environ['N_BATCH'],
    n_ctx=os.environ['N_CTX'],
    f16_kv=True
)

# Vectorstore
vectorstore = Chroma(
    embedding_function=embeddings, persist_directory=os.environ['VECTOR_STORAGE_PATH']
)

# Search
search = AsyncDuckDuckGoSearchAPIWrapper()


# Initialize
web_research_retriever = DDGWebResearchRetriever.from_llm(
    vectorstore=vectorstore, llm=llm, search=search
)

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm, retriever=web_research_retriever
)


prompt = ChatPromptTemplate.from_template("{question}")


app = FastAPI(
    title="DDG QA MemoryChain Server",
    version="1.0",
    description="simple langchain/langserve app."
)


add_routes(
    app,
    prompt | qa_chain,
    path="/qa"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
