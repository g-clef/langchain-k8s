import os

from fastapi import FastAPI
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes

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

prompt = ChatPromptTemplate.from_template("You are a helpful agent. Answer a question about {topic}")

add_routes(
    app,
    prompt | model,
    path="/chat"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
