# langchain-k8s
Set of kubernetes deployment files, Dockerfiles, etc to run Langchain applications in a local k8s

This includes a Redis install to use for context storage, but assumes that it will use a local, in-RAM
voyager for a vectordb. 

Also assuming it will use a local LLM model, usually mistral (so that they fit in RAM on my rPi cluster)
