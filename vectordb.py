import os
from langchain_community.vectorstores import Pinecone as PineconeVectorstore
from langchain_community.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
import pandas as pd
from datetime import datetime

import config

pc = Pinecone()


# Reference: https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/08-langchain-retrieval-agent.ipynb
embedder = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)
try:
    pc_index = pc.Index(
        config.PINECONE_INDEX, namespace=config.PINECONE_INDEX_NAMESPACE
    )
except Exception as e:
    pc_spec = ServerlessSpec(cloud="aws", region="us-east-1")
    pc_index = pc.create_index(
        config.PINECONE_INDEX,
        dimension=config.EMBEDDING_MODEL_DIMENSIONALITY,  # dimensionality of ada 002
        metric="dotproduct",
        spec=pc_spec,
    )
    # wait for index to be initialized
    while not pc.describe_index(config.PINECONE_INDEX).status["ready"]:
        time.sleep(1)
# initialize the vector store object
vectorstore = PineconeVectorstore(
    pc_index,
    embedder.embed_query,
    config.METADATA_TEXT_FIELD,
    namespace=config.PINECONE_INDEX_NAMESPACE,
)


def all_pinecone_docs():
    # hacky way to get all docs in the index. fetching docs by ids is not performant.
    docs = vectorstore.similarity_search(
        query="", k=10000  # our search query  # return
    )
    doc_dicts = []
    for d in docs:
        doc_dict = d.metadata
        doc_dict[config.METADATA_TEXT_FIELD] = d.page_content
        doc_dicts.append(doc_dict)
    return pd.DataFrame(sorted(doc_dicts, key=lambda d: d["date"], reverse=True))


def add_documents_to_index(medical_documents):
    ids = [f"{d.filepath}#{datetime.now()}" for d in medical_documents]
    embeds = embedder.embed_documents([str(d) for d in medical_documents])
    metadatas = [d.metadata() for d in medical_documents]
    pc_index.upsert(
        vectors=zip(ids, embeds, metadatas), namespace=config.PINECONE_INDEX_NAMESPACE
    )
