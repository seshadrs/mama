from langchain_openai import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA

import config
import vectordb

# TODO: add LLM caching.

model = ChatOpenAI(model=config.OPENAI_MODEL_NAME, temperature=0)

conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True
)

conversational_qa = RetrievalQA.from_chain_type(
    llm=model, chain_type="stuff", retriever=vectordb.vectorstore.as_retriever()
)
