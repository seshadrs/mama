import streamlit as st
import os
from stqdm import stqdm
import multiprocessing
from annotated_text import annotated_text
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA

import config
import ingest
import vectordb
import llm


def main():

    st.title("Medical Archive Memory Aid (MAMA)")

    with st.sidebar:
        pinecone_index_namespace = st.text_input(
            "Pinecone Index namespace to use", value=config.PINECONE_INDEX_NAMESPACE
        )
        openai_model_name = st.text_input(
            "OpenAI model name", value=config.OPENAI_MODEL_NAME
        )
        if st.button("Update"):
            config.PINECONE_INDEX_NAMESPACE = pinecone_index_namespace
            config.OPENAI_MODEL_NAME = openai_model_name

    tab1, tab2, tab3 = st.tabs(
        ["Documents Index", "Search documents", "QNA with documents"]
    )

    with tab1:
        folder_path = st.text_input(
            "Path to local folder containing medical documents",
            value=config.MEDICAL_DOCS_DIRPATH,
        )
        if st.button("Add to index"):
            if folder_path and pinecone_index_namespace:
                config.MEDICAL_DOCS_DIRPATH = folder_path
                st.write(
                    f"Adding PDFs in folder: {folder_path} to Pinecone index namespace: {pinecone_index_namespace}"
                )

                pdf_files = [
                    f
                    for f in os.listdir(config.MEDICAL_DOCS_DIRPATH)
                    if f.endswith(".pdf")
                ]
                file_paths = [
                    os.path.join(config.MEDICAL_DOCS_DIRPATH, pdf_file)
                    for pdf_file in pdf_files
                ]

                with multiprocessing.Pool() as pool:
                    medical_documents = list(
                        stqdm(
                            pool.imap(ingest.process_pdf, file_paths),
                            total=len(file_paths),
                            desc=f"Processing PDFs",
                        )
                    )
                vectordb.add_documents_to_index(medical_documents)
                # TODO: detect file duplicates and warn user.

            else:
                st.error("Please enter a folder path and a Pinecone Index name")

        st.divider()

        st.subheader("Documents in index:")
        st.write(vectordb.all_pinecone_docs())
    with tab2:
        query = st.text_input("Enter a query to search for in the documents")
        num_docs = st.number_input(
            "Maximum number of documents to return", value=10, min_value=1
        )
        if st.button("Search"):
            st.divider()

            if query:
                results = vectordb.vectorstore.similarity_search(query, k=int(num_docs))
                for result in results:
                    annotated_text(
                        (result.metadata["date"], "", "silver"),
                        " " * 4,
                        result.metadata["medical_specialty"],
                        " " * 4,
                        (result.metadata["type"], "", "yellow"),
                    )
                    st.write(
                        f"_{result.metadata['medical_professional_name']}_ @ {result.metadata['medical_institution_name']}"
                    )
                    st.caption("SUMMARY: " + result.metadata["summary"])
                    st.code(result.page_content, wrap_lines=True)
                    st.caption("FILEPATH: " + result.metadata["filepath"])
                    st.divider()
    with tab3:
        query = st.text_input("Enter a query to start the conversation")
        if st.button("Converse"):
            if query:
                st.write(llm.conversational_qa.run(query))
                # TODO: maintain and display conversation history.


if __name__ == "__main__":
    main()
