from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

from medical_document import MedicalDocument
import config
import llm

set_llm_cache(InMemoryCache())

medical_document_parser = llm.model.with_structured_output(MedicalDocument.Parse)


def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return MedicalDocument(pages, medical_document_parser)
