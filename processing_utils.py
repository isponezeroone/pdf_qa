import os

import faiss
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.embeddings import (HuggingFaceEmbeddings,
                                            HuggingFaceInferenceAPIEmbeddings)
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStoreRetriever


def read_pdf(pdf_link: str) -> list:
    """
    Loading from a link and reading a PDF

    Parameters:
    - pdf_link (str): Link to the PDF file.

    Returns:
    - list: A list containing the document split into parts.
    """
    loader = OnlinePDFLoader(pdf_link)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits


def format_docs(docs):
    """
    Format the content of a list of documents.

    Parameters:
    - docs (list): A list of documents,
    where each document is expected to have a 'page_content' attribute.

    Returns:
    - str: A formatted string containing the concatenated content
    of all documents, separated by two newline characters.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def extract_embeddings(splits: list, device: str) -> VectorStoreRetriever:
    """
    Extract embeddings

    Parameters:
    - splits (list): A list containing the document split into parts.
    - device (str): The device performing the embeddings (cpu or gpu).

    Returns:
    - VectorStoreRetriever: Base Retriever class for VectorStore.
    """

    if device == 'gpu':
        model_kwargs = {'device': 'cuda'}
    elif device == 'cpu':
        model_kwargs = {'device': 'cpu'}

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs=model_kwargs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity",
                                         search_kwargs={"k": 5})
    return retriever

def extract_embeddings_inference_api(splits: list,
                                     device: str) -> VectorStoreRetriever:
    """
    Extract embeddings using HuggingFace API

    Parameters:
    - splits (list): A list containing the document split into parts.
    - device (str): The device performing the embeddings (cpu or gpu).

    Returns:
    - VectorStoreRetriever: Base Retriever class for VectorStore.
    """
    # Loading environment variables from the .env file
    load_dotenv()

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HF_API_KEY"),
        model_name="sentence-transformers/all-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    if device == 'gpu':
        resources = faiss.StandardGpuResources()
        vectorstore.index = faiss.index_cpu_to_gpu(resources, 0,
                                                   vectorstore.index)
    elif device == 'cpu':
        pass
    retriever = vectorstore.as_retriever(search_type="similarity",
                                         search_kwargs={"k": 5})

    return retriever

def get_model_path_from_directory(directory_path: str) -> Optional[str]:
    """
    Get model path from directory
    Parameters:
    - directory_path (str): The path to the directory
    where the model file is located.

    Returns:
    - str: The path to the model.
    """
    files = os.listdir(directory_path)

    files = [
        f for f in files if os.path.isfile(os.path.join(directory_path, f))
    ]

    if len(files) == 1:
        file_path = os.path.join(directory_path, files[0])
        return file_path
    else:
        print("Error: There must be exactly one file in the folder.")
        sys.exit()