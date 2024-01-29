from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import OnlinePDFLoader


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