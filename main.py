import argparse
import warnings
from argparse import Namespace

from langchain_community.llms import GPT4All
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from processing_utils import (create_prompt, extract_embeddings, format_docs,
                              get_model_path_from_directory, read_pdf)

warnings.filterwarnings('ignore',
                        category=UserWarning,
                        message='TypedStorage is deprecated')


def main(args: Namespace) -> None:
    """
    Main function for processing a question in the question-answering system.

    :param args: Command-line arguments containing
    the link to the PDF file, device, and question.
    :type args: Namespace
    :return: None
    """
    splits = read_pdf(args.pdf_link)
    retriever = extract_embeddings(splits, args.device)
    model_path = get_model_path_from_directory('./model')
    llm = GPT4All(model=model_path, max_tokens=3000)
    rag_prompt = create_prompt()
    question = args.question
    qa_chain = ({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
                | rag_prompt
                | llm
                | StrOutputParser())

    print(qa_chain.invoke(question))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_link", type=str)
    parser.add_argument("--question", type=str)
    parser.add_argument("--device", type=str, choices=['cpu', 'gpu'])
    args = parser.parse_args()
    main(args)