# pdf_qa

# Online PDF-based Question-Answering System

This project implements a system capable of processing PDF documents sourced online and generating answers to user queries. 

Employing natural language processing (NLP) techniques, the system offers functionalities like reading and splitting PDFs, extracting embeddings, and utilizing language models.

# Project Structure

### **1. processing_util.py**: Contains utility functions for PDF processing, embedding extraction, and prompt creation.

- read_pdf(pdf_link: str) -> list: Loads and reads a PDF from the provided online link, returning a list of document splits.
- format_docs(docs: list) -> str: Formats the content of a list of documents for further processing.
- extract_embeddings(splits: list, device: str) -> VectorStoreRetriever: Extracts embeddings from document splits based on the chosen device.
- get_model_path_from_directory(directory_path: str) -> Optional[str]: Retrieves the model path from the specified directory.
- create_prompt() -> PromptTemplate: Creates a prompt template for language models.

### **2. main.py**: The main script orchestrating the question-answering system.

- main(args: Namespace) -> None: The main function for processing a user's question. It reads the online PDF, extracts embeddings, uses a language model, and prints the generated answer.

## Instalation

#### 1. Update Local Copy:

Execute the following command to update your local copy of the project and fetch the latest changes:
```
git pull master
```
#### 2. Install Dependencies:

Install the required dependencies by running the following command:
```
pip install -r requirements.txt
```
#### 3. Create .env File:

Place your API key in the .env file at the project's root.
#### 4. Download GPT4ALL Model and place it in the './model' folder.:

https://gpt4all.io/models/gguf/nous-hermes-llama2-13b.Q4_0.gguf

## Using

Use the following command to run the program with the specified flags:
```
python3 main.py --pdf_link <your_pdf_link> --question <your_question> --device <cpu/gpu>
```
## Examples

#### In English
```
python3 main.py --pdf_link ""https://www.jetir.org/papers/JETIR1804008.pdf"" --question "What is this article about?" --device "cpu"
```
```
This article discusses how the Internet and social marketing has changed not only the way businesses operate 
but also how consumers choose their products, and it proposes a four-stage model that focuses on today's consumers 
using social media for advocating products and purchasing based on reviews and backing received.
```
#### In Russian
```
python3 main.py --pdf_link "https://cyberleninka.ru/article/n/chelovek-v-informatsionnom-obschestve/pdf" --question "О чем эта статья?" --device "cpu"
```
```
Эта статья посвящена проблемам информационного общества и их влиянию на человека. 
Автор подчеркивает, что развитие технологий создает широкие возможности для воздействия на сознание человека и манипуляции им. 
В связи с этим он указывает на необходимость системы образования, которая отражала бы новые вызовы 21 века.
```