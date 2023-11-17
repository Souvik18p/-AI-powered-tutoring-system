import os
import textwrap
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS 
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

# Set the Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load the text document
document_path = r'C:\Users\souvi\OneDrive\Desktop\ai p\data1.txt'
loader = TextLoader(document_path)
document = loader.load()

# Function to wrap text while preserving newlines
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(document)

# Create embeddings for the document chunks
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# Load the question-answering chain
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.8, "max_length": 512})
chain = load_qa_chain(llm, chain_type="stuff")

# Start an interactive loop
while True:
    # Ask the user for a question
    query_text = input("Ask a question (or type 'goodbye roy' to exit): ")

    # Check if the user wants to exit
    if query_text.lower() == "goodbye roy":
        print("Goodbye!")
        break

    # Perform similarity search
    docs_result = db.similarity_search(query_text)

    # Display the answers
    answers = chain.run(input_documents=docs_result, question=query_text)

    if answers:
        combined_answer = ''.join([str(answer).strip() for answer in answers])

        # Post-process to add spaces between words
        words = combined_answer.split()
        result = ' '.join(words)
        print("Answer:")
        print(result)
    else:
        print("No answers found.")
