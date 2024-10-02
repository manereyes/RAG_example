from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

path_to_file = "./local_data/diet.txt"
loader = TextLoader(file_path=path_to_file, encoding='utf8')
data = loader.load()
#print(data[0].page_content)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=60)
all_splits = text_splitter.split_documents(data)

embedding = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
retrieved_docs = retriever.invoke("What is my Wednesday lunch?")
#print(retrieved_docs[0].page_content)

llm = OllamaLLM(model="llama3.1", temperature=0.5)

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
"""

prompt = PromptTemplate.from_template(template)

#example = prompt.format(
#    context="yes", question="no"
#)
#print(example)

# Formatting #
def format_docs(data):
    return "\n\n".join(doc.page_content for doc in data)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("What is Wednesday lunch?"):
    print(chunk, end="", flush=True)