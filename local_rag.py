from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import bs4
import os

path_to_file = "./local_data/data.txt"
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
    )

data = loader.load() # Content of the blog inside a list
#print(data[0].page_content)

# Check if RAG file exists #
if os.path.isfile(path_to_file):
    print("RAG file created already!")
else:
    print("Creating RAG txt file...")
    with open(path_to_file, 'w', encoding="utf8") as output_file:
        output_file.write(data[0].page_content)
        print("RAG file created!")

# Text Splitting #
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
#print(len(data[0].page_content))
#print(len(all_splits))

# Vector Store Embedding - after the similarity search test #
local_embedding = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(documents=all_splits, embedding=local_embedding)

llm = OllamaLLM(model="llama3.1", temperature=0.5)
RAG_TEMPLATE = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    <context>
    {context}
    </context>

    Question: {question}
    """
rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# Turn the loaded content of docs into readable strings with this function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


#chain = {"docs": format_docs} | prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Similarity Search test #
question = "What is Self Reflection?"
docs = vector_store.similarity_search(question)
#print(len(docs))
#print(docs[0])

# Run #
print(chain.invoke({"context": docs, "question": question}))

