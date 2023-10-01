import os
import sys

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = True

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]


from langchain.document_loaders import DirectoryLoader

directory = './articles/'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
print(f'no of documents=',len(documents))

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents,chunk_size=10000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(f'no of chunks=',len(docs))

from langchain.embeddings.cohere import CohereEmbeddings
import constants

os.environ["COHERE_API_KEY"] = constants.COHERE_APIKEY
embeddings = CohereEmbeddings(model = "multilingual-22-12")

from langchain.vectorstores import Chroma
persist_directory = "chroma_db"
vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_directory
)

vectordb.persist()

from langchain.llms import Cohere
from langchain.chains import ConversationalRetrievalChain
chain = ConversationalRetrievalChain.from_llm(
  llm=Cohere(cohere_api_key=constants.COHERE_APIKEY),
  retriever=vectordb.as_retriever(search_kwargs={"k": 1}),
)

chat_history=[]
while True:
    if not query:
        query = input("Search: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()

    print(f'query={query}')  
    if query.split()[0].lower() in ['who', 'where','when']:
        results = chain({"question": query,"chat_history": chat_history})
        result=results["answer"]
        chat_history.append((query, result))
    else:
        result = vectordb.similarity_search(query)[0]

    print(f'result=',result)

    query = None

