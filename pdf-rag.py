from langchain_community.document_loaders import PyPDFLoader
import getpass
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


os.environ["LANGCHAIN_TRACING_V2"] = "true"
file_path = "./documents/ASWAACC_Automatic_Semantic_Web_Annotatio.pdf"
loader = PyPDFLoader(file_path)

parser = StrOutputParser()
docs = loader.load()

llm = ChatOpenAI(model="gpt-4o")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)




question_answer_chain = create_stuff_documents_chain(llm, prompt, output_parser = parser)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# rag_chain = rag_chain | parser

results = rag_chain.invoke({"input": "Who are the authors of this paper?"})
print(results)
# for chunk in rag_chain.stream("Who are the authors of this paper?"):
#     print(chunk, end="", flush=True)