import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set in the .env file")


# --------------------------------------------------
# 2. Load the PDF
# --------------------------------------------------

PDF_PATH = "notes/college_notes.pdf"

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print(f"Loaded {len(documents)} pages from the PDF.")


# --------------------------------------------------
# 3. Split the documents into chunks
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

def clean_text(text):
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")

for chunk in chunks:
    chunk.page_content = clean_text(chunk.page_content)

print(f"Created {len(chunks)} chunks.")
# --------------------------------------------------
# 4. Create Gemini embedding model
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# --------------------------------------------------
# 5. Create Chroma vector store
# --------------------------------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="college_notes",
    persist_directory="./chroma_db"
)

print("PDF content stored in ChromaDB.")


# --------------------------------------------------
# 6. Create retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


# --------------------------------------------------
# 7. Create Gemini LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
)


# --------------------------------------------------
# 8. Create the prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are a helpful college study assistant.

Answer the question using ONLY the information provided
in the context below.

If the answer cannot be found in the context,
say:

"I couldn't find that information in your notes."

Do not make up information.

Context:
{context}

Question:
{question}
""")


# --------------------------------------------------
# 9. Convert retrieved Documents into text
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# --------------------------------------------------
# 10. Build the RAG chain
# --------------------------------------------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# --------------------------------------------------
# 11. Chat loop
# --------------------------------------------------

print("\n----------------------------------------")
print("📚 College Notes RAG Chatbot")
print("----------------------------------------")
print("Ask questions about your notes.")
print("Type 'exit' to quit.\n")


while True:

    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    answer = rag_chain.invoke(question)

    print("\nBot:", answer)
    print()