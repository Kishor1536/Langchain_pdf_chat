from groq import Groq
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
import os
import streamlit as st
from pdfreader import SimplePDFViewer
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
api = os.getenv("api")

os.environ["api"] = os.getenv("GROQ_API_KEY")

def main():
    st.title("Chat with pdf🐷")

    pdf = st.file_uploader("Upload ur files")
    
    userin = st.text_input("Enter ur question")

    if pdf is not None:
        pdfreader = PdfReader(pdf)
        text  =""
        for page in pdfreader.pages:
            text += page.extract_text()
            
        textsplit = CharacterTextSplitter(
            separator="\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2" ,
        model_kwargs={"device": "cpu"}
        )
        chunks = textsplit.split_text(text)
        llm = ChatGroq(model = "llama3-8b-8192", temperature=0)
        knowledge_base = FAISS.from_texts(chunks,embeddings)
        
        

        if userin is not None:
            docs = knowledge_base.similarity_search(userin)
            

            chain = load_qa_chain(llm,"stuff")
            response = chain.run(input_documents = docs,question = userin)
            st.write(response)

if __name__ == "__main__":
    main()