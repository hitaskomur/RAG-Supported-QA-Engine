import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
import os
import tempfile

# Streamlit uygulamasının başlığı
st.title("Multi PDF QA Engine")

# Groq API anahtarını Streamlit arayüzünden al
groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
os.environ["GROQ_API_KEY"] = groq_api_key

# PDF dosyalarını yükleme alanı
uploaded_files = st.file_uploader("Uplado PDf Files", type="pdf", accept_multiple_files=True)

# Oturum durumunu (session state) kullanarak vektör veritabanını sakla
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
    st.session_state.retriever = None

# PDF dosyaları değiştiyse veya ilk defa yüklendiyse vektör veritabanını oluştur
if uploaded_files and (st.session_state.vectordb is None or st.session_state.retriever is None):
    with st.spinner("We are working on it..."):
        # 1. PDF'leri yükle ve metin parçalarına ayır
        docs = []
        for uploaded_file in uploaded_files:
            # Geçici bir dosya oluştur ve PDF içeriğini kaydet
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # PyPDFLoader'ı geçici dosya yoluyla kullan
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs.extend(text_splitter.split_documents(pages))

            # Geçici dosyayı sil
            os.remove(temp_file_path)

        # 2. Vektör veritabanını oluştur
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectordb = Chroma.from_documents(
            collection_name="Rag_Info",
            documents=docs,
            embedding=embeddings,
            persist_directory="./chroma_langchain_db"  # Kalıcılık için bir dizin belirtin
        )

        # 3. Retriever'ı oluştur
        retriever = vectordb.as_retriever()

        # Vektör veritabanını ve retriever'ı oturum durumunda sakla
        st.session_state.vectordb = vectordb
        st.session_state.retriever = retriever

# Soru sorma alanı
question = st.text_input("Enter Your Question:")

# "Cevapla" butonu
if st.button("Bring Me Answer"):
    if not uploaded_files:
        st.warning("Please upload at least 1 file.")
    elif not question:
        st.warning("Please enter a question.")
    elif not groq_api_key:
        st.warning("Please ensure that your Groq API Key is correct.")
    elif st.session_state.retriever is None:
        st.warning("Please Load Your PDF Files and Let Us Make It Ready.")
    else:
        with st.spinner("Looking for Answer..."):
            # 4. LLM'yi (Groq) oluştur
            llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

            # 5. Prompt şablonunu oluştur
            prompt = PromptTemplate.from_template(
                """Use the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Use three sentences maximum and keep the answer as concise as possible.
                Always say "thanks for asking!" at the end of the answer.

                {context}

                Question: {question}

                Helpful Answer:"""
            )

            # 6. Soru-cevaplama zincirini oluştur
            output_parser = StrOutputParser()
            rag_chain = (
                {
                    "context": st.session_state.retriever,  # Oturumdaki retriever'ı kullan
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | output_parser
            )

            # 7. Soruyu zincire gönder ve cevabı al
            answer = rag_chain.invoke(question)

            # 8. Cevabı görüntüle
            st.write("Answer:", answer)