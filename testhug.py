import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,LLMChain,RetrievalQA
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import csv
import pickle
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from docx import Document
from function_call import mail
# Chemin du fichier CSV de sortie
chemin_fichier_csv = "chat.csv"
colonnes = ["clien","Assistant IA"]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    # #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # with open("isotech.pkl","wb") as f:
    #     pickle.dump(vectorstore,f)
    with open("isotech.pkl","rb") as f:
        vectorstore = pickle.load(f)
    return vectorstore


def get_conversation_chain(vectorstore):
    support_template = """"
     vous ete  un assistant commercial qui  fournit de nombreux détails spécifiques au client pour le convaincre d'acheter nos produits.
      et poser des question comme  le budget de client ,le deadline de projet , lieu de projet le nom la'adress mail et numero whatsapp ... 
      Si l'IA ne connaît pas la réponse à une question, elle vous encourage à contacter le service commercial de RestoConcept au numéro 00 32 23 35 80 01.

    {context}

    Question : {question}
    """
    SALES_PROMPT = PromptTemplate(
    template=support_template, input_variables=["context", "question"])

    llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
   
    max_tokens=512,
)
    #llm=HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", model_kwargs={"temperature":0.1, "max_new_tokens":250})
    #llm=HuggingFaceHub(repo_id="tiiuae/falcon-40b", model_kwargs={"temperature":0.1})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    # conversation_chain = LLMChain(
    #     llm=llm,
    #      prompt=prompt,
    #     retriever=vectorstore.as_retriever(),
    #     memory=memory
    # )
    conversation_chain = RetrievalQA.from_chain_type(
    llm=llm,
    memory=memory,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": SALES_PROMPT},
)
    #conversation_chain= ConversationChain(memory=memory, prompt=prompt, llm=llm, retriever=vectorstore.as_retriever())
    return conversation_chain

nom_fichier = 'donnees.txt'
def handle_userinput(user_question):
    response = st.session_state.conversation({'query': user_question})
    mail(user_question)
    st.session_state.chat_history = response['chat_history']
    ligne=['client :'+ user_question]
    ajouter_ligne_texte(nom_fichier, ligne)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            ajouter_ligne_texte(nom_fichier, message.content)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            ajouter_ligne_texte(nom_fichier, message.content)    
def ajouter_ligne_texte(nom_fichier, ligne):
    with open(nom_fichier, 'a') as fichier:
        fichier.write('\n'+str(ligne) + '\n')

def main():
    
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()