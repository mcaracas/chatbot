# Import pysqlite3 and modify the sys.modules to replace sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import toml
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI

# Load the secrets from the specified path
secrets_path = "/workspaces/chatbot/secrets.toml"
secrets = toml.load(secrets_path)

# Access the OpenAI API key from the loaded secrets
api_key = secrets["openai_api_key"]

# Initialize the OpenAI LLM with "GPT-4 Mini" model
openai_llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

st.title("📄 PDF Chatbot with RAG")
st.write(
    "This app allows you to chat with a PDF document. It uses OpenAI's GPT-4 Mini model to generate responses "
    "based on the content of the PDF."
)

# Set the file path directly
file_path = "Codigo_Trabajo_RPL.pdf"  # The PDF file name

if file_path:
    try:
        # Create an OpenAI client.
        client = OpenAI(api_key=api_key)

        # Step 1: Load the PDF
        with st.spinner("Loading and processing the PDF..."):
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Step 2: Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # Step 3: Create embeddings and load them into a vector store
            embeddings = OpenAIEmbeddings(api_key=api_key)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            # Step 4: Build the RAG Chain
            system_prompt = (
                "Eres un asistente para tareas de preguntas y respuestas. "
                "Utiliza los siguientes fragmentos de contexto recuperados para responder "
                "la pregunta. Si no sabes la respuesta, di que no lo sabes. Usa un máximo de tres oraciones y mantén la "
                "respuesta concisa."
                "\n\n"
                "{context}"
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(openai_llm, prompt_template)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Create a session state variable to store the chat messages. This ensures that the
        # messages persist across reruns.
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display the existing chat messages via `st.chat_message`.
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Create a chat input field to allow the user to enter a message.
        if prompt := st.chat_input("Ask a question about the PDF content:"):

            # Store and display the current prompt.
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate a response using the RAG pipeline.
            with st.spinner("Generating answer..."):
                results = rag_chain.invoke({"input": prompt})
                response = results['answer']

            # Stream the response to the chat and store it in session state.
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
