from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
import streamlit as st
import os


## Sqlite fix
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


st.set_page_config(layout='wide')

col1,col2 = st.columns([2,3])

class ResumeBot:
#Load the models
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def load_pdf(self):
        #Load the PDF and create chunks
        loader = PyPDFLoader("AllAboutMe.pdf")
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        pages = loader.load_and_split(text_splitter)

        #Turn the chunks into embeddings and store them in Chroma
        vectordb=Chroma.from_documents(pages,self.embeddings)

        #Configure Chroma as a retriever with top_k=1
        self.retriever = vectordb.as_retriever(search_kwargs={"k": 5})\

#Create the retrieval chain

    def create_chain(self):
        template = """
        You are a helpful AI assistant.
        The context provided is begins with owner's resume for job applications.
        Here is a brief about the owner:
        - MSCS grad student at SJSU
        - Worked 3 years as data engineer at Fractal on ETL in python and SQL. 
        - Now, a research assistant working on ML pipelines.
        The resume contains sections covering the different aspects of the her professional life.
        The sections are Education, Technical Skills, Work Experience, Internships, Projects and Achievements.
        This is followed by a writeup containing questions interviewers and answers given by the owner.
        Answer as the owner based on this and the context provided only. Do not say anything outside the context
        context: {context}
        input: {input}
        answer:
        """
        prompt = PromptTemplate.from_template(template)
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

    #Invoke the retrieval chain

    def testing(self):

        self.load_pdf()
        self.create_chain()
        while True:

            i = input("Ask a question about me?\n")
            response= self.retrieval_chain.invoke({"input":i})
            print(response["answer"])


    def generate_response(self, text):
        response = self.retrieval_chain.invoke({"input":text})

        return response['answer'], response['context']


    def setup(self):

        self.load_pdf()
        self.create_chain()

        col3,col4 = col1.columns([5,6])

        col3.image("sq_dp.jpg", width=200)

        col4.title('Mrunmayee Dhapre')
        col4.subheader('M.S. Computer Science')
        # col4.

        tab1, tab2, tab3 = col1.tabs(['Experience','Skills','Projects'])

        # tab1.markdown(""" 
        #               ##### Data Engineer @ Fractal AI (2020-2023)
        #                 - Led the backend team developing ETL pipelines and REST APIs
        #                 - Warehousing solution with event-driven microservices architecture

        #               ##### Data Engineer Intern @ LendingClub (Summer 2024)
        #                 - LLMs for categorization of customer chat data in Snowflake
        #                 - Customizable app for clustering and chi-square analysis in Streamlit
                      
        #               ##### Research Assistant @ SJSU (2023 - present)
        #                 - Time Series Forecasting on groundwater logs with LSTM, Transformer
        #                 - Real Time Machine Learning pipeline with Airflow
                      
        #                """)
        
        de_frac = tab1.button('Data Engineer @ Fractal AI (2020-2023)')
        if de_frac:
            tab1.markdown(""" 
                    - Led the backend team developing ETL pipelines and REST APIs
                    - Warehousing solution with event-driven microservices architecture""")
            
        
        de_lc = tab1.button('Data Engineer Intern @ LendingClub (Summer 2024)')
        if de_lc:
            tab1.markdown(""" 
                    - LLMs for categorization of customer chat data in Snowflake
                    - Customizable app for clustering and chi-square analysis in Streamlit""")
            
        
        ra = tab1.button('Research Assistant @ SJSU (2023 - present)')
        if ra:
            tab1.markdown(""" 
                    - Time Series Forecasting on groundwater logs with LSTM, Transformer
                    - Real Time Machine Learning pipeline with Airflow""")

        
        # tab2.markdown('### Skills:')

        col5, col6 = tab2.columns([2,2])

        lang = col5.button('Languages')
        if lang:
            col6.markdown("""
            - Python
            - SQL
            - Java
            - JavaScript
            - Scala
            """)


        webdev = col5.button('Web Development')
        if webdev:
            col6.markdown("""
                        - Streamlit
                        - Flask
                        - FastAPI
                        - Django
                        - Angular
                        """)
            
        etl = col5.button('ETL & BigData')
        if etl:
            col6.markdown("""
                        - Airflow
                        - Snowflake
                        - Spark
                        - Hadoop
                        - Hive
                        - Knime
                        """)
            
        db = col5.button('Database Management')
        if db:
            col6.markdown("""
                        - MSSQL
                        - MySQL
                        - Neo4J
                        - MongoDB
                        - CosmosDB
                        - Oracle
                        """)
            
        azure = col5.button('Cloud Platforms')
        if azure:
            col6.markdown("""
                        - Kubernetes
                        - ServiceBus
                        - FunctionApp
                        - WebApps
                        - LogAnalytics
                        - Managed Workflows for Apache Airflow
                        """)
            
        ai = col5.button('AI/ML Modules')
        if ai:
            col6.markdown("""
                        - Keras
                        - Tensorflow
                        - Pandas
                        - Pytorch
                        - networkx
                        - langchain
                        - llamaindex
                        - openai-whisper
                        """)

        nlp = tab3.button('Sentiment Analysis for Movie Review')
        if nlp:
            tab3.markdown("""
            - Natural Language Processing in Spark
            - Detected fake reviews on a web scraped dataset
            """)

        ot = tab3.button('Predictive Modeling for Operational Technology Application')
        if ot:
            tab3.markdown("""
            - Time Series Forecasting on Industrial Control System logs
            - Hybrid LSTM and GRU models for anomaly detection
            - Published in  International Journal for Artificial Intelligence, 2021
            """)

        rag = tab3.button('Book Recommendation Chatbot')
        if rag:
            tab3.markdown("""
            - Chatbot recommendating books by descriptions and genres
            - RAG chatbot implemented with LangChain and HuggingFace
            """)

        css = '''
        <style>
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size:1.2rem;
            }
        </style>
        '''

        col1.markdown(css, unsafe_allow_html=True)

        # col1.markdown(""" 
        #               ##### Research Assistant @ SJSU (2023 - present)

        #               ##### Data Engineer Intern @ LendingClub (Summer 2024)
                      
        #               ##### Data Engineer @ Fractal AI (2020-2023)
        #               """)

        messages = col2.container(height=600)

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            messages.chat_message(msg['role']).write(msg['content'])

        if prompt := col2.chat_input("Ask a question about me"):

            messages.chat_message("user").write(prompt)
            response,context = self.generate_response(prompt)
            messages.chat_message("ai").write(response)
            print(f"Context for {prompt}:{context}")


            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "ai", "content": response})


rb = ResumeBot()
rb.setup()
# rb.testing()