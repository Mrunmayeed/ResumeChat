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
        st.session_state['chain'] = self.retrieval_chain

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

    def education(self, tab):
        ms = tab.button('M. S. Computer Science')
        if ms:
            tab.markdown(""" 
            - San Jose State University
            - Courses: Big Data, Machine Learning, Cloud Computing, Graph Neural Network   
            """)

        bs = tab.button('B. Tech. Computer Engineering')
        if bs:
            tab.markdown(""" 
            - VJTI (Mumbai University)
            - Courses: Database Management, Operating Systems, Data Mining, Blockchain   
            """)

    def workex(self, tab):
        de_frac = tab.button('Data Engineer @ Fractal AI (2020-2023)')
        if de_frac:
            tab.markdown(""" 
                    - Led the backend team developing ETL pipelines and REST APIs
                    - Warehousing solution with event-driven microservices architecture""")
            
        
        de_lc = tab.button('Data Engineer Intern @ LendingClub (Summer 2024)')
        if de_lc:
            tab.markdown(""" 
                    - LLMs for categorization of customer chat data in Snowflake
                    - Customizable app for clustering and chi-square analysis in Streamlit""")
            
        
        ra = tab.button('Research Assistant @ SJSU (2023 - present)')
        if ra:
            tab.markdown(""" 
                    - Time Series Forecasting on groundwater logs with LSTM, Transformer
                    - Real Time Machine Learning pipeline with Airflow""")
            
    def skills(self, tab):
        col1, col2 = tab.columns([2,2])

        lang = col1.button('Languages')
        if lang:
            col2.markdown("""
            - Python
            - SQL
            - Java
            - JavaScript
            - Scala
            """)


        webdev = col1.button('Web Development')
        if webdev:
            col2.markdown("""
                        - Streamlit
                        - Flask
                        - FastAPI
                        - Django
                        - Angular
                        """)
            
        etl = col1.button('ETL & BigData')
        if etl:
            col2.markdown("""
                        - Airflow
                        - Snowflake
                        - Spark
                        - Hadoop
                        - Hive
                        - Knime
                        """)
            
        db = col1.button('Database Management')
        if db:
            col2.markdown("""
                        - MSSQL
                        - MySQL
                        - Neo4J
                        - MongoDB
                        - CosmosDB
                        - Oracle
                        """)
            
        azure = col1.button('Cloud Platforms')
        if azure:
            col2.markdown("""
                        - Kubernetes
                        - ServiceBus
                        - FunctionApp
                        - WebApps
                        - LogAnalytics
                        - Managed Workflows for Apache Airflow
                        """)
            
        ai = col1.button('AI/ML Modules')
        if ai:
            col2.markdown("""
                        - Keras
                        - Tensorflow
                        - Pandas
                        - Pytorch
                        - networkx
                        - langchain
                        - llamaindex
                        - openai-whisper
                        """)
            
    def projects(self, tab):
        nlp = tab.button('Sentiment Analysis for Movie Review')
        if nlp:
            tab.markdown("""
            - Natural Language Processing in Spark
            - Detected fake reviews on a web scraped dataset
            """)

        ot = tab.button('Predictive Modeling for Operational Technology Application')
        if ot:
            tab.markdown("""
            - Time Series Forecasting on Industrial Control System logs
            - Hybrid LSTM and GRU models for anomaly detection
            - Published in  International Journal for Artificial Intelligence, 2021
            """)

        rag = tab.button('Book Recommendation Chatbot')
        if rag:
            tab.markdown("""
            - Chatbot recommendating books by descriptions and genres
            - RAG chatbot implemented with LangChain and HuggingFace
            """)

    def links(self, col):
        # log1,log2,log3 = col1.columns([2,2,2])
        # col1.markdown(
        # '''
        #     [![Title](https://badgen.net/badge/icon/linkedin?color=orange&icon=https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg&label)](https://in.linkedin.com/in/mrunmayee-dhapre) 
        #     [![Title](https://badgen.net/badge/icon/GitHub?color=orange&icon=github&label)](https://github.com/Mrunmayeed) 
        #     [![Title](https://badgen.net/badge/icon/medium?color=orange&icon=medium&label)](https://medium.com/@mrunmayee.dhapre)
        # ''')
        col.markdown(
        '''
            [![Title](https://badgen.net/badge/icon/GitHub?color=orange&icon=github&label)](https://github.com/Mrunmayeed) 
        ''')

        col.markdown(
        '''
            [![Title](https://badgen.net/badge/icon/medium?color=orange&icon=medium&label)](https://medium.com/@mrunmayee.dhapre)
        ''')

        # st.logo('github-logo.png',link='https://github.com/Mrunmayeed')
        st.logo('linkedin-logo.png',link='https://in.linkedin.com/in/mrunmayee-dhapre')
    
    def sidepane(self, col1):
        col3,col4 = col1.columns([5,6])

        col3.image("dp_photo.png", width=200)

        col4.title('Mrunmayee Dhapre')
        # col4.subheader('M.S. Computer Science')
        self.links(col4)

        tab1, tab2, tab3, tab4 = col1.tabs(['Education','Experience','Skills','Projects'])
        self.education(tab1)
        
        self.workex(tab2)

        self.skills(tab3)

        self.projects(tab4)
        
        # tab2.markdown('### Skills:')

        css = '''
        <style>
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size:1.2rem;
            }
        </style>
        '''

        col1.markdown(css, unsafe_allow_html=True)

    def showchats(self, col2):
        messages = col2.container(height=600)

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            messages.chat_message(msg['role']).write(msg['content'])

        if prompt := col2.chat_input("Ask a question about me"):

            messages.chat_message("ai").write(prompt)
            response,context = self.generate_response(prompt)
            messages.chat_message("user").write(response)
            print(f"Context for {prompt}:{context}")


            st.session_state.messages.append({"role": "ai", "content": prompt})
            st.session_state.messages.append({"role": "user", "content": response})

    def setup(self):

        if 'chain' in st.session_state:
            self.retrieval_chain=st.session_state['chain']
        else:
            self.load_pdf()
            self.create_chain()


        col1,col2 = st.columns([2,3])
        self.sidepane(col1)
        self.showchats(col2)        


rb = ResumeBot()
rb.setup()
# rb.testing()