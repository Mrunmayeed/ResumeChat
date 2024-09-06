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
        return response['answer']


    def setup(self):

        self.load_pdf()
        self.create_chain()
        
        # with st.form('my_form'):

        # text = st.text_area('Ask a question about me', 'Tell me about yourself')
        # print(text)
        # submitted = st.button('Ask me')

        # st.metric('','Ask questions here')
        # if submitted:
        #     st.info(self.generate_response(text))

        # prmt = st.chat_input("Tell me about yourself")
        # if prmt:
        #     st.write(f"User has sent the following prompt: {prmt}")

        # with st.sidebar:


        # if 'hist' in st.session_state:
        #     messages = st.session_state['hist']
        #     with messages:
        #         st.write('done')

        # else:
        #     messages = st.container(height=600)
        #     if prompt := st.chat_input("Say something"):
        #         messages.chat_message("user").write(prompt)
        #         messages.chat_message("human").write(f"Echo: {prompt}")
            
        #     st.session_state['hist'] = messages
        #     st.write('wrote msg')

        col3,col4 = col1.columns([2,3])

        col3.image("sq_dp.jpg", width=200)

        col4.title('Mrunmayee Dhapre')
        col4.subheader('M.S. Computer Science')
        # col4.

        col1.markdown(""" 
                      ##### Research Assistant @ SJSU (2023 - present)
                        - Time Series Forecasting on groundwater logs with LSTM, Transformer
                        - Real Time Machine Learning pipeline with Airflow

                      ##### Data Engineer Intern @ LendingClub (Summer 2024)
                        - LLMs for categorization of customer chat data in Snowflake
                        - customizable app for clustering and chi-square analysis in Streamlit
                      
                      ##### Data Engineer @ Fractal AI (2020-2023)
                        - Led the backend team developing ETL pipelines and REST APIs
                        - Warehousing solution with event-driven microservices architecture
                        - POCs involving Azure Data Factory, Stream Analytics, graphQL
                      """)
        
        # col1.markdown(""" 
        #               ##### Work Experience:
        #               - Research Assistant @ SJSU (2023 - present)
        #               - Data Engineer Intern @ LendingClub (Summer 2024)
        #               - Data Engineer @ Fractal AI (2020-2023)
        #               """)
        

        # col1.markdown('##### Projects:')
        # col1.markdown(""" 
        #             - [Sentiment Analysis for Movie Review (Spark)](https://github.com/Mrunmayeed/nlp_in_spark)
        #             - [Predictive Modeling for Operational Technology Application](https://www.aut.upt.ro/~rprecup/IJAI_90.pdf)
        #             - [Book Recommendation Chatbot](https://github.com/Mrunmayeed/ChatbotRAG)
        #               """)


        # col1.markdown('### Skills:')

        # col5, col6, col7 = col1.columns([2,2,2])
        # col5.markdown("""
        #               ##### WebDev
        #               - Python
        #               - SQL
        #               - Streamlit
        #               - Flask
        #               - FastAPI
        #             """)
        # col6.markdown("""
        #               ##### ETL & BigData
        #               - Airflow
        #               - Snowflake
        #               - Spark
        #               - Hadoop
        #               - Hive
        #             """)
        # col7.markdown("""
        #               ##### Cloud: Azure
        #               - Kubernetes
        #               - ServiceBus
        #               - FunctionApp
        #               - WebApps
        #               - LogAnalytics
        #             """)
        


        messages = col2.container(height=600)

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            messages.chat_message(msg['role']).write(msg['content'])

        if prompt := col2.chat_input("Ask a question about me"):

            messages.chat_message("user").write(prompt)
            response = self.generate_response(prompt)
            messages.chat_message("ai").write(response)


            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "ai", "content": response})


rb = ResumeBot()
rb.setup()
# rb.testing()