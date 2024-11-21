import os
import re
import time
import asyncio

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone,ServerlessSpec

from db.Query_Database import Query_Database
from utils.Helper_Functions import create_text_from_dict,create_uuid4_from_list
class Chatbot:
    def __init__(self):
        self._llm_model = None

    def __Extract_Sql_Query__(self,text):
        # Define the regex pattern
        pattern = r'\|\|\|(.*?)\|\|\|'

        # Search for the SQL query between delimiters
        match = re.search(pattern, text, re.DOTALL)  # Use re.DOTALL to match across multiple lines
        if match:
            return match.group(1).strip()  # Extract the query and remove extra whitespace

        return None
    def __Get_Sql_query__(self,userPrompt):
        groq_api = os.getenv("GROQ_API_KEY")
        self._llm_model = ChatGroq(api_key = groq_api,model="llama3-8b-8192")

        schema_prompt = """
        Given the following database schema and a user's query, generate an appropriate SQL query  and make sure the output only contains the sql no extra text or words other than the query. make sure the query part start and end with three (|||).
        '. 
        format:

        ### Database Schema
        1. **Table: Customer**
           - **customer_id** (Primary Key): A unique identifier for each customer (e.g., "7590-VHVEG").
           - **gender**: Gender of the customer (e.g., "Male", "Female").
           - **senior_citizen**: Indicates whether the customer is a senior citizen (1 for Yes, 0 for No).
           - **partner**: Indicates if the customer has a partner ("Yes" or "No").
           - **dependents**: Indicates if the customer has dependents ("Yes" or "No").

        2. **Table: Services**
           - **service_id** (Primary Key): A unique identifier for each service record.
           - **customer_id** (Foreign Key): References the `customer_id` in the Customer table.
           - **phone_service**: Indicates if the customer has phone service ("Yes", "No").
           - **multiple_lines**: Indicates if the customer has multiple phone lines ("Yes", "No").
           - **internet_service**: Type of internet service used by the customer (e.g., "DSL", "Fiber optic", "No").
           - **online_security**: Indicates if online security is enabled ("Yes", "No").
           - **online_backup**: Indicates if online backup is enabled ("Yes", "No").
           - **device_protection**: Indicates if device protection is enabled ("Yes", "No").
           - **tech_support**: Indicates if tech support is enabled ("Yes", "No").
           - **streaming_tv**: Indicates if streaming TV is enabled ("Yes", "No").
           - **streaming_movies**: Indicates if streaming movies is enabled ("Yes", "No").

        3. **Table: Contract**
           - **contract_id** (Primary Key): A unique identifier for each contract record.
           - **customer_id** (Foreign Key): References the `customer_id` in the Customer table.
           - **contract_type**: Type of contract (e.g., "Month-to-month", "One year", "Two year").
           - **paperless_billing**: Indicates if paperless billing is enabled ("Yes", "No").
           - **payment_method**: Payment method used by the customer (e.g., "Electronic check", "Mailed check").
           - **tenure**: The number of months the customer has been with the service.

        4. **Table: Billing**
           - **billing_id** (Primary Key): A unique identifier for each billing record.
           - **customer_id** (Foreign Key): References the `customer_id` in the Customer table.
           - **monthly_charges**: Monthly charges for the customer.
           - **total_charges**: Total charges for the customer.

        5. **Table: Churn**
           - **churn_id** (Primary Key): A unique identifier for each churn record.
           - **customer_id** (Foreign Key): References the `customer_id` in the Customer table.
           - **churn**: Indicates whether the customer has churned ("Yes" or "No").
        """
        messages = [
            SystemMessage(content=schema_prompt),
            HumanMessage(content=userPrompt),
        ]

        response = self._llm_model.invoke(messages)

        print(response.content)

        final_sql_query = self.__Extract_Sql_Query__(response.content)

        return final_sql_query

    def __Create_Contextual_Documents_From_List__(self,texts,sql_query):
        sys_msg = """<sql_query>
            {WHOLE_DOCUMENT}
            </sql_query>""".format(WHOLE_DOCUMENT=sql_query)
        chunk_msg = """Here is the chunk we want you describe reading the user prompt
            <chunk>
            {CHUNK_CONTENT}
            </chunk>
            Please give a short succinct context to describe this chunk according to the sql query for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
        documents = []

        for i in texts:
            messages = [
                SystemMessage(content=sys_msg),
                HumanMessage(content=chunk_msg.format(CHUNK_CONTENT=i)),
            ]
            response = self._llm_model.invoke(messages)
            document = Document(
                page_content=response.content + i,
                metadata={"Database": "query result"},
            )
            documents.append(document)
        return documents

    # async def __Add_To_Vector_Store__(self,index,embeddings_model,documents,uuids):
    #     vector_store = PineconeVectorStore(index=index, embedding=embeddings_model)
    #     await vector_store.add_documents(documents=documents, ids=uuids)
    #
    #     # The next line will wait for the above operations to complete.
    #     print("Documents added successfully!")
    #     return vector_store

    def __RAG_Implementation__(self,userPrompt,texts,sql_query):
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


        pc = Pinecone(api_key='pcsk_2vPbjG_EGKrzCoUgTQ84DmFpu3SmWGQuroDCxPXeVwLrhmrVABh1WTBJcm3odatxLDK7h5')
        # try:
        #     pc.delete_index('sqlchatbot2')
        #     time.sleep(2)
        # except:
        #     pass
        index_name = "sqlchatbot2"  # change if desired
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        index = pc.Index(index_name)
        documents = self.__Create_Contextual_Documents_From_List__(texts,sql_query)
        uuids = create_uuid4_from_list(texts)

        vector_store =  PineconeVectorStore(index=index, embedding=embeddings_model)
        vector_store.add_documents(documents=documents, ids=uuids)

        print(vector_store.similarity_search('count the users that have no dependents'))
        retriever = vector_store.as_retriever(search_kwargs={'k': 10})



        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. . Use three sentences maximum and keep the "
            "answer concise."
            "if you donot know what you are answering. say i do not know the answer"
            "use the user question and the additional information to generate a response"
            "the information from the retriever context is the result of the query generated from user question from an sql database"
            "make sure to just describe the answer in appropriate way"
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self._llm_model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": userPrompt})
        return response["answer"]

    def Query_Chatbot(self,userPrompt):
        print('chatbot on work')
        sql_query = self.__Get_Sql_query__(userPrompt)
        sql_response = Query_Database(sql_query)
        texts = [create_text_from_dict(item) for item in sql_response]
        rag_response = self.__RAG_Implementation__(userPrompt,texts,sql_query)
        return rag_response









