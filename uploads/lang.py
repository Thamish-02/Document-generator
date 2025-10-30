from langchain_Openai import ChatOpenAI
from langchain_core.prompt import ChatPromptTemplate
from lanchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import laod_dotenv

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAT_API_KEY")

# langmith tracking
os.environ["LANGMITH_TRACKING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# prompt template

prompt = ChatPromptTemplate.from_template(
    [
        ("system", "you are a helpful assistent. Please response to the user")
        ("user", " Question: {question}"),
    ]
)

#streamlit framework
st.title("Langchain with OpenAI and Streamlit")
input_text = st.text_input("Enter your question:")

#openAI

llm = ChatOpenAI(model = "gpt-3.5-turbo")
output_parser = StrOutputParser()
chain= prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
    