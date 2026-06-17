import os
import joblib
import pandas as pd
import pdfplumber
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PDF_PATH = "Guidelines.pdf"

VECTORSTORE_PATH = "vectorstore"

model = joblib.load("loan_pipeline.pkl")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=25
)

index_file = os.path.join(VECTORSTORE_PATH, "index.faiss")
pkl_file = os.path.join(VECTORSTORE_PATH, "index.pkl")

if os.path.exists(index_file) and os.path.exists(pkl_file):

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

else:

    docs = []

    with pdfplumber.open(PDF_PATH) as pdf:

        for page in pdf.pages:

            text = page.extract_text()

            if text:

                chunks = splitter.split_text(text)
                docs.extend(chunks)

    vectorstore = FAISS.from_texts(docs, embeddings)

    vectorstore.save_local(VECTORSTORE_PATH)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.5
)


class LoanRequest(BaseModel):

    Age: int
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: float
    NumCreditLines: float
    InterestRate: float
    LoanTerm: float
    DTIRatio: float

    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str

    question: str


@app.get("/")
def home():

    return {
        "message": "AI Loan Assistant Running"
    }

@app.post("/loan-assessment")

def loan_assessment(data: LoanRequest):


    df = pd.DataFrame([{

        "Age": data.Age,
        "Income": data.Income,
        "LoanAmount": data.LoanAmount,
        "CreditScore": data.CreditScore,
        "MonthsEmployed": data.MonthsEmployed,
        "NumCreditLines": data.NumCreditLines,
        "InterestRate": data.InterestRate,
        "LoanTerm": data.LoanTerm,
        "DTIRatio": data.DTIRatio,
        "Education": data.Education,
        "EmploymentType": data.EmploymentType,
        "MaritalStatus": data.MaritalStatus,
        "HasMortgage": data.HasMortgage,
        "HasDependents": data.HasDependents,
        "LoanPurpose": data.LoanPurpose,
        "HasCoSigner": data.HasCoSigner
    }])

    pd_value = model.predict_proba(df)[0][1]

    risk_score = int(900 - (pd_value * 600))



    vector_store = vectorstore


    prompt_template = PromptTemplate(

        input_variables=["context", "question"],

        template="""
You are a loan approval assistant for Indian Banking.
Treat all income, loan amounts, property values, EMI amounts, and other monetary figures as Indian Rupees (INR).Also consider Loan Term in months, Interest Rate in percentage, and DTI Ratio as a decimal value.
Answer using the context.Using the context and the applicant information, answer the user's question. 
If question is not related to the context say:
"I don't know the answer to that question based on the provided information."

Context:
{context}

Question:
{question}

Answer:
"""
    )


    qa_chain = RetrievalQA.from_chain_type(

        llm=llm,

        retriever=vector_store.as_retriever(
            search_kwargs={"k": 3}
        ),

        chain_type="stuff",

        chain_type_kwargs={
            "prompt": prompt_template
        },

    )

    final_query = f"""

Applicant Information:
{df.to_dict(orient='records')[0]}

Risk Score:
{risk_score}

User Question:
{data.question}
"""

    result = qa_chain.invoke({
        "query": final_query
    })


    return {

        "risk_score": risk_score,

        "confidence": float(pd_value),

        "answer": result["result"],

    }
