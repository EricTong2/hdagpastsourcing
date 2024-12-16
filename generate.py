import os
import json
import base64
import awswrangler as wr
import pandas as pd
from io import StringIO
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.prompts.prompt import PromptTemplate

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key

# Pinecone setup
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "case-study-index"

def setup_pinecone_index(cases_df):
    embeddings = OpenAIEmbeddings()
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index, embeddings, "text")
    
    texts = cases_df["combined_text"].tolist()
    metadatas = cases_df.to_dict('records')

    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    
    return vectorstore

# generate email for compnay based on company_info and similar cases
def generate_email(company_info, retriever):
    company_text = f"""
    Company Name: {company_info.get('Company Name', '')}
    Industry: {company_info.get('Industry', '')}
    Technologies: {company_info.get('Technologies', '')}
    Keywords: {company_info.get('Keywords', '')}
    """

    retrieved_cases = retriever.invoke(company_text)

    retrieved_cases_text = "\n".join([
        f"Case Name: {doc.metadata['Case Name']}, Outcome: {doc.metadata['Outcome']}, Quoted Price: {doc.metadata['Quoted Price']}"
        for doc in retrieved_cases
    ])

    prompt_template = """
    You are a sourcing analyst for a Data Analytics Consulting Group at Harvard College. You have been tasked with reaching out to a client and providing an email to introduce our services. You have access to a database of case studies from previous clients.

    **Company Information:**
    {company_info}

    **Relevant Case Studies:**
    {retrieved_cases}

    **Instructions:**
    - Write a concise, engaging email introducing our services following roughly the structure below:
        1. Introduction to Harvard Data Analytics Group; Harvard Undergraduate Data Analytics Group (HDAG) is a non-profit student organization at Harvard College dedicated to helping organizations make smarter, data-driven decisions and achieve their strategic goals by translating their data into meaningful and actionable information. 
        2. Mention the relevant case studies from the database that are similar to the client's needs. Use the case studies to highlight the successful outcomes and the value we can provide. Be sure to mention the case primarily by the company that it was done for.
        3. Include a call to action to schedule a meeting or a call to discuss further.
        
        Here's an example for Novartis. 
    I hope this email finds you well. I understand your time is valuable, but please give me two minutes of your time. My name is Kevin Liu, and I help represent the Harvard Undergraduate Data Analytics Group (HDAG). HDAG is a non-profit student organization at Harvard College dedicated to helping organizations make smarter, data-driven decisions and achieve their strategic goals by translating data into meaningful and actionable insights.
 	We understand that as the Chief Strategy and Growth at Novartis, a leading organization in the pharmaceuticals space, you may face challenges meeting your ESG expectations. HDAG is uniquely equipped to help you achieve your goals by leveraging our proven expertise and diverse project experience.
	HDAG has worked with a wide variety of clients across industries, including Coca-Cola, the World Health Organization, and Hewlett-Packard. More relevant to your case, we have worked with UNIDO to help identify and visualize key indicators for their ESG goals in developing areas. 
	I would love to explore potential engagement opportunities between Novartis and HDAG. I completely understand if you are unable to respond at this time. When you’re able, I would love to find time with you or a colleague to schedule a quick chat about how HDAG can help support your goals!

        Here's the used template:
    I hope this email finds you well. My name is [Your Name], and I help represent the Harvard Undergraduate Data Analytics Group (HDAG). HDAG is a non-profit student organization at Harvard College dedicated to helping organizations make smarter, data-driven decisions and achieve their strategic goals by translating data into meaningful and actionable insights.
    We understand that as the {title} at {company}, a leading organization in the {industry} space, you may face challenges in [specific area from "brief summary"]. HDAG is uniquely equipped to help you achieve your goals by leveraging our proven expertise and diverse project experience.
    """

    prompt = PromptTemplate(input_variables=["company_info", "retrieved_cases"], template=prompt_template)
    final_prompt = prompt.format(company_info=company_text, retrieved_cases=retrieved_cases_text)
    # Generate email using LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    messages = [{"role": "system", "content": "You are a professional email writer."},
                {"role": "user", "content": final_prompt}]
    response = llm.invoke(messages)
    
    return response.content

def lambda_handler(event, context):
    """AWS Lambda handler function."""
    try:
        # Decode the base64-encoded CSV file
        file_content = base64.b64decode(event["body"]).decode("utf-8")
        companies_df = pd.read_csv(StringIO(file_content))

        # Load case study data (can be stored in S3 for better scalability)
        case_study_s3_path = "s3://your-bucket/case-study-data.csv"
        cases_df = wr.s3.read_csv(case_study_s3_path)

        # Prepare Pinecone retriever
        vectorstore = setup_pinecone_index(cases_df)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Generate emails
        emails = []
        for _, row in companies_df.iterrows():
            company_info = {
                "Company Name": row.get("Company Name for Emails", ""),
                "Industry": row.get("Industry", ""),
                "Technologies": row.get("Technologies", ""),
                "Keywords": row.get("Keywords", "")
            }
            email_content = generate_email(company_info, retriever)
            emails.append({"Company Name": row["Company Name for Emails"], "Email Content": email_content})

        # Convert results to CSV
        emails_df = pd.DataFrame(emails)
        output_csv = StringIO()
        emails_df.to_csv(output_csv, index=False)
        output_csv.seek(0)

        # Return the CSV as the response
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "text/csv"},
            "body": output_csv.getvalue()
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
