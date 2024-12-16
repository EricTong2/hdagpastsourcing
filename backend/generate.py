import os
import argparse
from dotenv import load_dotenv
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI 
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.prompts import PromptTemplate

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key

pc = Pinecone(api_key=pinecone_api_key)

index_name = "case-study-index"

def load_and_process_data(filepath):
    cases_df = pd.read_csv(filepath)
    cases_df["combined_text"] = cases_df.apply(
        lambda row: (
            f"Case Name: {row['Case Name']}. "
            f"Semester: {row['Semester']}. "
            f"Company Name: {row['Company Name']}. "
            f"Description: {row['Detailed Case Description']}. "
            f"Outcome: {row['Outcome']}. "
            f"Tech Stack: {row['Tech Stack']}. "
            f"KPI: {row['KPIs']}. "
            f"Quoted Price: {row['Quoted Price']}."
        ),
        axis=1
    )

    cases_df = transform_webscraped_df(cases_df)
    return cases_df

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

# process csv of companies and generate emails for each company
def process_companies_and_generate_emails(companies_filepath, retriever):
    companies_df = pd.read_csv(companies_filepath)

    emails = []
    for _, row in companies_df.iterrows():
        company_info = {
            "Company Name": row.get("Company Name for Emails", ""),
            "Industry": row.get("Industry", ""),
            "Technologies": row.get("Technologies", ""),
            "Keywords": row.get("Keywords", "")
        }

        email_content = generate_email(company_info, retriever)
        emails.append({
            "Company Name": row.get("Company Name for Emails", ""),
            "Email Content": email_content
        })
    # save to generated_emails.csv
    emails_df = pd.DataFrame(emails)
    emails_df.to_csv("generated_emails.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate emails for companies.')
    parser.add_argument('--company', action='store_true', help='Generate email for a single company.')
    parser.add_argument('--company_name', type=str, help='Company Name')
    parser.add_argument('--industry', type=str, help='Industry')
    parser.add_argument('--technologies', type=str, help='Technologies')
    parser.add_argument('--keywords', type=str, help='Keywords')
    parser.add_argument('--companies_file', type=str, default='companies.csv', help='CSV file containing companies information')
    args = parser.parse_args()

    # temporary dummy data filepath
    cases_df = load_and_process_data("dummy_data.csv")

    vectorstore = setup_pinecone_index(cases_df)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    if args.company:
        company_info = {
            "Company Name": args.company_name or "",
            "Industry": args.industry or "",
            "Technologies": args.technologies or "",
            "Keywords": args.keywords or ""
        }
        process_single_company(company_info, retriever)
    else:
        process_companies_and_generate_emails(args.companies_file, retriever)
