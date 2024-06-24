# Run this cell and paste the API key in the prompt
import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv(".env")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                 temperature=0.7, top_p=0.85)
def load_content(file_path):
    """Load content from a file"""
    with open(file_path, "r") as f:
        return f.read()
llm_prompt_template = """Viết một bản tóm tắt đầy đủ thông tin về những điều sau đây:
"{text}"
Tóm tắt :"""
llm_prompt = PromptTemplate.from_template(llm_prompt_template)

print(llm_prompt)
for file in os.listdir("corpus"):
    path = os.path.join("corpus", file)
    #path = "corpus/benh-dau-mua"
    doc =load_content(path)
    new_path =os.path.join("corpus_summarize",os.path.basename(path))
    if os.path.exists(new_path)==True:  
        continue
    stuff_chain = (
        # Extract data from the documents and add to the key `text`.
        {
            "text": lambda doc: doc
        }
        | llm_prompt         # Prompt for Gemini
        | llm                # Gemini function
        | StrOutputParser()  # output parser
    )
    #print(doc)
    summarize_doc =stuff_chain.invoke(doc)
    print(summarize_doc)
    with open(new_path, "w") as f:
        f.write(summarize_doc)