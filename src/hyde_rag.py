# --- Setting API KEY ---
import os

os.environ['GOOGLE_API_KEY']='AIzaSyApejrG65DFwqnIzWuVRxMdVt17SBhrhgI'

# --- Model Loading ---
# Import the necessary modules from the langchain_google_genai package.
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Create a ChatGoogleGenerativeAI Object and convert system messages to human-readable format.
llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

# Create a GoogleGenerativeAIEmbeddings object for embedding our Prompts and documents
Embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# --- Data Loading ---
# Import the WebBaseLoader class from the langchain_community.document_loaders module.
from langchain_community.document_loaders import WebBaseLoader

# Create a WebBaseLoader object with the URL of the blog post to load.
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/")

# Load the blog post and store the documents in the `docs` variable.
docs = loader.load()
# --- Splitting / Creating Chunks ---
# Import the RecursiveCharacterTextSplitter class from the 
# langchain.text_splitter module.
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Create a RecursiveCharacterTextSplitter object using the provided 
# chunk size and overlap.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,
chunk_overlap=50)


# Split the documents in the `docs` variable into smaller chunks and 
#store the resulting splits in the `splits` variable.
splits = text_splitter.split_documents(docs)
# --- Creating Embeddings by Passing Hyde Embeddings to Vector Store ---
from langchain_community.vectorstores import Chroma


# passing the hyde embeddings to create and store embeddings
vectorstore = Chroma.from_documents(documents=splits,
                                   collection_name='my-collection',
                                   embedding=Embeddings)


# Creating Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# Importing the Prompt Template
from langchain.prompts import ChatPromptTemplate
# Creating the Prompt Template
template = """For the given question try to generate a hypothetical answer\
Only generate the answer and nothing else:
Question: {question}
"""

Prompt = ChatPromptTemplate.from_template(template)
query = Prompt.format(question = 'What are different Chain of Thought(CoT) Prompting?')

hypothetical_answer = llm.invoke(query).content
print(hypothetical_answer)
# retrieval with hypothetical answer/document
similar_docs = retriever.get_relevant_documents(hypothetical_answer)
for doc in similar_docs:
 print(doc.page_content)
 print()
# retrieval with original query
similar_docs = retriever.get_relevant_documents('What are different \
Chain of Thought(CoT) Prompting?')

for doc in similar_docs:
 print(doc.page_content)
 print()