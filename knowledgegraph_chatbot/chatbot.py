import logging
from uuid import uuid4
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain import hub
import streamlit as st
import os
import shelve

# Configure logging
logging.basicConfig(level=logging.INFO)

# Generate unique session ID
SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

# Initialize the ChatOpenAI model
try:
    llm = ChatOpenAI(openai_api_key="YOUR_OPENAI_API_KEY_HERE")
    logging.info("OpenAI model initialization successful")
except Exception as e:
    logging.error(f"OpenAI model initialization failed: {e}")
    raise

# Establish connection to the Neo4j database
try:
    graph = Neo4jGraph(
        url="bolt+s:ur url",
        username="neo4j",
        password="YOUR_NEO4J_PASSWORD_HERE"
    )
    logging.info("Neo4j connection successful")
except Exception as e:
    logging.error(f"Neo4j connection failed: {e}")
    raise

# Define a prompt template for term chat
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert in medical terminology. You provide information about terms and their origins."),
        ("human", "{input}"),
    ]
)

# Define term chat pipeline
term_chat = prompt | llm | StrOutputParser()

# Function to retrieve message history
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Function to query the database for terms, synonyms, and semantic types
def query_neo4j(code):
    try:
        if not code:
            return "Please provide a valid code to search."

        term_query = (
            "MATCH (n:Term) WHERE n.code = $code "
            "OPTIONAL MATCH (n)-[:HAS_DEFINITION]->(d:Definition) "
            "OPTIONAL MATCH (n)-[:HAS_SYNONYM]->(s:Synonym) "
            "OPTIONAL MATCH (n)-[:HAS_SEMANTIC_TYPE]->(st:SemanticType) "
            "RETURN n.term AS term, n.code AS code, d.definition AS definition, "
            "collect(s.synonym) AS synonyms, st.type AS semantic_type"
        )
        
        with graph._driver.session() as session:
            results = session.run(term_query, code=code).data()

        if results:
            record = results[0]
            synonyms = ', '.join(record['synonyms']) if record['synonyms'] else 'None'
            semantic_type = record['semantic_type'] if record['semantic_type'] else 'None'
            return (
                f"Term: {record['term']}, Code: {record['code']}\n"
                f"Definition: {record['definition']}\n"
                f"Synonyms: {synonyms}\n"
                f"Semantic Type: {semantic_type}"
            )
        else:
            return None

    except Exception as e:
        logging.error(f"Query Neo4j failed: {e}")
        return "An error occurred while querying Neo4j."

# Function to query OpenAI model for general queries
def query_openai_model(query):
    try:
        response = llm.ask(query)
        return response
    except Exception as e:
        logging.error(f"OpenAI query failed: {e}")
        return "An error occurred while querying OpenAI model."

# Function to handle the overall querying logic
def handle_query(input_query):
    # Extract the code from the input query if possible
    code = extract_code(input_query)
    
    # Query Neo4j first
    neo4j_response = query_neo4j(code)
    if neo4j_response:
        return neo4j_response
    else:
        # If no data is found in Neo4j, fallback to querying OpenAI
        return query_openai_model(input_query)

# Function to extract code from the query (simple extraction logic)
def extract_code(query):
    import re
    match = re.search(r'\bC\d{6}\b', query)
    return match.group(0) if match else None

# Create tools for the agent
tools = [
    Tool.from_function(
        name="Medical Information",
        description="Queries Neo4j for medical terms, definitions, synonyms, and semantic types.",
        func=handle_query,
    ),
    Tool.from_function(
        name="OpenAI Query",
        description="Query the OpenAI model for general questions.",
        func=query_openai_model,
    ),
]

# Create the agent using the defined tools and prompt
try:
    agent_prompt = hub.pull("hwchase17/react-chat")
    agent = create_react_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    logging.info("Agent creation successful")
except Exception as e:
    logging.error(f"Agent creation failed: {e}")
    raise

# Define the chat agent with message history
chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Streamlit UI
st.title("CRDChat: The ChatGPT of the CRDC!")
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Sidebar with a button to delete chat history
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        try:
            response = chat_agent.invoke({"input": prompt}, {"configurable": {"session_id": SESSION_ID}})
            full_response = response["output"]
        except Exception as e:
            logging.error(f"Error during chat: {e}")
            full_response = "An error occurred during the chat."
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Save chat history after each interaction
save_chat_history(st.session_state.messages)
