# main.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

# --- LOAD ENV and CONFIG ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST") 

EMBEDDING_MODEL = "models/text-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"

# --- INITIALIZE SERVICES ---
genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)

# --- FASTAPI APP ---
app = FastAPI()

# --- DATA MODELS ---
class QueryRequest(BaseModel):
    query: str
    # Expecting a list of dicts: [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]
    conversation_history: List[Dict[str, str]] = [] 

class QueryResponse(BaseModel):
    response: str
    sources: List[str]

@app.post("/query", response_model=QueryResponse)
async def query_model(request: QueryRequest):
    try:
        user_question = request.query
        history = request.conversation_history
        model = genai.GenerativeModel(GENERATION_MODEL)

        # --- PROCESS CONVERSATION HISTORY ---
        # We turn the list of objects into a clean text string for the LLM
        history_text = ""
        if history:
            history_text = "--- Start of Conversation History ---\n"
            for turn in history:
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                history_text += f"{role}: {content}\n"
            history_text += "--- End of Conversation History ---\n"

        # --- Step 1: Intent Recognition ---
        # We include history here so it knows that "it" refers to the previous topic
        intent_prompt = f"""
        Classify the user's latest query as 'medical' or 'conversational'.
        
        {history_text}
        
        Latest Query: "{user_question}"
        Classification:
        """
        
        intent_response = model.generate_content(intent_prompt)
        intent = intent_response.text.strip().lower()

        # --- Step 2: Route the request ---
        if "medical" in intent:
            # --- RAG Process ---
            
            # Note: Ideally, we should rewrite the query based on history here, 
            # but for now, we embed the raw question.
            query_embedding = genai.embed_content(
                model=EMBEDDING_MODEL, content=user_question, task_type="RETRIEVAL_QUERY"
            )['embedding']

            retrieval_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

            context = ""
            retrieved_sources = []
            for match in retrieval_results['matches']:
                context += match['metadata']['text'] + "\n---\n"
                retrieved_sources.append("Retrieved Medical Document")

            prompt_template = f"""
            You are a medical information assistant.
            
            **INSTRUCTIONS:**
            1. Use the Retrieved Context to answer the Latest Question.
            2. Use the Conversation History to understand what the user is referring to (e.g., if they say "it" or "that muscle").
            3. Format your answer as a point-wise list (using bullets or numbers).
            4. For each point, provide a detailed and clear explanation.
            5. Do not add information not found in the context.

            {history_text}

            Retrieved Context:
            {context}
            
            Latest Question: {user_question}
            
            Answer (as a detailed, point-wise list):
            """
            final_response = model.generate_content(prompt_template)
            return QueryResponse(response=final_response.text, sources=list(set(retrieved_sources)))

        else:
            # --- Conversational Process ---
            conversational_prompt = f"""
            You are a friendly chatbot assistant. 
            Use the history to carry on the conversation naturally.
            Give a brief, conversational reply.

            {history_text}

            User: "{user_question}"
            Assistant:
            """
            final_response = model.generate_content(conversational_prompt)
            return QueryResponse(response=final_response.text, sources=[])

    except Exception as e:
        print(f"An error occurred: {e}")
        return QueryResponse(response="Sorry, I encountered an error.", sources=[])

@app.get("/")
def read_root():
    return {"status": "Medical Chatbot API is running."}\
    

