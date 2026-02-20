# index.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
from pypdf import PdfReader
import time

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATA_DIR = "medical_data"
EMBEDDING_MODEL = "models/text-embedding-004"
CHUNK_SIZE = 500      # Updated small chunk size
CHUNK_OVERLAP = 200   # Updated overlap
BATCH_SIZE = 100      # Process 100 chunks at a time
LOG_FILE = "indexed_files.log"

# --- INITIALIZE SERVICES ---
print("Initializing services...")
pc = Pinecone(api_key=PINECONE_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

# --- HELPER FUNCTIONS ---
def get_pdf_text(pdf_path):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def get_text_chunks(text):
    chunks = []
    # Loop with the stride (step) of CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunks.append(text[i:i + CHUNK_SIZE])
    return chunks

def get_indexed_files():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, 'r') as f:
        return set(line.strip() for line in f.readlines())

def add_file_to_log(filename):
    with open(LOG_FILE, 'a') as f:
        f.write(filename + '\n')

# --- MAIN SCRIPT ---
def main():
    PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
    index = pc.Index(host = PINECONE_INDEX_HOST)

    indexed_files = get_indexed_files()
    print(f"Found {len(indexed_files)} already indexed files.")

    print("Checking for new documents to process...")
    new_docs_processed = 0
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            if filename in indexed_files:
                continue

            print(f"Processing new file: {filename}...")
            file_path = os.path.join(DATA_DIR, filename)

            full_text = get_pdf_text(file_path)
            if not full_text:
                continue
                
            chunks = get_text_chunks(full_text)
            print(f"  > Split into {len(chunks)} chunks. Starting upload...")
            
            # --- OPTIMIZED BATCH UPLOAD ---
            for i in range(0, len(chunks), BATCH_SIZE):
                batch_chunks = chunks[i:i + BATCH_SIZE]
                vectors_to_upsert = []
                
                try:
                    # 1. GENERATE EMBEDDINGS (Batch Call - 1 Request instead of 100)
                    response = genai.embed_content(
                        model=EMBEDDING_MODEL,
                        content=batch_chunks,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                    embeddings = response['embedding'] # This is a list of vectors

                    # 2. PREPARE VECTORS
                    for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                        chunk_id = f"{filename}-chunk-{i+j}"
                        vectors_to_upsert.append({
                            "id": chunk_id,
                            "values": embedding,
                            "metadata": {"text": chunk}
                        })
                    
                    # 3. UPSERT TO PINECONE
                    if vectors_to_upsert:
                        index.upsert(vectors=vectors_to_upsert)
                        print(f"  > Uploaded batch {int(i/BATCH_SIZE) + 1} ({len(batch_chunks)} chunks)")
                    
                    # Sleep briefly between BATCHES, not chunks
                    time.sleep(1) 

                except Exception as e:
                    print(f"Error processing batch {i}: {e}")
            
            # --- END OF FILE PROCESSING ---
            add_file_to_log(filename)
            print(f"Successfully indexed and logged {filename}")
            new_docs_processed += 1

    if new_docs_processed == 0:
        print("\nNo new documents to process.")
    else:
        print(f"\nProcessing complete. Indexed {new_docs_processed} new document(s).")

if __name__ == "__main__":
    main()