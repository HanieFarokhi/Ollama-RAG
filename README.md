# Ollama-RAG
# Setting Up RAG with Ollama and ChromaDB

## Step 1: Install and Run Docker Containers

Ensure Docker is running on your system.

### 1.1 Run ChromaDB as a Docker Container
Run the following command to start ChromaDB:
```sh
docker run -d --name chromadb -p 8000:8000 chromadb/chroma
docker run -d --rm --name chromadb -p 8000:8000 -v ./chroma:/chroma/chroma -e IS_PERSISTENT=TRUE -e ANONYMIZED_TELEMETRY=TRUE chromadb/chroma
```
- `-d` runs the container in the background.
- `--name chromadb` gives the container a name.
- `-p 8000:8000` exposes it on port 8000.

To check if it's running:
```sh
docker ps
curl http://localhost:8000/api/v1/heartbeat
wget http://0.0.0.0:8000
```
If it returns `{ "heartbeat": "OK" }`, ChromaDB is working.

### 1.2 Run Ollama as a Docker Container
Run the following command:
```sh
docker run -d --name ollama -p 11434:11434 ollama/ollama
```
- Exposes Ollama’s API on port 11434.

To verify, run:
```sh
curl http://localhost:11434/api/tags
```
If it responds with available models, Ollama is working.

### 1.3 Run Open-webui
```sh
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -e OLLAMA_API_BASE_URL=http://192.168.203.206:11434 -v open-webui-data:/app/backend/data --name open-webui --restart unless-stopped ghcr.io/open-webui/open-webui:latest
```

## Step 2: Download an LLM Model for Ollama
Ollama needs a model to generate responses. You can download one like Mistral:
```sh
ollama pull mistral
```
To list available models:
```sh
ollama list
```
To test if Ollama works, run:
```sh
ollama run mistral "Hello, how are you?"
```

## Step 3: Install Python and Required Libraries
You'll need Python for the backend API.

### 3.1 Install Python Dependencies
First, install Python (if not already installed), then create a virtual environment:
```sh
python -m venv rag_env
source rag_env/bin/activate   # For Linux/macOS
rag_env\Scripts\activate      # For Windows
```
Now install dependencies:
```sh
pip install chromadb requests fastapi uvicorn ollama llama-index langchain llama-index-readers-file llama-index-embeddings-openai pypdf
```

## Step 4: Create a Python API to Connect ChromaDB and Ollama
Create a file called `rag_api.py` and add the following code:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import chromadb

app = FastAPI()

class Document(BaseModel):
    text: str

class QueryRequest(BaseModel):
    user_input: str

chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(name="rag_collection")

@app.post("/add_document/")
def add_document(doc: Document):
    embedding = ollama.embeddings(model="mistral", prompt=doc.text)["embedding"]
    collection.add(ids=[doc.text[:10]], embeddings=[embedding], metadatas=[{"text": doc.text}])
    return {"message": "Document added successfully"}

@app.post("/query/")
def query_rag(request: QueryRequest):
    embedding = ollama.embeddings(model="mistral", prompt=request.user_input)["embedding"]
    results = collection.query(query_embeddings=[embedding], n_results=3)
    retrieved_texts = [item["text"] for item in results["metadatas"][0]]
    context = "\n".join(retrieved_texts)
    prompt = f"Use the following information to answer:\n{context}\n\nUser: {request.user_input}"
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return {"response": response["message"]["content"]}
```

## Step 5: Run the API
Save the file and run:
```sh
uvicorn rag_api:app --host 0.0.0.0 --port 5000 --reload
```
Now your API is running at `http://localhost:5000`.

To restart the API:
```sh
uvicorn rag_api:app --host 0.0.0.0 --port 5000 --reload
```

## Step 6: Test the System
### 6.1 Add a document:
```sh
curl -X POST "http://localhost:5000/add_document/" -H "Content-Type: application/json" -d '{"text": "Ollama is an open-source AI model server that runs LLMs locally."}'
```
### 6.2 Query the system:
```sh
curl -X POST "http://localhost:5000/query/" -H "Content-Type: application/json" -d '{"user_input": "What is Ollama?"}'
```
You should get a response generated using the retrieved information.

---

## Upload PDF File Support

### Step 1: Install Required Libraries
```sh
pip install pypdf langchain chromadb fastapi uvicorn ollama
```
- `pypdf` → Extracts text from PDFs
- `langchain` → Helps with text splitting
- `chromaDB` → Vector database
- `FastAPI` → API for communication

### Step 2: Update `rag_api.py` to Handle PDFs
```python
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import ollama
import chromadb
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI()

chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(name="rag_collection")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

class QueryRequest(BaseModel):
    user_input: str

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_reader = pypdf.PdfReader(file.file)
    text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    text_chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(text_chunks):
        embedding = ollama.embeddings(model="mistral", prompt=chunk)["embedding"]
        collection.add(ids=[f"{file.filename}_{i}"], embeddings=[embedding], metadatas=[{"text": chunk}])
    return {"message": f"PDF {file.filename} processed and stored in ChromaDB"}
```

### Step 3: Restart FastAPI Server
```sh
uvicorn rag_api:app --host 0.0.0.0 --port 5000 --reload
```

### Step 4: Upload and Query PDF Data
Upload a PDF:
```sh
curl -X POST "http://localhost:5000/upload_pdf/" -F "file=@path/to/your.pdf"
```

