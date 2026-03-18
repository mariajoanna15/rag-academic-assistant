#  Academic Research Assistant (RAG)

An AI-powered research assistant that allows users to upload PDFs and ask questions.
The system retrieves relevant information using FAISS and generates grounded answers using a local LLM (LLaMA3 via Ollama).

## 🛠 Tech Stack

* Python
* Streamlit
* LangChain
* FAISS (Vector Database)
* HuggingFace Embeddings
* Ollama (LLaMA3)


## 📂 Project Structure

```bash
rag-academic-assistant/
│── app.py                # Main Streamlit app
│── ingest.py            # PDF processing & vector DB creation
│── requirements.txt     # Dependencies
│── .gitignore
│── README.md
│
├── data/                # Uploaded PDFs (auto-created)
├── vectorstore/         # FAISS index (auto-created)
└── venv/                # Virtual environment (not included)
```


## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-academic-assistant.git
cd rag-academic-assistant
```


### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```


### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


### 4. Install Ollama

Download and install: https://ollama.com

Then run:

```bash
ollama pull llama3
```


### 5. Run the Application

```bash
streamlit run app.py
```


## 📌 How to Use

1. Open the web app
2. Upload PDF files from the sidebar
3. Ask questions in the chat
4. View answers with sources and page numbers


## 🎥 Demo


https://github.com/user-attachments/assets/7e824a49-d9fa-4c2e-94f5-8b87cd7d20a0



##  Notes

* `data/` and `vectorstore/` folders are created automatically
* Do NOT upload large PDFs to GitHub
* Everything runs locally — no API key required




