# 🤖 Free Local RAG Chatbot

![Free Local RAG Chatbot](https://github.com/user-attachments/assets/06cfdf03-a6b3-4f99-bec5-03e038527830)


> Upload PDFs, ask questions, and get instant answers using cutting-edge open-source AI—no API keys, no internet required. Run entirely **locally**.

---

## 🚀 Project Overview

This is a **100% free, fully local Retrieval-Augmented Generation (RAG) chatbot** built using:

- **LangChain**: For chaining together components.
- **Hugging Face Transformers**: Free, open-source AI models (e.g., `flan-t5-base`).
- **Chroma**: Lightweight vector database.
- **Sentence Transformers**: For generating embeddings.
- **Gradio**: To create a simple, elegant web interface.

👉 No API keys. No paid services. Just pure AI on your machine.

Please note; we are using a Free Ai model, LLm result may not be consistent, for better feedback use your open ai key, 

---

## 🧩 Features

✅ Upload any PDF document.  
✅ Ask natural language questions.  
✅ AI extracts relevant knowledge & generates answers.  
✅ Runs **offline** on your local machine.  
✅ 100% free and open-source.

---

## 🛠 Tech Stack

| Layer | Tool/Library |
|-------|--------------|
| Embeddings | `sentence-transformers` → `all-MiniLM-L6-v2` |
| VectorDB | `Chroma` |
| LLM (Answer Engine) | `google/flan-t5-base` via `transformers` |
| Interface | `Gradio` |

---

## 🏗 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/free-local-rag-chatbot.git
cd free-local-rag-chatbot
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

Required libraries:
```
gradio
langchain
chromadb
sentence-transformers
transformers
```

---

## 💻 Usage (Recommended: Jupyter Notebook)

### Option 1: Run Jupyter Notebook

```bash
jupyter notebook
```
Open the provided notebook and follow the step-by-step cells.

### Option 2: Run as Python App (Optional)

```bash
python app.py
```

Gradio will open at `http://127.0.0.1:7860`

---

## 📄 Example Code (Jupyter Notebook Friendly)

```python
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Language Model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

# Text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Main QA Function
def answer_question(file, question):
    loader = PyPDFLoader(file.name)
    documents = loader.load()
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(chunks, embedding=embedding_model)
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa_chain.run(question)

# Gradio Interface
gr.Interface(
    fn=answer_question,
    inputs=[
        gr.File(label="Upload PDF", file_types=[".pdf"]),
        gr.Textbox(label="Ask a Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Free Local RAG Chatbot",
    description="Upload a PDF and ask questions—no keys, no APIs, fully offline."
).launch()
```

---

## 📌 Example Use Cases

- Legal document Q&A 📑
- Academic papers summary 📚
- Business reports analysis 📊
- Personal knowledge bases 🧠

---

## 🎨 Screenshots

![image](https://github.com/user-attachments/assets/b5512257-4d42-4480-9ad7-af0821258196)

![image](https://github.com/user-attachments/assets/6d3861bb-07f7-4162-b5bb-aacf805662e8)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## ⭐ License

This project is licensed under the MIT License.

---

> Made with ❤️ by [Your Name] | Free and open-source AI for everyone.
