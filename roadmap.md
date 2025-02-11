# Setup Plan for Retrieval-Augmented Generation (RAG) Model using Azure OpenAI Chatbot

This document provides a step-by-step guide to extending the existing Azure OpenAI Chatbot with Retrieval-Augmented Generation (RAG) capabilities, allowing the chatbot to answer questions based on internal documents.

---

## **1. Understanding the Current Setup**
### Existing Repository Information
- **GitHub Repo:** [https://github.com/microsoft/sample-app-aoai-chatGPT](https://github.com/microsoft/sample-app-aoai-chatGPT)
- **Current Functionality:**
  - The chatbot answers questions using Azure OpenAI's ChatGPT model.
  - It is a Python-based web app using Quart (I think).
  - Deployed on **Azure App Service**.

---

## **2. Introduction to Retrieval-Augmented Generation (RAG)**
In a RAG setup, we combine **document retrieval** with a generative language model. Here is how it works:

- **Document Retrieval:** Internal documents are stored as vectors (embeddings) in a **vector database** (e.g., FAISS).
- **Query Processing:** When a user asks a question, the system retrieves the most relevant documents. Starting with preprocessed instead of real-time. Enhances speed and documents don't change often. Maybe later real-time.
- **Answer Generation:** The generative model (ChatGPT) uses the retrieved documents as context to generate a response.

---

## **3. Technical Stack for RAG Implementation**
- **LLMS:**
-   **LangChain:** Less complex and flexible
-   **LLamaIndex:** Specializes in indexing a lot of documents
-   **[PydanticAI](https://ai.pydantic.dev/examples/rag/#example-code):** Integrates really well with Logfire and Pydantic (same creators). New and promises to be the FastAPI of LLMs. Really good documentation.
-   **Haystack:** Focusses more on NLP and a little complex
-   
- **Document Loading:**  **Unstructured, PyPDF, python-docx or something else** to load various document formats (PDF, Word, Excel, PowerPoint). pdfplumber is slower but more thorough.
- **Vector Database:** Use **FAISS** (Fast Approximate Nearest Neighbors) for document indexing and similarity search.
- **Embeddings:** Use **OpenAI embeddings** for transforming documents into vector representations.
- **Flask Integration:** Add new API routes to handle document-based question-answering.

---

Voorbeelden van PydanticAI zijn te vinden in [hun documentatie](https://ai.pydantic.dev/examples/rag/#example-code)
Voorbeeld implementatie met LlamaIndex:
## **4. Implementation Plan**

### **Step 1: Clone the Existing Repository**
1. Open your terminal and run the following commands:
   ```bash
   git clone https://github.com/microsoft/sample-app-aoai-chatGPT.git
   cd sample-app-aoai-chatGPT
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate   # Windows
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the application locally:
   ```bash
   python app.py
   ```
5. Verify that the app runs at [http://localhost:5000](http://localhost:5000) or a similar port.

---

### **Step 2: Extend the Application with RAG Support**

#### **1. Update Dependencies**
Add the following packages to your `requirements.txt` file:
```
langchain
faiss-cpu
unstructured
pypdf
python-docx
pandas
python-pptx
```
Install the new dependencies:
```bash
pip install -r requirements.txt
```

#### **2. Create a Document Loader Module**
Create a new file `document_loader.py` to load and index your internal documents.

```python
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def load_and_index_documents(directory="path/to/your/documents"):
    # Load documents
    loader = UnstructuredFileLoader(directory)
    documents = loader.load()

    # Generate embeddings using OpenAI
    embeddings = OpenAIEmbeddings(deployment="your-openai-deployment")

    # Index documents using FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore
```

#### **3. Integrate RAG into `app.py`**
Modify `app.py` to integrate the RAG functionality:

```python
from document_loader import load_and_index_documents

# Load and index documents at the start of the app
vectorstore = load_and_index_documents("path/to/your/documents")

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json.get("question")

    # Retrieve relevant documents
    docs = vectorstore.similarity_search(question, k=3)
    context = " ".join([doc.page_content for doc in docs])

    # Generate an answer using Azure OpenAI with the retrieved context
    response = openai.Completion.create(
        engine="your-openai-deployment",
        prompt=f"Answer the question based on the following context:\n{context}\n\nQuestion: {question}",
        max_tokens=500
    )
    return jsonify({"answer": response['choices'][0]['text']})
```

---

### **Step 3: Deploy the Updated Application**
1. Commit your changes:
   ```bash
   git add .
   git commit -m "Added RAG support with document indexing"
   git push
   ```
2. The Azure App Service will automatically build and deploy the updated application.

---

## **5. Future Enhancements**
- **SharePoint Integration:** Use the **Microsoft Graph API** to automatically retrieve documents from SharePoint.
- **Real-time Updates:** Add a cron job or event trigger to periodically update the document index.
- **Advanced Logging and Monitoring:** Utilize **Application Insights** for tracking user queries and application performance.
- **User-specific Answer Filtering:** Implement access controls to filter responses based on user roles.

---

## **6. Conclusion**
By implementing the RAG model, the chatbot will be able to answer questions based on internal documents, significantly improving its usefulness for internal users. The proposed setup is scalable and can be enhanced with real-time updates, SharePoint integration, and advanced logging in future iterations.

