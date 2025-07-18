from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import requests

# Thiết lập retriever từ Vector DB đã build
embedding = HuggingFaceEmbeddings(model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Hàm gửi prompt đến LM Studio (LLM local)
def query_llm_with_context(query: str, docs: list[Document]) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""Bạn là trợ lý học vụ thông minh. Dưới đây là một số thông tin tham khảo:

{context}

Dựa trên các thông tin trên, hãy trả lời cho câu hỏi sau một cách ngắn gọn nhưng đầy đủ, rõ ràng bằng tiếng Việt:

{query}
"""
    
    # print(prompt)
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "local-model",
            "messages": [
                {"role": "system", "content": "Bạn là một trợ lý AI thân thiện, trả lời bằng tiếng Việt thật tự nhiên."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
        },
    )
    try:
        response_json = response.json()
        if "choices" not in response_json:
            print("⚠️ Lỗi từ LLM:", response_json)
            return "❌ Không nhận được phản hồi hợp lệ từ mô hình LLM."
        return response_json["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ Lỗi khi gọi LLM:", e)
        return "❌ Có lỗi xảy ra trong quá trình truy vấn LLM."

# Thử nghiệm
while True:
    query = input("❓ Nhập câu hỏi của bạn (hoặc gõ 'exit' để thoát): ")
    if query.lower() == "exit":
        break
    docs = retriever.get_relevant_documents(query)
    # print(docs)
    answer = query_llm_with_context(query, docs)
    print("\n🤖 Trợ lý trả lời:\n", answer, "\n")
