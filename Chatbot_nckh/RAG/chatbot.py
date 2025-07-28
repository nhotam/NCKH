from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import requests
import time

# Cấu hình Embedding và Vectorstore
embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Hàm gọi LLM từ LM Studio
def query_llm_with_context(query: str, docs: list[Document]) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""Bạn là trợ lý học vụ thông minh. Dưới đây là một số thông tin tham khảo:

{context}

Dựa DUY NHẤT trên các thông tin tham khảo dưới đây, hãy trả lời thật chính xác và NGẮN GỌN (tối đa 4 câu) câu hỏi sau. 
Nếu không đủ thông tin, hãy trả lời: "Tôi không tìm thấy thông tin trong dữ liệu".

{query}
"""

    payload = {
        "model": "qwen2.5-3b-instruct",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý AI. Dữ liệu người dùng đã được cung cấp, hãy trả lời chính xác "
                    "dựa vào dữ kiện trong dữ liệu truy xuất. Trả lời đúng trọng tâm, ngắn gọn, không lan man."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        response_json = response.json()
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "❌ Không có nội dung phản hồi.")
    except Exception as e:
        print("❌ Lỗi khi gọi LLM:", e)
        return "❌ Có lỗi xảy ra trong quá trình truy vấn LLM."

# Hàm xử lý truy vấn người dùng
def handle_query(query: str):
    start_retrieval = time.time()
    docs = retriever.invoke(query)
    end_retrieval = time.time()

    start_llm = time.time()
    answer = query_llm_with_context(query, docs)
    end_llm = time.time()

    # Hiển thị kết quả
    print("\n🤖 Trợ lý trả lời:\n", answer, "\n")
    print(f"⏱ Truy xuất dữ liệu: {end_retrieval - start_retrieval:.2f}s")
    print(f"⏱ Gọi LLM: {end_llm - start_llm:.2f}s")
    print(f"⏱ Tổng thời gian: {end_llm - start_retrieval:.2f}s\n")

# Vòng lặp chính
def main():
    print("📚 Trợ lý học vụ sẵn sàng. Gõ 'exit' để thoát.")
    while True:
        query = input("❓ Nhập câu hỏi: ")
        if query.lower().strip() == "exit":
            break
        handle_query(query)

if __name__ == "__main__":
    main()
