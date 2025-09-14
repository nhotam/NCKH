from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
import requests
import time
import re

# Cấu hình Embedding và Vectorstore
embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)

#  Thêm bộ lọc để loại bỏ tài liệu không liên quan
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
compressor = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.6)
retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

# Kiểm tra nếu LLM trả lời sai ngôn ngữ (ví dụ: tiếng Trung)
def contains_foreign_language(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))

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
                    "dựa vào dữ kiện trong dữ liệu truy xuất. Trả lời đúng trọng tâm, ngắn gọn, không lan man, đặc biệt là thân thiện với người dùng. "
                    "Bắt buộc phải trả lời bằng Tiếng Việt. Tuyệt đối KHÔNG được tự bịa ra thông tin hoặc suy đoán."
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
        content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        if contains_foreign_language(content) or not content.strip():
            return "Thật xin lỗi, tôi không hiểu yêu cầu của bạn."
        return content
    except Exception as e:
        print(" Lỗi khi gọi LLM:", e)
        return " Có lỗi xảy ra trong quá trình truy vấn LLM."

# Hàm xử lý truy vấn người dùng
def handle_query(query: str):
    start_retrieval = time.time()
    docs = retriever.invoke(query)
    end_retrieval = time.time()

    #  CHÈN: In nội dung truy xuất được
    print("\n Các đoạn văn bản được truy xuất:")
    for i, doc in enumerate(docs):
        print(f"[{i+1}] {doc.page_content}\n")

    # Kiểm tra nếu không có dữ liệu đủ
    total_content = " ".join(doc.page_content for doc in docs).strip()
    if not docs or len(total_content) < 30:
        print("\n Trợ lý trả lời:\nThật xin lỗi, tôi không hiểu yêu cầu của bạn.\n")
        print(f" Truy xuất dữ liệu: {end_retrieval - start_retrieval:.2f}s")
        print(f" Tổng thời gian: {end_retrieval - start_retrieval:.2f}s\n")
        return answer

    start_llm = time.time()
    answer = query_llm_with_context(query, docs)
    end_llm = time.time()

    # Hiển thị kết quả
    print("\n Trợ lý trả lời:\n", answer, "\n")
    print(f" Truy xuất dữ liệu: {end_retrieval - start_retrieval:.2f}s")
    print(f" Gọi LLM: {end_llm - start_llm:.2f}s")
    print(f" Tổng thời gian: {end_llm - start_retrieval:.2f}s\n")
    return answer

# Vòng lặp chính
def main():
    print(" Trợ lý học vụ sẵn sàng. Gõ 'exit' để thoát.")
    while True:
        query = input(" Nhập câu hỏi: ")
        if query.lower().strip() == "exit":
            break
        handle_query(query)

if __name__ == "__main__":
    main()
