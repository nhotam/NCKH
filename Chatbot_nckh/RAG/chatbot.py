from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
import requests
import time

# Thiết lập embedding mạnh hơn
embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# Thiết lập Vectorstore từ Chroma
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Tích hợp bộ lọc EmbeddingFilter để làm reranker đơn giản
# compressor = EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.75)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=retriever
# )

# Hàm gửi prompt đến LM Studio (LLM local)
def query_llm_with_context(query: str, docs: list[Document]) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""Bạn là trợ lý học vụ thông minh. Dưới đây là một số thông tin tham khảo:

{context}

Dựa DUY NHẤT trên các thông tin tham khảo dưới đây, hãy trả lời thật chính xác và NGẮN GỌN(tối đa 4 câu) câu hỏi sau. Nếu không đủ thông tin nhưng đầy đủ, rõ ràng bằng tiếng Việt, hãy nói \"Tôi không tìm thấy thông tin trong dữ liệu\".

{query}
"""

    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "vinallama-7b-chat",
            "messages": [
                {"role": "system", "content": "Bạn là trợ lý AI. Dữ liệu người dùng đã được cung cấp, hãy trả lời chính xác bằng cách dựa vào dữ kiện trong dữ liệu truy xuất, trả lời tự nhiên và ngắn gọn. Chỉ cần trả lời đúng trọng tâm câu hỏi người dùng, không lan man."},
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

# Vòng lặp chính để nhận câu hỏi từ người dùng
while True:
    query = input("❓ Nhập câu hỏi của bạn (hoặc gõ 'exit' để thoát): ")
    if query.lower() == "exit":
        break
    retriver_start_time = time.time()
    # chỉnh theo code line 17
    # docs = compression_retriever.invoke(query)
    docs = retriever.get_relevant_documents(query)

    retriver_end_time = time.time()
    retriver_duration_time = retriver_end_time - retriver_start_time 
    # print("📄 Các đoạn truy xuất:")
    # for i, doc in enumerate(docs, 1):
    #     print(f"\n--- Document {i} ---\n{doc.page_content}")
    llm_start_time = time.time()
    answer = query_llm_with_context(query, docs)
    llm_end_time = time.time()
    llm_duration_time = llm_end_time - llm_start_time 
    print("\n🤖 Trợ lý trả lời:\n", answer, "\n")
    print(f"Duration for Retriver: {retriver_duration_time}seconds")
    print(f"Duration for LLM: {llm_duration_time}seconds")
    print(f"Duration for Response: {retriver_duration_time + llm_duration_time } seconds")

