from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Câu hỏi cần truy xuất
query = "điểm bộ phận đánh giá là gì?"

# Dùng lại embedding + vector store đã build
embedding_model = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
embedding = HuggingFaceEmbeddings(model_name=embedding_model)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)

# Truy vấn top-k kết quả
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents(query)

# Hiển thị các kết quả
print(f"❓ Truy vấn: {query}\n")
for i, doc in enumerate(docs):
    print(f"[Kết quả {i+1}]\n{doc.page_content}\n— Nguồn: {doc.metadata.get('source')}\n")
