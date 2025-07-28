import os
import json
import glob
import shutil
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_qa_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    docs = []
    for pair in qa_pairs:
        text = f"Câu hỏi: {pair['question']}\nTrả lời: {pair['answer']}"
        docs.append(Document(page_content=text, metadata={"source": file_path}))
    return docs

# Đường dẫn vectorstore
VECTORSTORE_DIR = "chroma_db"

# Xoá vectorstore cũ nếu tồn tại
if os.path.exists(VECTORSTORE_DIR):
    print(f"⚠️ Đang xoá thư mục vectorstore cũ: {VECTORSTORE_DIR}")
    shutil.rmtree(VECTORSTORE_DIR)

# Load tất cả file JSON trong thư mục data/
all_docs = []
for path in glob.glob("data/*.json"):
    all_docs.extend(load_qa_json(path))

# Khởi tạo HuggingFace Embeddings (offline)
embedding_model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

# In kích thước embedding
sample_vec = embedding.embed_query("xin chào")
print(f"ℹ️ Embedding dimension: {len(sample_vec)}")

# Tạo Chroma vectorstore
vectorstore = Chroma.from_documents(all_docs, embedding=embedding, persist_directory=VECTORSTORE_DIR)
vectorstore.persist()

print(f"✅ Vector DB đã được tạo và lưu vào {VECTORSTORE_DIR}/")
#đây là comment
