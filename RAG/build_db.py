import os
import json
import glob
import shutil
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Cấu hình
DATA_DIR = "data"
VECTORSTORE_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# ----------------------------- #
# Hàm: Load 1 file QA JSON
# ----------------------------- #
def load_qa_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
    except Exception as e:
        print(f" Lỗi khi đọc {file_path}: {e}")
        return []

    docs = []
    for pair in qa_pairs:
        if 'question' in pair and 'answer' in pair:
            text = f"Câu hỏi: {pair['question']}\nTrả lời: {pair['answer']}"
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(file_path)}))
        else:
            print(f" Bỏ qua cặp QA không hợp lệ trong {file_path}")
    return docs

# ----------------------------- #
# Hàm: Load toàn bộ file trong thư mục
# ----------------------------- #
def load_all_qa_documents(data_dir):
    all_docs = []
    file_paths = glob.glob(os.path.join(data_dir, "*.json"))
    for path in file_paths:
        docs = load_qa_json(path)
        all_docs.extend(docs)
    print(f" Tổng số tài liệu đã load: {len(all_docs)}")
    return all_docs

# ----------------------------- #
# Hàm: Xây dựng vectorstore
# ----------------------------- #
def build_vectorstore(docs, model_name, persist_dir):
    try:
        embedding = HuggingFaceEmbeddings(model_name=model_name)
    except ImportError:
        print("\n Thiếu thư viện 'sentence-transformers'.")
        print(" Vui lòng cài đặt bằng lệnh: pip install sentence-transformers")
        exit(1)

    sample_vector = embedding.embed_query("xin chào")
    print(f"ℹ Embedding dimension: {len(sample_vector)}")

    vectorstore = Chroma.from_documents(docs, embedding=embedding, persist_directory=persist_dir)
    
    print(f" Vectorstore đã được lưu tại: {persist_dir}/")

# ----------------------------- #
# Main
# ----------------------------- #
if __name__ == "__main__":
    # Xoá vectorstore cũ nếu có
    if os.path.exists(VECTORSTORE_DIR):
        print(f" Đang xoá thư mục vectorstore cũ: {VECTORSTORE_DIR}")
        shutil.rmtree(VECTORSTORE_DIR)

    print(" Đang load dữ liệu và xây dựng vectorstore...")
    documents = load_all_qa_documents(DATA_DIR)

    if documents:
        build_vectorstore(documents, EMBEDDING_MODEL_NAME, VECTORSTORE_DIR)
    else:
        print(" Không có tài liệu nào được load. Dừng chương trình.")
