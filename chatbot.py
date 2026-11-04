# ==========================================
# 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT
# ==========================================
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv 

# Tải các biến môi trường từ file .env (ví dụ: OPENAI_API_KEY)
load_dotenv()


# ==========================================
# 2. TẢI TÀI LIỆU (DOCUMENT LOADING)
# ==========================================
# Khởi tạo DirectoryLoader để quét và đọc tất cả các file PDF trong thư mục ./Papers
loader = DirectoryLoader(
    path= "./Papers",
    glob="**/*.pdf",                   # Đọc tất cả dữ liệu có dạng pdf trong đường dẫn
    loader_cls=UnstructuredFileLoader, # Dùng Unstructured để xử lý các định dạng phức tạp như PDF
    show_progress=True,                # Hiển thị thanh tiến trình load dữ liệu trên terminal
    use_multithreading=True,           # Sử dụng đa luồng để load dữ liệu nhanh hơn
)
docs = loader.load()


# ==========================================
# 3. CẤU HÌNH BỘ CHIA TÁCH VĂN BẢN (TEXT SPLITTING)
# ==========================================
# Định nghĩa danh sách các bộ ký tự phân cách ưu tiên dựa trên cấu trúc Markdown
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",          # Tiêu đề Markdown
    "```\n",                # Khối code
    "\n\\*\\*\\*+\n",       # Các đường kẻ ngang (*, -, _)
    "\n---+\n",
    "\n___+\n",
    "\n\n",                 # Đoạn văn
    "\n",                   # Dòng
    " ",                    # Khoảng trắng
    "",                     # Ký tự
]

# Khởi tạo Text Splitter để cắt nhỏ tài liệu thành các chunk phù hợp cho việc embedding
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1200,          # Chia độ dài mỗi chunk tối đa là 1200 kí tự
    chunk_overlap = 200,        # Độ dài chồng lấp giữa các chunk liên tiếp là 200 kí tự để giữ ngữ cảnh
    add_start_index = True,     # Đánh dấu vị trí bắt đầu (thứ tự) cho từng chunk
    strip_whitespace = True,    # Loại bỏ khoảng trắng thừa ở đầu và cuối các chunk
    separators = MARKDOWN_SEPARATORS, # Sử dụng các ký tự phân cách markdown đã định nghĩa
)

# Thực hiện cắt nhỏ toàn bộ tài liệu đã tải
splits = text_splitter.split_documents(docs)


# ==========================================
# 4. TẠO EMBEDDING VÀ LƯU TRỮ VÀO VECTOR DATABASE
# ==========================================
# Khởi tạo mô hình Embedding của OpenAI để chuyển đổi văn bản thành vector
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# Chuyển các chunks thành vector và lưu trữ vào FAISS Vector Store, sử dụng khoảng cách Cosine
vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings,
    distance_strategy=DistanceStrategy.COSINE
)


# ==========================================
# 5. CẤU HÌNH TRUY XUẤT (RETRIEVER)
# ==========================================
# Chuyển vectorstore thành retriever để tìm kiếm các đoạn văn bản liên quan nhất khi có câu hỏi
retriver = vectorstore.as_retriever(
    search_type = "similarity_score_threshold", # Tìm kiếm dựa trên ngưỡng điểm tương đồng
    search_kwargs = {"k": 3, "score_threshold": 0.2}, # k = số lượng kết quả tối đa trả về, score_threshold = ngưỡng lọc điểm tối thiểu
) 


# ==========================================
# 6. THIẾT LẬP PROMPT VÀ MÔ HÌNH NGÔN NGỮ (LLM)
# ==========================================
# Định nghĩa template cấu trúc câu lệnh nhắc (Prompt) giới hạn hành vi của AI
template = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer.\n"
    "2) If the answer is not clearly contained in the context, say: "
    "\"I don't know based on the provided documents.\"\n"
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) If applicable, cite sources as (source:page) using the metadata.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

# Tạo ChatPromptTemplate từ chuỗi template trên
prompt = ChatPromptTemplate.from_messages(
    template
)

# Khởi tạo mô hình ngôn ngữ OpenAI LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo-mini",
    temperature=0.0, # Đặt bằng 0.0 để giảm thiểu tính ngẫu nhiên, giúp câu trả lời tập trung và chính xác theo context nhất
    max_tokens=1000  # Giới hạn số lượng token tối đa trong câu trả lời
)


# ==========================================
# 7. XÂY DỰNG LUỒNG RAG PIPELINE (LCEL)
# ==========================================
# Kết nối các thành phần lại thành một chuỗi xử lý (Pipeline) hoàn chỉnh:
# - Truy xuất context từ retriever và nhận câu hỏi gốc qua RunnablePassthrough
# - Đưa vào prompt -> Gọi LLM -> Phân tích kết quả đầu ra thành dạng chuỗi văn bản thuần túy (StrOutputParser)
rag_chain = (
    {"context": retriver, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ==========================================
# 8. THỰC THI TRUY VẤN (EXECUTION)
# ==========================================
# Nhận câu hỏi từ người dùng thông qua giao diện dòng lệnh (CLI)
question = input("Enter your question: ")

# Chạy pipeline RAG với câu hỏi đã nhập
answer = rag_chain.invoke(question)

# In kết quả câu trả lời ra màn hình terminal
print(answer)