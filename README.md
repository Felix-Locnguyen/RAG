# RAG Chatbot (Pet Project)

Hệ thống Hỏi - Đáp tài liệu nội bộ (Retrieval-Augmented Generation - RAG) đơn giản nhưng đầy đủ thành phần, sử dụng LangChain, FAISS Vector Database và OpenAI LLM để truy vấn thông tin từ các tài liệu PDF cá nhân.

---

## Cấu trúc thư mục dự án

```text
RAG/
│
├── Papers/                          # Thư mục chứa tài liệu PDF đầu vào
│   ├── paper_1_smart_poultry_monitoring.pdf
│   └── paper_2_vietnamese_ocr_framework.pdf
│
├── .env.example                     # File mẫu chứa các biến môi trường
├── chatbot.py                       # Script chính chạy luồng RAG Chatbot
├── example.py                       # Script ví dụ/thử nghiệm phụ trợ
├── requirements.txt                 # Danh sách các thư viện Python cần thiết
└── README.md                        # Tài liệu hướng dẫn dự án
```

---

## Các tính năng chính

* **Tải và phân tách tài liệu:** Đọc tự động các file PDF từ thư mục `Papers` và cắt nhỏ văn bản bằng `RecursiveCharacterTextSplitter` với các Markdown separators tối ưu.
* **Vector Database (FAISS):** Chuyển đổi văn bản thành các embedding vector (`text-embedding-3-large`) và lưu trữ cục bộ bằng FAISS, sử dụng chiến lược khoảng cách Cosine.
* **Retrieval thông minh:** Truy xuất theo ngưỡng điểm tương đồng (`similarity_score_threshold`) để lọc các kết quả kém chất lượng.
* **Strict Citation Prompt:** Cấu hình prompt chuyên biệt ép buộc mô hình chỉ trả lời dựa trên context được cung cấp và trích dẫn nguồn rõ ràng.

---

## Hướng dẫn cài đặt và chạy dự án

### 1. Clone hoặc tải mã nguồn
Đảm bảo bạn đang đứng ở thư mục gốc của dự án.

### 2. Tạo và kích hoạt môi trường ảo (Virtual Environment)
```bash
python -m venv venv
# Trên Linux/macOS:
source venv/bin/activate
# Trên Windows:
# venv\Scripts\activate
```

### 3. Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### 4. Cấu hình biến môi trường
Tạo một file `.env` từ file mẫu `.env.example` và điền OpenAI API Key của bạn:
```bash
cp .env.example .env
```
Nội dung file `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Chạy ứng dụng Chatbot RAG
Đặt các file PDF tài liệu của bạn vào thư mục `Papers/`, sau đó chạy câu lệnh:
```bash
python chatbot.py
```
Nhập câu hỏi của bạn trực tiếp trên terminal khi được yêu cầu để nhận câu trả lời từ hệ thống RAG.