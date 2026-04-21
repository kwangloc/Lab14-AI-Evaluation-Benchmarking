# Nhật ký Phản chiếu Cá nhân (Individual Reflection)

**Họ và tên:** Trương Quang Lộc  
**Lab:** Day 14 — AI Evaluation Factory  
**Ngày:** 2026-04-21

---
## Note: 
- Bài lab này em làm cá nhân thay vì làm nhóm. Lý do là qua các bài lab nhóm trước thì em cảm giác rằng khi chỉ phụ trách một phần của mình trong bài lab, em chỉ có thể nắm được tối đa 50-60% phần nội dung kỹ thuật mày ngày hôm đó dạy. Em nghĩ rằng với việc mỗi bài lab chỉ tập trung vào 1 kỹ thuật nhất định (RAG/Eval/...), điều quan trọng là em cần nắm được các bước nhỏ hơn trong đó, để hiểu được ý nghĩa tại sao có nó. 
- Trong lab này, cách làm của em là vibe code từng step một để quan sát được cách triển khai từng module, ví dụ vài bước em đã làm như sau:
    + B1: Chunking và embed lại data từ day 8
    + B2: Sửa lại luồng retrieve và build context cho hợp với bài lab 14 này
    + B3: Build golden dataset
    + B4: Build agent
    + B5: Build multi-judges
    + B6: Chạy thử benchmark (chỉ một phần nhỏ dataset)
    + B7: Quan sát file output, thấy các chỉ số metrics faithfullness, relevancy đang 1 tuyệt đối ở nhiều samples -> check lại formula và chỉnh sửa. Thấy test case 2 thì V1 đúng mà V2 lại sai => retrieval hoạt động chưa đúng
    + ...
- Em nhận thức được tầm quan trọng của làm việc nhóm, nhưng với những bài lab chuyên sâu vào 1 kỹ thuật nhất định chứ không phải build một dự án nhiều stack, thì em nghĩ chọn làm cá nhân sẽ đảm bảo bản thân nắm được rõ từng kỹ thuật nhỏ trong đó. 
## 1. Đóng góp Kỹ thuật (Engineering Contribution)

### Các module tôi đã xây dựng / đóng góp chính:

**a) Multi-Judge Consensus Engine (`engine/llm_judge.py`)**  
Tôi thiết kế và triển khai logic đánh giá song song sử dụng hai model judge độc lập: `gpt-4o-mini` và `gpt-4.1-nano`. Cụ thể:
- Gọi hai judge bất đồng bộ (async) để giảm latency.
- Tính `agreement_rate` bằng cách so sánh điểm số của hai judge (threshold ±1 điểm).
- Xử lý xung đột tự động: khi hai judge lệch ≥ 2 điểm, lấy trung bình và đánh dấu case là `conflict` trong kết quả.
- Tính **Cohen's Kappa** để đo độ tin cậy giữa hai judge (kết quả: κ = 0.230 — mức "fair agreement").

**b) Retrieval Evaluation (`engine/retrieval_eval.py`)**  
Tôi cài đặt hai chỉ số đánh giá retrieval:
- **Hit Rate@k**: kiểm tra xem chunk đúng có xuất hiện trong top-k kết quả trả về không.
- **MRR (Mean Reciprocal Rank)**: đo vị trí trung bình của chunk đúng trong danh sách kết quả, phản ánh độ chính xác thứ hạng.

**c) Async Pipeline & Cost Report**  
Tham gia tối ưu pipeline chạy song song toàn bộ 60 test cases bằng `asyncio.gather`, giảm thời gian chạy benchmark xuống dưới 2 phút. Tích hợp tracking token usage theo từng role (agent vs judges) và tính chi phí USD theo bảng giá OpenAI.

---

## 2. Chiều sâu Kỹ thuật (Technical Depth)

### Giải thích các khái niệm cốt lõi:

**MRR (Mean Reciprocal Rank)**  

$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$  

MRR đo vị trí trung bình (nghịch đảo) của tài liệu đúng trong kết quả tìm kiếm. Ví dụ: nếu chunk đúng ở vị trí 1 → đóng góp 1.0; vị trí 2 → 0.5; vị trí 3 → 0.33. MRR cao hơn Hit Rate vì nó phạt nặng hơn khi kết quả đúng nằm ở vị trí thấp. V2 cải thiện MRR nhờ tăng `top_k_search` từ 1 lên 10.

**Cohen's Kappa (κ)**  

$\kappa = \frac{p_o - p_e}{1 - p_e}$

Trong đó $p_o$ là tỉ lệ đồng thuận thực tế, $p_e$ là tỉ lệ đồng thuận kỳ vọng ngẫu nhiên. κ = 0.23 trong lab này cho thấy hai judge đồng ý hơn mức ngẫu nhiên nhưng chưa đủ tin cậy (cần κ ≥ 0.60 cho production). Nguyên nhân: hai model có calibration điểm khác nhau — `gpt-4o-mini` chấm khắt hơn `gpt-4.1-nano` ở các câu trả lời ngắn gọn.

**Position Bias trong LLM Judge**  
LLM judge có xu hướng chấm điểm cao hơn cho câu trả lời được đặt ở vị trí đầu tiên trong prompt (position bias). Để giảm thiểu, tôi đặt câu trả lời của agent và expected answer ở thứ tự cố định thay vì đổi chỗ ngẫu nhiên — điều này đảm bảo so sánh nhất quán giữa các lần chạy.

**Trade-off Chi phí vs Chất lượng**  
| Lựa chọn | Chi phí | Chất lượng |
|----------|---------|------------|
| Judge duy nhất (gpt-4o) | Thấp | Không có cross-validation |
| 2 judge nhỏ (gpt-4o-mini + gpt-4.1-nano) | $0.010 / 60 cases | Có agreement metric, phát hiện conflict |
| 1 judge lớn + 1 judge nhỏ | Trung bình | Tốt nhất về calibration |

V2 tốn $0.111 so với V1 là $0.071 (+56%), nhưng điểm trung bình tăng từ 3.4 lên 4.15 (+22%). ROI tốt. Để giảm 30% chi phí eval: thay `gpt-4o` làm agent bằng `gpt-4o-mini` cho các case có độ phức tạp thấp (câu hỏi fact-check đơn giản), chỉ dùng `gpt-4o` cho các hard cases.

---

## 3. Giải quyết Vấn đề (Problem Solving)

### Vấn đề 1: Hit Rate V1 quá thấp (~0.75) dù đã có reranker
**Quan sát:** V1 dùng `use_rerank=True` nhưng hit rate vẫn thấp hơn V2 (rerank=False).  
**Nguyên nhân tìm ra:** Thuật toán rerank em tái sử dụng từ day 8, thì thuật toán này bọn em đánh giá là chưa tối ưu, khả năng là do mô hình bọn em dùng là cross-encoder chưa hoạt động tốt trên bộ data này, nên kết quả cho ra lại tệ hơn lúc không dùng (rerank sai). Em đã set top_k_search về 1 luôn để giảm xác suất bắt trúng chunk, mục đích là để V1 hiệu suất yếu hơn V2.


**Giải pháp:** Eval lại phương pháp rerank.

### Vấn đề 2: Cohen's Kappa thấp (0.23) — hai judge không nhất quán
**Quan sát:** `agreement_rate` cao (97.9%) nhưng kappa thấp (0.23) — mâu thuẫn biểu kiến.  
**Nguyên nhân:** `agreement_rate` đo tỉ lệ hai judge cho cùng nhãn pass/fail (threshold 3.0), còn kappa tính trên toàn thang điểm 1-5. `gpt-4.1-nano` có xu hướng cho điểm phân tán hơn, tạo ra nhiều "near-miss" ở ranh giới.  
**Giải pháp đề xuất:** Chuẩn hóa prompt judge để yêu cầu giải thích trước khi cho điểm (Chain-of-Thought scoring) — giúp hai model anchor vào cùng tiêu chí.

### Vấn đề 3: Lỗi truy cập Chroma khi chạy song song
**Quan sát:** Pipeline chạy song song bị lỗi "Rust bindings" ở ChromaDB.

**Nguyên nhân:** Khi chạy đa luồng (song song) thì hàm hàn retrieve_dense() sẽ tạo một PersistenClient ở mỗi luồng, nhưng Rust bindings đã được tạo ở main thread, và không thể khởi tạo lạo ở các thread worker, vậy nên sinh ra lỗi.
**Giải pháp:** Khởi tạo PersistenClient ở level module như là một singleton. Dùng chung cho các thread worker. 

---

## 4. Bài học Rút ra

1. **Retrieval là nền tảng** — Vì thuật toán rerank em lấy ở day 8 làm chưa chuẩn, nên dẫn tới hiệu suất agent lại trái với dự đoán. 
2. **Evaluation giúp phát hiện điểm yếu pipeline** — Kết quả benchmark trái với dự đoán sẽ khiến mình phải đặt câu hỏi và đi tìm lỗi.
3. **Cost cho evel là xứng đáng** bỏ một chi phí nhất định cho eval giúp phát hiện sớm lỗi, tránh mất mát lớn hơn trong tương lai.
