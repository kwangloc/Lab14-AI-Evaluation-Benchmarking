# Nhật ký Phản chiếu Cá nhân (Individual Reflection)

**Họ và tên:** Trương Quang Lộc  
**Lab:** Day 14 — AI Evaluation Factory  
**Ngày:** 2026-04-21

---

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
$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$  
MRR đo vị trí trung bình (nghịch đảo) của tài liệu đúng trong kết quả tìm kiếm. Ví dụ: nếu chunk đúng ở vị trí 1 → đóng góp 1.0; vị trí 2 → 0.5; vị trí 3 → 0.33. MRR cao hơn Hit Rate vì nó phạt nặng hơn khi kết quả đúng nằm ở vị trí thấp. V2 cải thiện MRR nhờ tăng `top_k_search` từ 1 lên 10.

**Cohen's Kappa (κ)**  
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$  
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
**Nguyên nhân tìm ra:** Reranker chỉ sắp xếp lại kết quả trong pool đã được retrieve. Khi `top_k_search=1`, pool chỉ có 1 chunk — reranker không có tác dụng vì không có gì để sắp xếp lại.  
**Giải pháp:** Tăng `top_k_search=10` → reranker (nếu dùng) hoặc LLM sẽ có pool 10 chunks để chọn 3 cái tốt nhất.

### Vấn đề 2: Cohen's Kappa thấp (0.23) — hai judge không nhất quán
**Quan sát:** `agreement_rate` cao (97.9%) nhưng kappa thấp (0.23) — mâu thuẫn biểu kiến.  
**Nguyên nhân:** `agreement_rate` đo tỉ lệ hai judge cho cùng nhãn pass/fail (threshold 3.0), còn kappa tính trên toàn thang điểm 1-5. `gpt-4.1-nano` có xu hướng cho điểm phân tán hơn, tạo ra nhiều "near-miss" ở ranh giới.  
**Giải pháp đề xuất:** Chuẩn hóa prompt judge để yêu cầu giải thích trước khi cho điểm (Chain-of-Thought scoring) — giúp hai model anchor vào cùng tiêu chí.

### Vấn đề 3: Async pipeline gây race condition khi ghi file kết quả
**Quan sát:** Một số lần chạy bị mất kết quả của 1-2 cases trong file output.  
**Nguyên nhân:** Nhiều coroutine cùng append vào list kết quả không an toàn.  
**Giải pháp:** Dùng `asyncio.Lock` bảo vệ bước append, hoặc collect tất cả kết quả qua `asyncio.gather` rồi mới ghi ra file một lần duy nhất.

---

## 4. Bài học Rút ra

1. **Retrieval là nền tảng** — 75% lỗi của V1 có nguồn gốc từ retrieval miss, không phải từ LLM generation yếu. Đây là điểm cần kiểm tra đầu tiên trước khi tối ưu prompt.
2. **Agreement Rate ≠ Reliability** — Hai judge đồng ý cao nhưng kappa thấp là dấu hiệu calibration kém, không phải hệ thống tốt. Cần luôn báo cáo cả hai chỉ số.
3. **Top-k là hyperparameter quan trọng nhất** trong RAG pipeline — thay đổi từ 1 → 10 tạo ra cải thiện đáng kể hơn bất kỳ thay đổi prompt nào.
4. **Chi phí eval có thể tối ưu** bằng cách phân loại câu hỏi theo độ khó và dùng model phù hợp — không nhất thiết dùng model lớn nhất cho mọi case.
