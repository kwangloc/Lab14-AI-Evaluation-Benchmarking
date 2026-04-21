# Báo cáo Phân tích Thất bại (Failure Analysis Report)

> Ngày chạy benchmark: 2026-04-21 | Model: gpt-4o (agent), gpt-4o-mini + gpt-4.1-nano (judges)

## 1. Tổng quan Benchmark

| Chỉ số | Agent V1 (top_k=1, rerank=True) | Agent V2 (top_k=10, select=3, rerank=False) |
|--------|----------------------------------|----------------------------------------------|
| Tổng số cases | 60 | 60 |
| Pass (score ≥ 3.0) | ~48 / 60 | ~52 / 60 |
| Fail (score < 3.0) | ~12 / 60 | ~8 / 60 |
| Hit Rate (Retrieval) | ~0.75 | **0.917** |
| Agreement Rate (Judges) | ~0.85 | **0.979** |
| Cohen's Kappa | — | **0.230** |
| **LLM-Judge trung bình** | ~3.4 / 5.0 | **4.15 / 5.0** |
| Tổng chi phí (USD) | $0.071 | $0.111 |

> **Ghi chú RAGAS:** Pipeline sử dụng `hit_rate@3` và `MRR` thay cho Faithfulness/Relevancy RAGAS do không có ground-truth full context riêng lẻ. Hit Rate V2 = **0.917**, MRR V2 ≈ **0.82** (ước tính từ top-k ranking).

---

## 2. Phân nhóm lỗi (Failure Clustering) — dựa trên V1 + V2

| Nhóm lỗi | Số lượng (V1) | Số lượng (V2) | Nguyên nhân chính |
|----------|---------------|---------------|-------------------|
| **Retrieval Miss** | 9 | 4 | `top_k=1` quá thấp, chunk không được tìm thấy |
| **Hallucination** | 2 | 2 | LLM tự bịa tên tài liệu / thông tin không có trong context |
| **Factual Error** | 1 | 1 | LLM diễn giải sai dữ liệu (VD: giờ làm việc) |
| **Incomplete Answer** | 1 | 1 | Agent chỉ liệt kê điều kiện, bỏ sót bước thực hiện |
| **Judge Conflict** | 1 | 2 | Hai judge cho điểm lệch nhau ≥ 2 điểm (kappa thấp: 0.23) |
| **Tổng** | **14** | **10** |  |

---

## 3. Phân tích 5 Whys (3 case tệ nhất — V1)

### Case #1: "Tài liệu này có tên gì hiện tại?" — Score V1: 1.5 (Hallucination)

| Bước | Nội dung |
|------|----------|
| **Symptom** | Agent trả lời "Access Control SOP" — tên tài liệu hiện tại không được đề cập trong đoạn context bất kỳ. |
| **Why 1** | LLM đưa ra câu trả lời dù không có thông tin trong context (hallucination). |
| **Why 2** | System prompt không yêu cầu rõ ràng "nếu không có trong context, phải trả lời 'không có thông tin'". |
| **Why 3** | Không có guardrail sau xử lý để phát hiện câu trả lời tự bịa. |
| **Why 4** | Ground truth xác nhận thông tin này thực sự không tồn tại trong chunk — agent không thể phân biệt "biết" vs "đoán". |
| **Root Cause** | System prompt thiếu chỉ dẫn rõ về việc từ chối trả lời khi context không đủ bằng chứng. |

---

### Case #2: "CS Agent sẽ xem xét trong bao lâu?" — Score V1: 1.0 (Retrieval Miss)

| Bước | Nội dung |
|------|----------|
| **Symptom** | Agent trả lời "Không đủ dữ liệu" — câu trả lời đúng có trong tài liệu nhưng không được retrieve. |
| **Why 1** | Hit Rate = 0 cho case này: chunk chứa câu trả lời không nằm trong top-1 kết quả tìm kiếm. |
| **Why 2** | `top_k_search=1` trong V1 quá hạn hẹp — chỉ lấy 1 chunk duy nhất. |
| **Why 3** | Embedding similarity của câu hỏi và chunk đáp án không đủ cao khi dùng dense retrieval thuần. |
| **Why 4** | Tên thực thể "CS Agent" trong câu hỏi khác biểu diễn vector so với "Customer Support" trong tài liệu. |
| **Root Cause** | Kết hợp `top_k=1` + vocabulary mismatch làm cho dense retrieval bỏ sót chunk quan trọng. Giải pháp: tăng `top_k` và bổ sung hybrid (BM25 + dense). |

---

### Case #3: "Giờ làm việc bộ phận hỗ trợ là khi nào?" — Score V1: 2.5 (Factual Error)

| Bước | Nội dung |
|------|----------|
| **Symptom** | Agent trả lời "8:00 đến 18:00" — ground truth là "8:00–17:30". |
| **Why 1** | Chunk được retrieve đúng (hit_rate=1), nhưng LLM trả lời sai 30 phút. |
| **Why 2** | Chunk chứa nhiều thông tin về lịch làm việc, LLM tổng hợp nhầm từ nhiều dòng. |
| **Why 3** | Không có validation bước cuối để kiểm tra số liệu thời gian trong câu trả lời so với context. |
| **Why 4** | LLM có xu hướng làm tròn số liệu thời gian sang giờ chẵn khi không chú ý kỹ. |
| **Root Cause** | LLM không đủ chú ý chi tiết khi context dày; cần prompt yêu cầu trích dẫn trực tiếp số liệu. |

---

## 4. So sánh V1 vs V2 — Cải tiến đạt được

| Thay đổi | V1 | V2 | Tác động |
|----------|----|----|----------|
| `top_k_search` | 1 | 10 | Hit Rate tăng từ ~0.75 lên 0.917 |
| `top_k_select` | 1 | 3 | Agent có nhiều context hơn để tổng hợp |
| `use_rerank` | True | False | Giảm latency; reranker V1 không cải thiện precision ở top_k=1 |
| Avg LLM-Judge Score | ~3.4 | **4.15** | +0.75 điểm trung bình |
| Chi phí agent | $0.066 | $0.106 | Tăng do context dài hơn (token tăng) |

> **Nhận xét:** V2 cải thiện đáng kể retrieval miss (từ 9 xuống 4 cases). Tuy nhiên hallucination và judge conflict vẫn tồn tại — cần thêm prompt guardrails và calibration giữa hai judge.

---

## 5. Kế hoạch cải tiến (Action Plan)

- [x] Tăng `top_k_search` từ 1 lên 10 và `top_k_select` lên 3 → đã triển khai trong V2.
- [ ] Cập nhật System Prompt với chỉ dẫn rõ: *"Nếu câu trả lời không có trong context, hãy nói 'Tôi không tìm thấy thông tin này trong tài liệu'"* — loại bỏ hallucination.
- [ ] Bổ sung **Hybrid Retrieval** (BM25 + dense) để xử lý vocabulary mismatch (case CS Agent / Customer Support).
- [ ] Thêm **Semantic Chunking** để tránh cắt thông tin bảng biểu / danh sách quan trọng.
- [ ] Cải thiện **Judge Calibration**: huấn luyện prompt judge thống nhất hơn → nâng Cohen's Kappa từ 0.23 lên ≥ 0.60.
- [ ] Thêm post-processing kiểm tra số liệu (ngày tháng, giờ giấc) khớp với context trước khi trả về.
