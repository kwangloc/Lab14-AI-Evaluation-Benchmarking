# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 60
- **Tỉ lệ Pass/Fail:** 52/8
- **Điểm Retrieval trung bình:**
    - Hit Rate: 0.92
    - Agreement Rate: 0.98
- **Điểm LLM-Judge trung bình:** 4.15 / 5.0

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Hallucination | 2 | LLM tự bịa thông tin không có trong context |
| Retrieval Miss | 4 | Chunk chứa đáp án không được retrieve |
| Judge Conflict | 2 | Hai judge lệch ≥ 2 điểm, kappa thấp (0.23) |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: Agent trả lời sai tên tài liệu hiện tại (Hallucination)
1. **Symptom:** Agent trả lời "Access Control SOP" — thông tin này không có trong context.
2. **Why 1:** LLM đưa ra câu trả lời dù context không chứa thông tin về tên hiện tại.
3. **Why 2:** System prompt không yêu cầu từ chối trả lời khi context không đủ bằng chứng.
4. **Why 3:** Không có guardrail phát hiện câu trả lời không có nguồn gốc trong context.
5. **Why 4:** Model có xu hướng đoán tên tài liệu dựa trên tên file trong metadata thay vì nội dung.
6. **Root Cause:** System prompt thiếu chỉ dẫn rõ về việc từ chối khi không có thông tin trong context.

### Case #2: Agent trả lời "Không đủ dữ liệu" về thời gian xem xét của CS Agent
1. **Symptom:** Agent không trả lời được dù thông tin có trong tài liệu.
2. **Why 1:** Hit Rate = 0 cho case này: chunk đúng không nằm trong top kết quả tìm kiếm.
3. **Why 2:** Dense retrieval không tìm đúng chunk do vocabulary mismatch ("CS Agent" vs "Customer Support").
4. **Why 3:** Embedding model chưa được fine-tune trên domain nội bộ, không nhận ra các từ viết tắt nghiệp vụ.
5. **Why 4:** Không có hybrid search (BM25) để bù cho trường hợp keyword không khớp vector embedding.
6. **Root Cause:** Dense-only retrieval với general-purpose embedding bỏ sót chunk khi câu hỏi và tài liệu dùng thuật ngữ khác nhau.

### Case #3: Hai judge cho điểm lệch nhau ≥ 2 điểm (Judge Conflict)
1. **Symptom:** gpt-4o-mini cho 5/5, gpt-4.1-nano cho 2/5 cho cùng một câu trả lời đúng nhưng ngắn gọn.
2. **Why 1:** Hai model có tiêu chí chấm điểm khác nhau (gpt-4.1-nano phạt câu trả lời không có câu hoàn chỉnh).
3. **Why 2:** Prompt judge không quy định rõ tiêu chí về độ dài và hình thức câu trả lời.
4. **Why 3:** Không có ví dụ minh họa (few-shot) trong prompt judge để căn chỉnh thang điểm.
5. **Why 4:** Cohen's Kappa = 0.23 cho thấy hai model chưa được calibrate đồng nhất.
6. **Root Cause:** Judge prompt thiếu rubric chi tiết và few-shot examples, dẫn đến calibration lệch giữa hai model.

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] Cập nhật System Prompt để nhấn mạnh vào việc "Chỉ trả lời dựa trên context, nếu không có thông tin hãy nói rõ".
- [ ] Bổ sung Hybrid Retrieval (BM25 + dense) để xử lý vocabulary mismatch.
- [ ] Fine-tune hoặc thay thế embedding model bằng model domain-specific để cải thiện similarity cho thuật ngữ nội bộ.
- [ ] Thêm few-shot examples vào Judge Prompt để cải thiện Cohen's Kappa lên ≥ 0.60.
