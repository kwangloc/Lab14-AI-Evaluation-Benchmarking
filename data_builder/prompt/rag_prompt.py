"""
rag_prompt.py — Prompt templates for grounded RAG answer generation.
"""


def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Build a grounded answer prompt that forces the model to cite sources
    and abstain when context is insufficient.

    Args:
        query: The user's question.
        context_block: Formatted context snippets (output of build_context_block).

    Returns:
        Full prompt string ready to send to the LLM.
    """
    return f"""Bạn là trợ lý hỗ trợ nội bộ chuyên nghiệp. Hãy trả lời câu hỏi dựa HOÀN TOÀN vào các đoạn tài liệu được cung cấp bên dưới.

QUY TẮC BẮT BUỘC:
1. Chỉ sử dụng thông tin có trong các đoạn [1], [2], [3]... để trả lời. Không bịa thêm thông tin.
2. Sau mỗi thông tin quan trọng, ghi nguồn trích dẫn trong ngoặc vuông, ví dụ: [1], [2].
3. Nếu các đoạn tài liệu KHÔNG chứa đủ thông tin để trả lời, hãy trả lời chính xác: "Không đủ dữ liệu để trả lời câu hỏi này."
4. Trả lời bằng tiếng Việt, ngắn gọn và súc tích.

--- TÀI LIỆU THAM KHẢO ---
{context_block}
--- HẾT TÀI LIỆU ---

CÂU HỎI: {query}

TRẢ LỜI:"""
