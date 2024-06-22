def create_system_message(config):
    wc = int(config["DEFAULT"]["MAX_WORD_COUNT"])
    sc = int(config["DEFAULT"]["MAX_SENTENCE_COUNT"])
    if sc > 1:
        sc = f"{sc-1} - {sc}"

    message = [{
        "role": "system",
        "content": f"""Tôi sẽ cung cấp cho bạn một phần nội dung (ví dụ: bài viết, giấy tờ, tài liệu, v.v.)

        Bạn sẽ tạo ra các bản tóm tắt nội dung ngày càng ngắn gọn, chi tiết hơn.

        Lặp lại 2 bước sau 5 lần.

        Bước 1. Xác định 1-3 Thực thể thông tin (";" được phân cách) từ Bài viết còn thiếu trong bản tóm tắt được tạo trước đó.

        Bước 2. Viết một bản tóm tắt mới, dày đặc hơn có độ dài giống hệt nhau, bao gồm mọi thực thể và chi tiết từ bản tóm tắt trước đó cộng với các Thực thể còn thiếu.

        Một thực thể bị thiếu là:

        Có liên quan: đến câu chuyện chính.
        Cụ thể: mô tả nhưng ngắn gọn (5 từ trở xuống).
        Tiểu thuyết: không có trong bản tóm tắt trước đó.
        Trung thành: có mặt trong phần nội dung.
        Anywhere: nằm ở bất cứ đâu trong Article.

        Hướng dẫn:

        Bản tóm tắt đầu tiên phải dài ({sc} câu, -{wc} từ) nhưng rất không cụ thể, chứa ít thông tin ngoài các mục được đánh dấu là thiếu. Sử dụng ngôn ngữ quá dài dòng và các từ đệm (ví dụ: "bài viết này thảo luận") để tiếp cận các từ -{wc}.
        Đếm từng từ: viết lại bản tóm tắt trước đó để cải thiện dòng chảy và tạo khoảng trống cho các nội dung bổ sung.
        Tạo không gian bằng cách kết hợp, nén và loại bỏ các cụm từ không mang tính thông tin như “bài viết thảo luận”.
        Các bản tóm tắt phải trở nên rất dày đặc và ngắn gọn nhưng vẫn khép kín, ví dụ: dễ hiểu mà không cần có Điều khoản.
        Các thực thể bị thiếu có thể xuất hiện ở bất kỳ đâu trong bản tóm tắt mới.
        Không bao giờ loại bỏ các thực thể khỏi bản tóm tắt trước đó. Nếu không thể tạo được không gian, hãy thêm ít thực thể mới hơn.
        Hãy nhớ rằng, hãy sử dụng cùng một số lượng từ cho mỗi bản tóm tắt.
        Trả lời bằng JSON. JSON phải là danh sách (độ dài 5) từ điển có khóa là "Missing_Entities" và "Denser_Summary"."""}]
    return message