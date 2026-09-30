"""
Kiểm mọi file chép từ repo chính và file mô hình vẫn đúng như lúc chép
======================================================================
Đọc SOURCES.sha256. Mỗi dòng: <sha256>  <đường dẫn trong repo này>  [<-  <nguồn>] .
Một file đạt khi SHA-256 của nó khớp, tính trên byte gốc HOẶC sau khi đổi CRLF → LF
(để kết quả không phụ thuộc core.autocrlf hay hệ điều hành).

Lý do tồn tại: rep_segmentation.py đã cảnh báo rằng hai bản song song, chỉ cần một bên sửa
tham số là số rep đến từ một thuật toán khác mà không có lỗi nào được báo. Code trong
zenodo_code/ và dữ liệu trong data/ là bản sao; sửa ở repo chính rồi chép lại.

    python verify_copies.py

Mã thoát 0 nếu mọi file khớp, 1 nếu có file lệch hoặc thiếu.
"""

import hashlib
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "SOURCES.sha256")


def _shas(path):
    with open(path, "rb") as f:
        b = f.read()
    return {hashlib.sha256(b).hexdigest(), hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()}


def main():
    bad = n = 0
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            want, rel = line.split()[:2]
            n += 1
            path = os.path.join(HERE, rel)
            if not os.path.exists(path):
                print(f"THIẾU  {rel}")
                bad += 1
                continue
            got = _shas(path)
            if want in got:
                print(f"OK    {rel}")
            else:
                # file LFS chưa tải về chỉ là con trỏ vài trăm byte
                hint = " (có thể là con trỏ Git LFS: chạy git lfs pull)" if os.path.getsize(path) < 1024 else ""
                print(f"LỆCH  {rel}{hint}")
                bad += 1
    if bad:
        print(f"\n{bad}/{n} file lệch hoặc thiếu. Không chạy phân tích cho tới khi biết vì sao.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
