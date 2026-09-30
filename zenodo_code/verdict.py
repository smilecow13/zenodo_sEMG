"""
verdict(): diễn giải kết quả kiểm định một cách bắt buộc phải đúng
===================================================================
Phiên WP1b có ba bug CÙNG MỘT LOẠI, và cả ba đều nằm ở đoạn IN PHÁN QUYẾT chứ
không ở đoạn tính toán — phần tính đúng cả ba lần, phần diễn giải sai cả ba lần:

  1. Đọc `cmp.get("sig")` trong khi `compare_paired` trả về khoá `"significant"`
     → luôn ra False → in "không có ý nghĩa" ngay dưới dòng báo p=0.0081 [SIG].
  2. Điều kiện một chiều `real_gain - full_gain > 0.2` → khi hiệu số ĐỔI DẤU
     (lợi tăng 1.13pp) thì nhánh else in "gần như không đổi", nuốt mất đúng cái
     nghịch lý quan trọng nhất.
  3. Chỉ kiểm `p < 0.05` mà không kiểm dấu → control lợi −0.07pp (bằng chứng
     KHÔNG rò rỉ) bị in thành "cần điều tra thêm".

Đây là lớp lỗi nguy hiểm nhất: không crash, không sai số, chỉ làm kết luận
NGƯỢC. Không sửa được bằng cách cẩn thận hơn, phải sửa bằng hạ tầng.

Module này buộc ba điều tại chỗ gọi:
  - phải KHAI BÁO CHIỀU KỲ VỌNG (`expected_sign` không có giá trị mặc định)
  - phải đọc trạng thái qua `.state` / `.is_confirmed`, KHÔNG dùng được
    `if v:` — `__bool__` cố tình raise
  - đọc p-value và delta từ dict của `compare_paired` qua `from_compare()`,
    không tự gõ tên khoá

Bốn trạng thái:
  CONFIRMED    có ý nghĩa, ĐÚNG chiều kỳ vọng
  REVERSED     có ý nghĩa nhưng NGƯỢC chiều kỳ vọng  (bug #3 thuộc loại này)
  INCONCLUSIVE không đủ bằng chứng (p ≥ alpha) — KHÔNG đồng nghĩa "bằng nhau"
  UNDEFINED    p hoặc delta không xác định

Usage:
    from verdict import verdict, from_compare, EXPECT_POSITIVE

    v = from_compare(cmp, expected_sign=EXPECT_POSITIVE,
                     claim="N1 tốt hơn F0")
    print(v)                      # dòng mô tả đầy đủ
    if v.is_confirmed: ...        # hợp lệ
    if v: ...                     # TypeError — cố ý
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

CONFIRMED = "CONFIRMED"
REVERSED = "REVERSED"
INCONCLUSIVE = "INCONCLUSIVE"
UNDEFINED = "UNDEFINED"

EXPECT_POSITIVE = 1  # kỳ vọng delta > 0
EXPECT_NEGATIVE = -1  # kỳ vọng delta < 0


@dataclass(frozen=True)
class Verdict:
    state: str
    p_value: float
    delta: float
    expected_sign: int
    alpha: float
    claim: Optional[str] = None

    # ── Guard: chặn hẳn việc dùng như boolean ──────────────────────────────
    def __bool__(self):
        raise TypeError(
            "Verdict không dùng được như boolean — đây chính là cách ba bug của "
            "phiên WP1b lọt qua. Dùng .is_confirmed / .is_reversed / "
            ".is_inconclusive, hoặc so sánh .state với CONFIRMED/REVERSED/"
            "INCONCLUSIVE/UNDEFINED."
        )

    @property
    def is_confirmed(self):
        return self.state == CONFIRMED

    @property
    def is_reversed(self):
        return self.state == REVERSED

    @property
    def is_inconclusive(self):
        return self.state == INCONCLUSIVE

    @property
    def is_undefined(self):
        return self.state == UNDEFINED

    @property
    def phrasing(self):
        """Cách phát biểu được phép dùng, theo quy ước báo cáo của dự án."""
        return {
            CONFIRMED: "viết thành khẳng định",
            REVERSED: "CẢNH BÁO: có ý nghĩa nhưng NGƯỢC chiều kỳ vọng",
            INCONCLUSIVE: "'xu hướng, chưa đủ bằng chứng' — KHÔNG được nói là bằng nhau",
            UNDEFINED: "không phát biểu được",
        }[self.state]

    def __str__(self):
        want = "dương" if self.expected_sign > 0 else "âm"
        head = f"[{self.state}]"
        claim = f" {self.claim}:" if self.claim else ""
        return (
            f"  {head}{claim} delta={self.delta:+.2f} (kỳ vọng {want}), "
            f"p={self.p_value:.4f}, alpha={self.alpha:g} -> {self.phrasing}"
        )


def verdict(p_value, delta, expected_sign, alpha=0.05, claim=None):
    """
    Diễn giải một kết quả kiểm định thành một trong bốn trạng thái.

    Parameters
    ----------
    p_value : float
    delta : float
        Hiệu ứng đo được. Chiều của nó phải cùng quy ước với `expected_sign`.
    expected_sign : int
        BẮT BUỘC. EXPECT_POSITIVE nếu giả thuyết dự đoán delta > 0,
        EXPECT_NEGATIVE nếu dự đoán delta < 0. Không có giá trị mặc định —
        caller phải nói rõ mình kỳ vọng gì, vì chính việc bỏ qua chiều là
        nguồn của bug #3.
    alpha : float
        Ngưỡng. Truyền ngưỡng ĐÃ hiệu chỉnh vào đây khi báo cáo dưới
        Holm–Bonferroni.
    claim : str
        Mô tả ngắn để dòng in ra tự giải thích được.

    Returns
    -------
    Verdict
    """
    if expected_sign not in (EXPECT_POSITIVE, EXPECT_NEGATIVE):
        raise ValueError(
            f"expected_sign phải là EXPECT_POSITIVE (+1) hoặc EXPECT_NEGATIVE (-1), "
            f"nhận được {expected_sign!r}"
        )

    if p_value is None or delta is None or not np.isfinite(p_value) or not np.isfinite(delta):
        state = UNDEFINED
    elif p_value >= alpha:
        state = INCONCLUSIVE
    elif np.sign(delta) == np.sign(expected_sign):
        state = CONFIRMED
    else:
        # Có ý nghĩa nhưng ngược chiều — trạng thái mà ba bug của phiên WP1b
        # đều xử lý sai, hoặc bỏ qua hoàn toàn.
        state = REVERSED

    return Verdict(
        state=state,
        p_value=float(p_value) if p_value is not None and np.isfinite(p_value) else float("nan"),
        delta=float(delta) if delta is not None and np.isfinite(delta) else float("nan"),
        expected_sign=int(expected_sign),
        alpha=float(alpha),
        claim=claim,
    )


def from_compare(cmp_result, expected_sign, alpha=0.05, claim=None):
    """
    Tạo Verdict từ dict của `eval_protocols.compare_paired`.

    Dùng hàm này thay vì tự đọc khoá: khoá đúng là "significant" và
    "delta_mean", còn "sig" KHÔNG tồn tại — gõ sai tên khoá chính là bug #1.
    Hàm sẽ raise nếu dict không có đúng các khoá mong đợi, để lỗi lộ ra ngay
    thay vì im lặng trả về False.
    """
    if cmp_result is None:
        return verdict(float("nan"), float("nan"), expected_sign, alpha, claim)

    missing = [k for k in ("p_value", "delta_mean") if k not in cmp_result]
    if missing:
        raise KeyError(
            f"cmp_result thiếu khoá {missing}. Có: {sorted(cmp_result)}. "
            "Lưu ý compare_paired dùng 'significant'/'delta_mean', không phải 'sig'/'delta'."
        )

    return verdict(cmp_result["p_value"], cmp_result["delta_mean"],
                   expected_sign, alpha=alpha, claim=claim)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print("Tái hiện ba bug của phiên WP1b để xác nhận verdict() bắt được:\n")

    # Bug #3: có ý nghĩa nhưng ngược chiều (control lợi -0.07pp, p=0.0171)
    v = verdict(0.0171, -0.07, EXPECT_POSITIVE, claim="control có rò rỉ biên")
    print(v)
    assert v.is_reversed, "phải là REVERSED, không phải 'cần điều tra thêm'"

    # Trường hợp đúng: N1 hơn F0
    v = verdict(0.0046, +1.74, EXPECT_POSITIVE, claim="N1 tốt hơn F0 (bỏ rep baseline)")
    print(v)
    assert v.is_confirmed

    # Sát ngưỡng, chết khi hiệu chỉnh
    v = verdict(0.0479, +0.53, EXPECT_POSITIVE, alpha=0.05, claim="top12 mới hơn 12 cũ (thô)")
    print(v)
    assert v.is_confirmed
    v2 = verdict(0.0479, +0.53, EXPECT_POSITIVE, alpha=0.05 / 5,
                 claim="top12 mới hơn 12 cũ (Holm trong họ)")
    print(v2)
    assert v2.is_inconclusive, "sau hiệu chỉnh phải thành INCONCLUSIVE"

    # Bug #1: guard chặn dùng như boolean
    try:
        if v:
            pass
        raise AssertionError("__bool__ phải raise")
    except TypeError as err:
        print(f"\n  Guard boolean hoạt động: {str(err)[:62]}...")

    # Bug #1: gõ sai tên khoá phải lộ ra ngay
    try:
        from_compare({"sig": True, "delta": 1.0}, EXPECT_POSITIVE)
        raise AssertionError("from_compare phải raise KeyError")
    except KeyError as err:
        print(f"  Guard tên khoá hoạt động: {str(err)[:62]}...")

    # expected_sign bắt buộc
    try:
        verdict(0.01, 1.0, 0)
        raise AssertionError("expected_sign=0 phải raise")
    except ValueError:
        print("  Guard expected_sign hoạt động")

    print("\nTất cả guard PASS")
