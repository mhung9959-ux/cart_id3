
from collections import Counter
from math import log2


#  độ đo
def entropy(rows, target):
    n = len(rows)
    cnt = Counter(r[target] for r in rows)
    return -sum(c / n * log2(c / n) for c in cnt.values())


def gini(rows, target):
    n = len(rows)
    cnt = Counter(r[target] for r in rows)
    return 1 - sum((c / n) ** 2 for c in cnt.values())


#  sinh các cách chia
def make_splits(rows, feat, numeric, mode):
    """Trả về list các (mô_tả, {nhãn_nhánh: [rows]})."""
    out = []
    vals = sorted(set(r[feat] for r in rows))
    if feat in numeric:                                   # thuộc tính liên tục
        for a, b in zip(vals, vals[1:]):
            t = (a + b) / 2
            left = [r for r in rows if r[feat] <= t]
            right = [r for r in rows if r[feat] > t]
            out.append((f"{feat} <= {t:g}", {"Yes": left, "No": right}))
    elif mode == "id3":                                   # rời rạc, chia nhiều nhánh
        out.append((feat, {v: [r for r in rows if r[feat] == v] for v in vals}))
    else:                                                 # CART: nhị phân "= v" / "còn lại"
        if len(vals) > 1:
            for v in vals:
                yes = [r for r in rows if r[feat] == v]
                no = [r for r in rows if r[feat] != v]
                out.append((f"{feat} = {v}", {"Yes": yes, "No": no}))
    return out


def score(parts, target, mode):
    """ID3: Gain (càng lớn càng tốt). CART: Gini sau chia (càng nhỏ càng tốt)."""
    n = sum(len(p) for p in parts)
    f = entropy if mode == "id3" else gini
    return sum(len(p) / n * f(p, target) for p in parts)


# xây cây
def build(rows, feats, target, numeric, mode, depth=0, max_depth=None):
    cnt = Counter(r[target] for r in rows)
    majority = cnt.most_common(1)[0][0]
    leaf = {"leaf": True, "label": majority, "n": len(rows), "dist": dict(cnt)}
    if len(cnt) == 1 or not feats or (max_depth is not None and depth >= max_depth):
        return leaf

    base = entropy(rows, target) if mode == "id3" else gini(rows, target)
    best = None
    for f in feats:
        for desc, branches in make_splits(rows, f, numeric, mode):
            parts = [p for p in branches.values() if p]
            if len(parts) < 2:
                continue
            s = score(parts, target, mode)
            if best is None or s < best[0] - 1e-12:       # s nhỏ nhất = Gain/Gini tốt nhất
                best = (s, f, desc, branches)
    if best is None or base - best[0] <= 1e-12:           # không cải thiện -> lá
        return leaf

    s, f, desc, branches = best

    rest = feats if (f in numeric or mode == "cart") else [x for x in feats if x != f]
    node = {"leaf": False, "text": desc, "n": len(rows), "dist": dict(cnt),
            "gain": base - s, "children": {}}
    for label, sub in branches.items():
        if sub:
            node["children"][label] = build(sub, rest, target, numeric, mode, depth + 1, max_depth)
    return node


def predict(tree, row, numeric=()):
    while not tree["leaf"]:
        txt = tree["text"]
        if " <= " in txt:                                  # chia số
            f, t = txt.split(" <= ")
            key = "Yes" if row[f] <= float(t) else "No"
        elif " = " in txt:                                 # CART nhị phân
            f, v = txt.split(" = ")
            key = "Yes" if str(row[f]) == v else "No"
        else:                                              # ID3 nhiều nhánh
            key = row[txt]
        tree = tree["children"][key]
    return tree["label"]


def print_tree(node, indent="", label=""):
    if label:
        print(f"{indent}[{label}]")
        indent += "   "
    if node["leaf"]:
        print(f"{indent}-> {node['label']}  {node['dist']}")
    else:
        print(f"{indent}{node['text']}?  (n={node['n']}, độ đo cải thiện={node['gain']:.3f})")
        for lb, ch in node["children"].items():
            print_tree(ch, indent + "   ", lb)


#  dữ liệu
BUYS = [  # age, income, student, credit_rating, buys_computer
    ("<=30", "high", "no", "fair", "no"), ("<=30", "high", "no", "excellent", "no"),
    ("31...40", "high", "no", "fair", "yes"), (">40", "medium", "no", "fair", "yes"),
    (">40", "low", "yes", "fair", "yes"), (">40", "low", "yes", "excellent", "no"),
    ("31...40", "low", "yes", "excellent", "yes"), ("<=30", "medium", "no", "fair", "no"),
    ("<=30", "low", "yes", "fair", "yes"), (">40", "medium", "yes", "fair", "yes"),
    ("<=30", "medium", "yes", "excellent", "yes"), ("31...40", "medium", "no", "excellent", "yes"),
    ("31...40", "high", "yes", "fair", "yes"), (">40", "medium", "no", "excellent", "no"),
]
BUYS_COLS = ["age", "income", "student", "credit_rating", "buys_computer"]

CREDIT = [  # Tuoi, HonNhan, BDS, ThuNhap, RuiRo
    (25, "Độc thân", "Ở cùng bố mẹ", 7000000, 0), (40, "Đã kết hôn", "Nhà sở hữu", 18000000, 0),
    (35, "Từng ly hôn", "Nhà thuê", 12000000, 1), (27, "Đã kết hôn", "Ở cùng bố mẹ", 9000000, 1),
    (31, "Độc thân", "Nhà thuê", 6000000, 1), (36, "Đã kết hôn", "Nhà sở hữu", 8000000, 1),
    (48, "Độc thân", "Nhà thuê", 7000000, 0), (26, "Đã kết hôn", "Nhà thuê", 8000000, 1),
    (33, "Từng ly hôn", "Ở cùng bố mẹ", 5000000, 1), (29, "Độc thân", "Nhà thuê", 10000000, 0),
    (38, "Đã kết hôn", "Nhà sở hữu", 15000000, 0), (44, "Độc thân", "Nhà sở hữu", 14000000, 1),
    (42, "Đã kết hôn", "Nhà sở hữu", 10000000, 0), (28, "Độc thân", "Nhà thuê", 7000000, 1),
    (30, "Đã kết hôn", "Ở cùng bố mẹ", 6000000, 1),
]
CREDIT_COLS = ["Tuổi", "Hôn nhân", "Sở hữu BĐS", "Thu nhập", "Rủi ro"]


def to_rows(data, cols):
    return [dict(zip(cols, r)) for r in data]


if __name__ == "__main__":
    #  Bài 1: buys_computer 
    rows = to_rows(BUYS, BUYS_COLS)
    feats = BUYS_COLS[:-1]
    for mode, name in (("id3", "ID3"), ("cart", "CART")):
        t = build(rows, feats, "buys_computer", numeric=set(), mode=mode)
        print(f"\n===== {name} - buys_computer =====")
        print_tree(t)
        acc = sum(predict(t, r) == r["buys_computer"] for r in rows) / len(rows)
        print(f"Độ chính xác trên tập train: {acc:.0%}")

    #  Bài 2: rủi ro tín dụng 
    rows = to_rows(CREDIT, CREDIT_COLS)
    feats = CREDIT_COLS[:-1]
    for mode, name in (("id3", "ID3"), ("cart", "CART")):
        t = build(rows, feats, "Rủi ro", numeric={"Tuổi", "Thu nhập"}, mode=mode)
        print(f"\n===== {name} - rủi ro tín dụng =====")
        print_tree(t)
