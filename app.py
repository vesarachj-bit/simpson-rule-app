import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simpson's Rule", page_icon="📐", layout="wide")

# ---------------------------------------------------------------- ฟังก์ชันตัวอย่าง
PRESETS = {
    "sin(x)": "sin(x)",
    "x² (x**2)": "x**2",
    "eˣ": "exp(x)",
    "1/(1+x²)": "1/(1+x**2)",
    "√x": "sqrt(x)",
    "กำหนดเอง (custom)": "",
}
SAFE_NAMES = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
    "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "pi": np.pi, "e": np.e,
}


def make_func(expr: str):
    """สร้างฟังก์ชัน f(x) จากข้อความ (จำกัดชื่อที่ใช้ได้เพื่อความปลอดภัย)"""
    code = compile(expr, "<expr>", "eval")

    def f(x):
        y = eval(code, {"__builtins__": {}}, {**SAFE_NAMES, "x": x})
        return np.broadcast_to(np.asarray(y, dtype=float), np.shape(x)).copy()

    return f


# ---------------------------------------------------------------- การคำนวณ
def coefficients(n: int, rule: str) -> np.ndarray:
    c = np.ones(n + 1)
    if rule == "1/3":
        c[1:-1:2] = 4
        c[2:-1:2] = 2
    else:  # 3/8
        c[1:-1] = 3
        c[3:-1:3] = 2
    return c


def simpson(f, a, b, n, rule):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    c = coefficients(n, rule)
    factor = h / 3 if rule == "1/3" else 3 * h / 8
    return factor * np.sum(c * y), x, y, c, h


def trapezoid(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (b - a) / n * (y.sum() - 0.5 * (y[0] + y[-1]))


# ---------------------------------------------------------------- หัวข้อ + คำอธิบาย
st.title("📐 Simpson's Rule Calculator")
st.info(
    "**แอปนี้ทำอะไร:** ประมาณค่าอินทิกรัล ∫ f(x) dx ด้วยกฎของ Simpson "
    "โดยแบ่งช่วง [a, b] เป็น n ช่วงย่อย แล้วใช้พาราโบลา (1/3) หรือกราฟกำลังสาม (3/8) "
    "ประมาณเส้นโค้งในแต่ละกลุ่ม\n\n"
    "**ผู้ใช้ต้องปรับ:** เลือกฟังก์ชัน f(x), ขอบเขต a และ b, จำนวนช่วงย่อย n "
    "และสูตร Simpson — แล้วดูว่าค่า error ลดลงอย่างไรเมื่อเพิ่ม n"
)

# ================================================================ 1) INPUT
st.header("1️⃣ Input")
c1, c2, c3 = st.columns(3)
with c1:
    choice = st.selectbox("ฟังก์ชัน f(x)", list(PRESETS))
    expr = PRESETS[choice]
    if choice.startswith("กำหนดเอง"):
        expr = st.text_input(
            "พิมพ์ f(x)", value="x*sin(x)",
            help="ใช้ x, sin, cos, tan, exp, log, sqrt, abs, pi, e และ ** สำหรับยกกำลัง",
        )
with c2:
    a = st.number_input("ขอบเขตล่าง a", value=0.0, step=0.5, format="%.4f")
    b = st.number_input("ขอบเขตบน b", value=3.14159, step=0.5, format="%.4f")
with c3:
    rule = st.radio("สูตร", ["1/3", "3/8"], horizontal=True,
                    format_func=lambda r: f"Simpson {r}")
    n_in = st.number_input("จำนวนช่วงย่อย n", min_value=1, max_value=100000,
                           value=6, step=1)

# ---------------------------------------------------------------- ตรวจค่าผิดปกติ
if a >= b:
    st.error("❌ ขอบเขตไม่ถูกต้อง: ต้องมี a < b")
    st.stop()

n = int(n_in)
need = 2 if rule == "1/3" else 3
if n % need != 0:
    n_fixed = n + (need - n % need)
    st.warning(
        f"⚠️ Simpson {rule} ต้องใช้ n ที่หารด้วย {need} ลงตัว — "
        f"ปรับ n จาก {n} เป็น {n_fixed} ให้อัตโนมัติ"
    )
    n = n_fixed
if n > 20000:
    st.warning("⚠️ n มีค่ามาก ตารางจะแสดงเฉพาะ 200 แถวแรก")

try:
    f = make_func(expr)
    xs_dense = np.linspace(a, b, 2001)
    with np.errstate(all="ignore"):
        ys_dense = f(xs_dense)
except Exception as ex:
    st.error(f"❌ อ่านสูตร f(x) ไม่ได้: {ex}")
    st.stop()

if not np.all(np.isfinite(ys_dense)):
    st.error("❌ f(x) หาค่าไม่ได้ (inf/nan) ในช่วง [a, b] เช่น หารด้วยศูนย์ หรือ log/√ ของค่าลบ "
             "— ลองเปลี่ยนช่วงหรือฟังก์ชัน")
    st.stop()

# ================================================================ 2) CALCULATION
with np.errstate(all="ignore"):
    S, x, y, coef, h = simpson(f, a, b, n, rule)
    ref, *_ = simpson(f, a, b, 100000, "1/3")  # ค่าอ้างอิงจาก Simpson ละเอียดมาก
    T = trapezoid(f, a, b, n)

abs_err = abs(S - ref)
rel_err = abs_err / abs(ref) * 100 if ref != 0 else float("nan")

# ================================================================ 3) OUTPUT
st.header("2️⃣ Calculation → 3️⃣ Output")

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Simpson {rule}", f"{S:.8f}")
m2.metric("ค่าอ้างอิง", f"{ref:.8f}")
m3.metric("Absolute error", f"{abs_err:.2e}")
m4.metric("Relative error", f"{rel_err:.4f} %" if np.isfinite(rel_err) else "—")

if np.isfinite(rel_err) and rel_err > 1:
    st.warning(f"⚠️ error ค่อนข้างสูง ({rel_err:.2f}%) — ลองเพิ่ม n")

st.caption(f"h = (b − a)/n = {h:.6f} | Trapezoid (n เท่ากัน) = {T:.8f} "
           f"(error {abs(T - ref):.2e})")

tab1, tab2 = st.tabs(["📈 กราฟ", "📋 ตารางการคำนวณ"])

with tab1:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs_dense, ys_dense, "k", lw=2, label="f(x)")
    k = 2 if rule == "1/3" else 3
    colors = ["tab:blue", "tab:orange"]
    for j, i in enumerate(range(0, n, k)):
        xi, yi = x[i:i + k + 1], y[i:i + k + 1]
        xp = np.linspace(xi[0], xi[-1], 60)
        yp = np.polyval(np.polyfit(xi, yi, k), xp)
        ax.fill_between(xp, yp, alpha=0.35, color=colors[j % 2])
        ax.plot(xp, yp, color=colors[j % 2], lw=1.2)
    ax.plot(x, y, "ro", ms=4 if n <= 60 else 2, label="nodes")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Simpson {rule} approximation (n = {n})")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with tab2:
    df = pd.DataFrame({
        "i": np.arange(n + 1),
        "x_i": x,
        "f(x_i)": y,
        "สัมประสิทธิ์": coef.astype(int),
        "สัมประสิทธิ์ × f(x_i)": coef * y,
    })
    st.dataframe(df.head(200), use_container_width=True, hide_index=True)
    st.latex(
        r"I \approx \frac{h}{3}\sum c_i f(x_i)" if rule == "1/3"
        else r"I \approx \frac{3h}{8}\sum c_i f(x_i)"
    )
