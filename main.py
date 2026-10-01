import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ---------------------------------------
# 기본 설정
# ---------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연도별 평균기온을 이용해 기온의 변화 추세를 살펴보고, "
    "회귀 직선을 이용해 특정 연도의 예상 평균기온을 확인합니다."
)


# ---------------------------------------
# 데이터 불러오기
# ---------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# ---------------------------------------
# 연도별 평균기온 계산
# ---------------------------------------
yearly = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 2025년 이후 제외
# 관측일수가 300일 미만인 해 제외
yearly = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# ---------------------------------------
# 전체 기간 회귀분석
# 독립변수 = 1908년부터 지난 연수
# ---------------------------------------
yearly["지난연수"] = yearly["연도"] - 1908

x = yearly["지난연수"].to_numpy()
y = yearly["연평균기온"].to_numpy()

slope, intercept = np.polyfit(x, y, 1)

yearly["회귀선"] = intercept + slope * yearly["지난연수"]

correlation = np.corrcoef(x, y)[0, 1]

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_count = len(yearly)


# ---------------------------------------
# 최근 20년 회귀분석
# ---------------------------------------
recent20 = yearly.tail(20).copy()

x_recent = recent20["지난연수"].to_numpy()
y_recent = recent20["연평균기온"].to_numpy()

recent_slope, recent_intercept = np.polyfit(
    x_recent,
    y_recent,
    1
)


# ---------------------------------------
# 100년에 몇 도 변하는지 계산
# ---------------------------------------
overall_100yr = slope * 100
recent_100yr = recent_slope * 100


# ---------------------------------------
# 전체 기간 기울기 크게 표시
# ---------------------------------------
st.subheader("🌡️ 기온 변화 속도")

st.markdown(
    f"""
    <div style="
        text-align:center;
        padding:25px;
        margin-top:10px;
        margin-bottom:25px;
        border-radius:15px;
        background-color:#FFF7D6;
    ">
        <div style="
            font-size:25px;
            color:#555;
        ">
            전체 기간 기준, 100년에
        </div>

        <div style="
            font-size:65px;
            font-weight:bold;
            margin-top:8px;
        ">
            {overall_100yr:+.2f} ℃
        </div>

        <div style="
            font-size:20px;
            color:#666;
            margin-top:5px;
        ">
            연평균기온이 변하는 회귀선의 기울기
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------
# 전체 기간 vs 최근 20년 비교
# ---------------------------------------
st.subheader("📊 전체 기간과 최근 20년 비교")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label=f"전체 기간 ({start_year}~{end_year})",
        value=f"{overall_100yr:+.2f} ℃",
        border=True
    )
    st.caption("100년에 몇 ℃ 변하는가")

with col2:
    st.metric(
        label=f"최근 20년 ({int(recent20['연도'].min())}~{int(recent20['연도'].max())})",
        value=f"{recent_100yr:+.2f} ℃",
        border=True
    )
    st.caption("100년에 몇 ℃ 변하는가")


# ---------------------------------------
# 회귀선 정보
# ---------------------------------------
st.info(
    f"전체 기간 회귀선은 {start_year}년부터 {end_year}년까지 "
    f"관측일수가 300일 이상인 {data_count}개 연도를 사용했습니다."
)


# ---------------------------------------
# 산점도 + 전체 기간 회귀선
# ---------------------------------------
fig = go.Figure()

# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=8),
        customdata=yearly[["관측일수"]],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f} ℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["회귀선"],
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(width=3),
        hovertemplate=(
            "%{x}년 예상기온: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title="서울 연평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified",
    height=550,
    xaxis=dict(
        tickmode="linear",
        dtick=10
    )
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------
# 회귀식
# ---------------------------------------
st.subheader("📈 전체 기간 회귀 직선")

st.write(
    "독립변수는 1908년부터 지난 연수로 계산했습니다."
)

st.latex(
    f"예상기온 = {intercept:.4f} + "
    f"{slope:.4f} \\times (연도 - 1908)"
)

st.write(
    f"→ 100년 기준으로 약 **{overall_100yr:+.2f} ℃** "
    f"변하는 기울기입니다."
)


# ---------------------------------------
# 최근 20년 회귀식
# ---------------------------------------
st.subheader("📈 최근 20년 회귀 직선")

st.latex(
    f"예상기온 = {recent_intercept:.4f} + "
    f"{recent_slope:.4f} \\times (연도 - 1908)"
)

st.write(
    f"→ 최근 20년의 기울기는 100년 기준 "
    f"**{recent_100yr:+.2f} ℃**입니다."
)


# ---------------------------------------
# 연도 슬라이더
# ---------------------------------------
st.subheader("🔎 연도별 예상기온")

selected_year = st.slider(
    "연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

selected_x = selected_year - 1908

# 전체 기간 회귀선으로 예상
predicted_temp = intercept + slope * selected_x


# ---------------------------------------
# 예상기온 크게 표시
# ---------------------------------------
st.markdown(
    f"""
    <div style="
        text-align:center;
        padding:20px;
        margin-top:10px;
        margin-bottom:20px;
        border-radius:15px;
        background-color:#FFF7D6;
    ">
        <div style="
            font-size:25px;
            color:#555;
        ">
            {selected_year}년 예상 연평균기온
        </div>

        <div style="
            font-size:65px;
            font-weight:bold;
            margin-top:8px;
        ">
            {predicted_temp:.2f} ℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------
# 실제 관측값 확인
# ---------------------------------------
actual = yearly[yearly["연도"] == selected_year]

if not actual.empty:
    actual_temp = actual.iloc[0]["연평균기온"]

    st.write(
        f"📌 {selected_year}년 실제 연평균기온: "
        f"**{actual_temp:.2f} ℃**"
    )
else:
    st.write(
        f"📌 {selected_year}년은 관측자료가 없거나 조건에 맞지 않아 "
        f"**회귀선으로 계산한 예상값**을 표시합니다."
    )


# ---------------------------------------
# 데이터 조건 안내
# ---------------------------------------
st.caption(
    "※ 2025년 이후 자료와 관측일수가 300일 미만인 연도는 "
    "회귀분석에서 제외했습니다."
)
