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

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 날짜와 평균기온이 모두 있는 자료만 사용
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# ---------------------------------------
# 연도별 자료 만들기
# ---------------------------------------
yearly = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 수업 기준 기간:
# 2025년까지
# 관측일수 300일 이상
yearly = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# ---------------------------------------
# 회귀분석
# 독립변수 = 1908년부터 지난 연수
# ---------------------------------------
yearly["지난연수"] = yearly["연도"] - 1908

x = yearly["지난연수"].to_numpy()
y = yearly["연평균기온"].to_numpy()

# 1차 선형회귀
slope, intercept = np.polyfit(x, y, 1)

# 회귀선 예측값
yearly["회귀선"] = intercept + slope * yearly["지난연수"]

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# 회귀선에 사용된 기간
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_count = len(yearly)


# ---------------------------------------
# 핵심 정보 표시
# ---------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("직선을 만든 해의 개수", f"{data_count}개")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

with col4:
    st.metric("상관계수", f"{correlation:.3f}")


st.info(
    f"회귀선은 {start_year}년부터 {end_year}년까지의 "
    f"관측일이 300일 이상인 연도 {data_count}개를 사용해 만들었습니다."
)


# ---------------------------------------
# 산점도 + 회귀선
# ---------------------------------------
fig = go.Figure()

# 실제 연평균기온 산점도
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

# 회귀선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["회귀선"],
        mode="lines",
        name="회귀 직선",
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
st.subheader("📈 회귀 직선")

st.write(
    f"1908년부터 지난 연수를 x라고 하면, "
    f"회귀 직선은 다음과 같습니다."
)

st.latex(
    f"예상기온 = {intercept:.4f} + {slope:.4f} \\times (연도 - 1908)"
)

st.write(
    f"즉, 1년이 지날 때마다 회귀선 기준으로 "
    f"연평균기온이 약 {slope:.4f} ℃씩 변하는 직선입니다."
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

# 선택한 연도의 회귀선 예상기온
selected_x = selected_year - 1908
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
# 선택한 연도의 실제 관측값이 있는 경우
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
        f"📌 {selected_year}년은 회귀 직선을 이용한 "
        f"**예상값**만 표시됩니다."
    )


# ---------------------------------------
# 데이터 조건 안내
# ---------------------------------------
st.caption(
    "※ 2025년 이후 자료와 해당 연도의 관측일수가 300일 미만인 연도는 "
    "회귀 직선을 계산할 때 제외했습니다."
)
