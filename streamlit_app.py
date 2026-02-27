import re
from typing import Tuple

import streamlit as st

from ml_model import TeamState, available_teams, predict_proba, train_model


TEAM_NAME_ZH = {
    "acmilan": "AC米兰",
    "antwerp": "安特卫普",
    "arsenal": "阿森纳",
    "astonvilla": "阿斯顿维拉",
    "atalanta": "亚特兰大",
    "atleticomadrid": "马德里竞技",
    "barcelona": "巴塞罗那",
    "bayernmuncih": "拜仁慕尼黑",
    "bayernmunich": "拜仁慕尼黑",
    "bayerleverkusen": "勒沃库森",
    "benfica": "本菲卡",
    "bologna": "博洛尼亚",
    "borussiadortmund": "多特蒙德",
    "braga": "布拉加",
    "brest": "布雷斯特",
    "celtic": "凯尔特人",
    "chelsea": "切尔西",
    "clubbrugge": "布鲁日",
    "copenhagen": "哥本哈根",
    "dinamozagreb": "萨格勒布迪纳摩",
    "eintrachtfrankfurt": "法兰克福",
    "feyenoord": "费耶诺德",
    "galatasaray": "加拉塔萨雷",
    "girona": "赫罗纳",
    "inter": "国际米兰",
    "internazionale": "国际米兰",
    "juventus": "尤文图斯",
    "lazio": "拉齐奥",
    "lens": "朗斯",
    "lille": "里尔",
    "liverpool": "利物浦",
    "manchestercity": "曼城",
    "manchesterunited": "曼联",
    "monaco": "摩纳哥",
    "napoli": "那不勒斯",
    "newcastleunited": "纽卡斯尔联",
    "parissaintgermain": "巴黎圣日耳曼",
    "porto": "波尔图",
    "psveindhoven": "埃因霍温",
    "rbleipzig": "RB莱比锡",
    "rbsalzburg": "萨尔茨堡红牛",
    "realmadrid": "皇家马德里",
    "realsociedad": "皇家社会",
    "redstarbelgrade": "贝尔格莱德红星",
    "sevilla": "塞维利亚",
    "shakhtardonetsk": "顿涅茨克矿工",
    "slovanbratislava": "布拉迪斯拉发斯洛万",
    "spartaprague": "布拉格斯巴达",
    "sportingcp": "葡萄牙体育",
    "sturmgraz": "格拉茨风暴",
    "tottenham": "托特纳姆热刺",
    "tottenhamhotspur": "托特纳姆热刺",
    "unionberlin": "柏林联合",
    "vfbstuttgart": "斯图加特",
    "youngboys": "年轻人",
}

TEXTS = {
    "en": {
        "title": "UCL Match Predictor (Web)",
        "subtitle": "Allowed GUI mode: browser interface via Streamlit",
        "language": "Interface Language",
        "home": "Home Team",
        "away": "Away Team",
        "predict": "Predict",
        "same_team": "Teams must be different.",
        "home_win": "Home Win",
        "draw": "Draw",
        "away_win": "Away Win",
        "pred": "Predicted Result",
        "metrics_left": "Home Team Metrics",
        "metrics_right": "Away Team Metrics",
        "model_metrics": "Model Metrics (2025 Test)",
        "metrics_elo": "Elo",
        "metrics_matches": "Matches",
        "metrics_ppm": "Points/Match",
        "metrics_gf": "Goals For/Match",
        "metrics_ga": "Goals Against/Match",
        "metrics_gd": "Goal Diff/Match",
        "no_metrics": "No model metrics available.",
        "row_fmt": "P>={threshold:.1f}: n={matches}, acc={accuracy}",
    },
    "zh": {
        "title": "欧冠比赛预测器（网页版）",
        "subtitle": "允许的GUI模式：使用 Streamlit 浏览器界面",
        "language": "界面语言",
        "home": "主队",
        "away": "客队",
        "predict": "开始预测",
        "same_team": "两支球队不能相同。",
        "home_win": "主队胜",
        "draw": "平局",
        "away_win": "客队胜",
        "pred": "预测结果",
        "metrics_left": "主队指标",
        "metrics_right": "客队指标",
        "model_metrics": "模型指标（2025 测试集）",
        "metrics_elo": "Elo 评分",
        "metrics_matches": "比赛场次",
        "metrics_ppm": "场均积分",
        "metrics_gf": "场均进球",
        "metrics_ga": "场均失球",
        "metrics_gd": "场均净胜球",
        "no_metrics": "暂无模型指标。",
        "row_fmt": "阈值>={threshold:.1f}: 场次={matches}, 准确率={accuracy}",
    },
}


@st.cache_resource
def load_model():
    return train_model()


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _team_display(name: str, lang: str) -> str:
    if lang == "zh":
        return TEAM_NAME_ZH.get(_normalize_key(name), name)
    return name


def _normalize_probs(p_home: float, p_draw: float, p_away: float) -> Tuple[float, float, float]:
    probs = [max(0.0, float(p_home)), max(0.0, float(p_draw)), max(0.0, float(p_away))]
    total = sum(probs)
    if total <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return tuple(p / total for p in probs)


def _percent_texts(p_home: float, p_draw: float, p_away: float) -> Tuple[str, str, str]:
    home_pct = round(p_home * 100, 1)
    draw_pct = round(p_draw * 100, 1)
    away_pct = round(100.0 - home_pct - draw_pct, 1)
    if away_pct < 0:
        away_pct = 0.0
        draw_pct = round(100.0 - home_pct, 1)
    if draw_pct < 0:
        draw_pct = 0.0
        away_pct = round(100.0 - home_pct, 1)
    return (f"{home_pct:.1f}%", f"{draw_pct:.1f}%", f"{away_pct:.1f}%")


def _metrics_text(model, team: str, txt: dict) -> str:
    state = model.states.get(team, TeamState())
    matches = max(1, state.matches)
    ppm = state.points / matches
    gf = state.goals_for / matches
    ga = state.goals_against / matches
    gd = (state.goals_for - state.goals_against) / matches
    return (
        f"{txt['metrics_elo']}: {state.elo:.1f}\n"
        f"{txt['metrics_matches']}: {state.matches}\n"
        f"{txt['metrics_ppm']}: {ppm:.2f}\n"
        f"{txt['metrics_gf']}: {gf:.2f}\n"
        f"{txt['metrics_ga']}: {ga:.2f}\n"
        f"{txt['metrics_gd']}: {gd:.2f}"
    )


def _model_metrics_text(model, txt: dict) -> str:
    if model.evaluation_summary.empty:
        return txt["no_metrics"]
    lines = []
    for _, row in model.evaluation_summary.iterrows():
        threshold = float(row["threshold"])
        matches = int(row["matches_evaluated"])
        accuracy = row["accuracy"]
        acc_txt = f"{float(accuracy):.3f}" if accuracy == accuracy else "N/A"
        lines.append(txt["row_fmt"].format(threshold=threshold, matches=matches, accuracy=acc_txt))
    return "\n".join(lines)


def main() -> None:
    st.set_page_config(page_title="UCL Predictor", page_icon=":soccer:", layout="wide")

    lang_label = st.sidebar.selectbox("Language / 语言", ["English", "中文"], index=0)
    lang = "zh" if lang_label == "中文" else "en"
    txt = TEXTS[lang]

    model = load_model()
    teams = sorted(available_teams(model), key=lambda s: s.lower())
    options = [_team_display(t, lang) for t in teams]
    display_to_raw = { _team_display(t, lang): t for t in teams }

    st.title(txt["title"])
    st.caption(txt["subtitle"])

    col1, col2 = st.columns(2)
    with col1:
        home_display = st.selectbox(txt["home"], options, index=0)
    with col2:
        away_display = st.selectbox(txt["away"], options, index=1 if len(options) > 1 else 0)

    home = display_to_raw[home_display]
    away = display_to_raw[away_display]

    if st.button(txt["predict"], type="primary"):
        if home == away:
            st.warning(txt["same_team"])
            return

        p_home, p_draw, p_away = predict_proba(model, home, away)
        p_home, p_draw, p_away = _normalize_probs(p_home, p_draw, p_away)
        home_pct, draw_pct, away_pct = _percent_texts(p_home, p_draw, p_away)

        home_show = _team_display(home, lang)
        away_show = _team_display(away, lang)
        if p_home >= p_draw and p_home >= p_away:
            winner = f"{home_show} ({txt['home_win']})"
        elif p_away >= p_home and p_away >= p_draw:
            winner = f"{away_show} ({txt['away_win']})"
        else:
            winner = txt["draw"]

        st.subheader(txt["pred"])
        st.write(f"**{home_show} vs {away_show}**")
        st.write(f"**{winner}**")

        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric(txt["home_win"], home_pct)
        pcol2.metric(txt["draw"], draw_pct)
        pcol3.metric(txt["away_win"], away_pct)

        st.progress(float(p_home), text=f"{txt['home_win']}: {home_pct}")
        st.progress(float(p_draw), text=f"{txt['draw']}: {draw_pct}")
        st.progress(float(p_away), text=f"{txt['away_win']}: {away_pct}")

        mcol1, mcol2 = st.columns(2)
        mcol1.text_area(txt["metrics_left"], _metrics_text(model, home, txt), height=150)
        mcol2.text_area(txt["metrics_right"], _metrics_text(model, away, txt), height=150)

        st.text_area(txt["model_metrics"], _model_metrics_text(model, txt), height=120)


if __name__ == "__main__":
    main()
