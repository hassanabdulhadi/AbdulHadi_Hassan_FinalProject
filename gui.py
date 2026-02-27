import base64
import os
import re
from pathlib import Path
from typing import Tuple

import streamlit as st
import streamlit.components.v1 as components

from ml_model import TeamState, available_teams, predict_proba, train_model


ROOT = Path(__file__).resolve().parent
THEME_MUSIC = ROOT / "fifa18_theme.mp3"
ASSETS_DIR = ROOT / "assets"
TEAM_LOGO_DIR = ASSETS_DIR / "team_logos"
SPLASH_IMAGE_CANDIDATES = (
    "Starting pic.jpeg",
    "starting pic.jpeg",
    "start_screen.png",
    "start_screen.jpg",
    "start_screen.jpeg",
    "fifa18_start.png",
    "fifa18_start.jpg",
    "fifa18_cover.png",
    "fifa18_cover.jpg",
    "ronaldo.png",
    "ronaldo.jpg",
    "cover.png",
    "cover.jpg",
    "image.png",
    "image.jpg",
    "image.jpeg",
)

TEAM_NAME_EN = {
    "acmilan": "AC Milan",
    "antwerp": "Antwerp",
    "arsenal": "Arsenal",
    "astonvilla": "Aston Villa",
    "atalanta": "Atalanta",
    "atleticomadrid": "Atletico Madrid",
    "barcelona": "Barcelona",
    "bayernmuncih": "Bayern Munich",
    "bayernmunich": "Bayern Munich",
    "bayerleverkusen": "Bayer Leverkusen",
    "benfica": "Benfica",
    "bologna": "Bologna",
    "borussiadortmund": "Borussia Dortmund",
    "braga": "Braga",
    "brest": "Brest",
    "celtic": "Celtic",
    "chelsea": "Chelsea",
    "clubbrugge": "Club Brugge",
    "copenhagen": "Copenhagen",
    "dinamozagreb": "Dinamo Zagreb",
    "eintrachtfrankfurt": "Eintracht Frankfurt",
    "feyenoord": "Feyenoord",
    "galatasaray": "Galatasaray",
    "girona": "Girona",
    "inter": "Inter",
    "internazionale": "Inter",
    "juventus": "Juventus",
    "lazio": "Lazio",
    "lens": "Lens",
    "lille": "Lille",
    "liverpool": "Liverpool",
    "manchestercity": "Manchester City",
    "manchesterunited": "Manchester United",
    "monaco": "Monaco",
    "napoli": "Napoli",
    "newcastleunited": "Newcastle United",
    "parissaintgermain": "Paris Saint-Germain",
    "porto": "Porto",
    "psveindhoven": "PSV Eindhoven",
    "rbleipzig": "RB Leipzig",
    "rbsalzburg": "RB Salzburg",
    "realmadrid": "Real Madrid",
    "realsociedad": "Real Sociedad",
    "redstarbelgrade": "Red Star Belgrade",
    "sevilla": "Sevilla",
    "shakhtardonetsk": "Shakhtar Donetsk",
    "slovanbratislava": "Slovan Bratislava",
    "spartaprague": "Sparta Prague",
    "sportingcp": "Sporting CP",
    "sturmgraz": "Sturm Graz",
    "tottenham": "Tottenham",
    "tottenhamhotspur": "Tottenham",
    "unionberlin": "Union Berlin",
    "vfbstuttgart": "VfB Stuttgart",
    "youngboys": "Young Boys",
}

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
        "start_title": "UCL MATCH PREDICTOR",
        "start_btn": "CLICK HERE TO START",
        "lang_title": "Choose Interface Language",
        "lang_subtitle": "Select one option to continue",
        "en_btn": "ENGLISH INTERFACE",
        "zh_btn": "CHINESE INTERFACE",
        "app_title": "UCL MATCH SELECT",
        "app_subtitle": "Select both sides to continue",
        "home": "HOME",
        "away": "AWAY",
        "predict": "CONTINUE",
        "pred_title": "Prediction and Metrics",
        "home_win": "Home Win",
        "draw": "Draw",
        "away_win": "Away Win",
        "model_metrics": "Model Metrics (2025 test)",
        "pick_diff": "Teams must be different.",
        "metrics_elo": "Elo",
        "metrics_matches": "Matches",
        "metrics_ppm": "Points/Match",
        "metrics_gf": "Goals For/Match",
        "metrics_ga": "Goals Against/Match",
        "metrics_gd": "Goal Diff/Match",
        "no_metrics": "No model metrics available.",
        "matchup": "{home} vs {away}",
        "winner_home": "{team} Win",
        "winner_draw": "Draw",
        "winner_away": "{team} Win",
        "model_row": "P>={threshold:.1f}: n={matches}, acc={accuracy}",
    },
    "zh": {
        "start_title": "欧冠比赛预测器",
        "start_btn": "点击这里开始",
        "lang_title": "选择界面语言",
        "lang_subtitle": "请选择一个选项继续",
        "en_btn": "英文界面",
        "zh_btn": "中文界面",
        "app_title": "欧冠比赛选择",
        "app_subtitle": "请选择双方球队后继续",
        "home": "主队",
        "away": "客队",
        "predict": "继续",
        "pred_title": "预测结果与指标",
        "home_win": "主队胜",
        "draw": "平局",
        "away_win": "客队胜",
        "model_metrics": "模型指标（2025 测试集）",
        "pick_diff": "两支球队不能相同。",
        "metrics_elo": "Elo 评分",
        "metrics_matches": "比赛场次",
        "metrics_ppm": "场均积分",
        "metrics_gf": "场均进球",
        "metrics_ga": "场均失球",
        "metrics_gd": "场均净胜球",
        "no_metrics": "暂无模型指标。",
        "matchup": "{home} 对阵 {away}",
        "winner_home": "{team} 胜",
        "winner_draw": "平局",
        "winner_away": "{team} 胜",
        "model_row": "阈值>={threshold:.1f}: 场次={matches}, 准确率={accuracy}",
    },
}


@st.cache_resource
def load_model():
    return train_model()


def _setting(name: str, default: str = "") -> str:
    # Read settings from Streamlit secrets first, then environment variables.
    value = ""
    try:
        value = str(st.secrets.get(name, "")).strip()
    except Exception:
        value = ""
    if value:
        return value
    return os.getenv(name, default).strip()


MUSIC_URL = _setting("UCL_MUSIC_URL")
SPLASH_IMAGE_URL = _setting("UCL_SPLASH_IMAGE_URL")
LOGO_BASE_URL = _setting("UCL_LOGO_BASE_URL")


def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _start_image() -> Path | None:
    for n in SPLASH_IMAGE_CANDIDATES:
        p1 = ROOT / n
        if p1.exists():
            return p1
        p2 = ASSETS_DIR / n
        if p2.exists():
            return p2
    return None


def _start_image_source() -> tuple[str, str | Path | bytes] | None:
    # Local image first; remote fallback second.
    local = _start_image()
    if local:
        return ("path", local)
    if SPLASH_IMAGE_URL:
        return ("url", SPLASH_IMAGE_URL)
    return None


def _team_name(name: str, lang: str) -> str:
    key = _norm_key(name)
    if lang == "zh":
        return TEAM_NAME_ZH.get(key, name)
    if key in TEAM_NAME_EN:
        return TEAM_NAME_EN[key]
    return " ".join(part.capitalize() for part in name.split())


def _dedup_teams(raw_teams: list[str], lang: str) -> tuple[list[str], dict[str, str]]:
    by_key: dict[str, str] = {}
    for t in raw_teams:
        k = _norm_key(t)
        if k not in by_key:
            by_key[k] = t

    labels: list[str] = []
    label_to_raw: dict[str, str] = {}
    for raw in sorted(by_key.values(), key=lambda s: _team_name(s, "en").lower()):
        label = _team_name(raw, lang)
        if label in label_to_raw:
            continue
        labels.append(label)
        label_to_raw[label] = raw
    return labels, label_to_raw


def _normalize_probs(p_home: float, p_draw: float, p_away: float) -> Tuple[float, float, float]:
    vals = [max(0.0, float(p_home)), max(0.0, float(p_draw)), max(0.0, float(p_away))]
    total = sum(vals)
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return tuple(v / total for v in vals)


def _percent_strings(p_home: float, p_draw: float, p_away: float) -> Tuple[str, str, str]:
    home_pct = round(p_home * 100, 1)
    draw_pct = round(p_draw * 100, 1)
    away_pct = round(100.0 - home_pct - draw_pct, 1)
    return (f"{home_pct:.1f}%", f"{draw_pct:.1f}%", f"{away_pct:.1f}%")


def _team_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def _logo_path(team: str) -> Path | None:
    p = TEAM_LOGO_DIR / f"{_team_slug(team)}.png"
    return p if p.exists() else None


def _logo_source(team: str) -> str | Path | None:
    local = _logo_path(team)
    if local:
        return local
    if LOGO_BASE_URL:
        base = LOGO_BASE_URL.rstrip("/")
        return f"{base}/{_team_slug(team)}.png"
    return None


def _music_src() -> str | None:
    # Return embeddable music source: data URI from local file, or remote URL fallback.
    if THEME_MUSIC.exists():
        data = base64.b64encode(THEME_MUSIC.read_bytes()).decode("utf-8")
        return f"data:audio/mpeg;base64,{data}"
    if MUSIC_URL:
        return MUSIC_URL
    return None


def _render_audio() -> None:
    src = _music_src()
    if not src:
        return
    components.html(
        f"""
        <script>
        (function() {{
          const doc = window.parent.document;
          const id = "ucl_bg_audio";
          const src = "{src}";
          let a = doc.getElementById(id);
          if (!a) {{
            a = doc.createElement("audio");
            a.id = id;
            a.src = src;
            a.loop = true;
            a.autoplay = true;
            a.preload = "auto";
            a.style.display = "none";
            a.addEventListener("loadedmetadata", () => {{
              if (a.currentTime < 4) a.currentTime = 4;
            }});
            doc.body.appendChild(a);
          }}
          if (Math.floor(a.currentTime) < 4) a.currentTime = 4;
          const tryPlay = () => a.play().catch(() => {{}});
          tryPlay();
          if (!window._uclAudioRetry) {{
            window._uclAudioRetry = setInterval(tryPlay, 1200);
          }}
        }})();
        </script>
        """,
        height=0,
    )


def _apply_theme() -> None:
    bg_css = ""
    splash_source = _start_image_source()
    if splash_source:
        source_type, source = splash_source
        if source_type == "path":
            bg = base64.b64encode(Path(source).read_bytes()).decode("utf-8")
            bg_css = (
                "background-image: linear-gradient(rgba(5, 12, 30, 0.78), rgba(5, 12, 30, 0.78)), "
                f"url('data:image/jpeg;base64,{bg}'); background-size: cover; background-attachment: fixed;"
            )
        else:
            bg_css = (
                "background-image: linear-gradient(rgba(5, 12, 30, 0.78), rgba(5, 12, 30, 0.78)), "
                f"url('{source}'); background-size: cover; background-attachment: fixed;"
            )
    st.markdown(
        f"""
        <style>
        .stApp {{
          {bg_css}
          background-color: #050c1e;
          color: #f3f8ff;
        }}
        .block-container {{padding-top: 1.2rem; max-width: 1100px;}}
        h1, h2, h3, p, label {{color: #f3f8ff !important;}}
        .stButton > button {{
          background: linear-gradient(90deg, #ffd166 0%, #fca311 100%) !important;
          color: #14213d !important;
          border: 0 !important;
          border-radius: 12px !important;
          font-weight: 900 !important;
          font-size: 20px !important;
          padding: 0.85rem 1rem !important;
          min-height: 56px !important;
          box-shadow: 0 8px 20px rgba(252, 163, 17, 0.35);
        }}
        .stButton > button:hover {{
          filter: brightness(1.06);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metrics_text(model, team: str, tx: dict) -> str:
    state = model.states.get(team, TeamState())
    m = max(1, state.matches)
    ppm = state.points / m
    gf = state.goals_for / m
    ga = state.goals_against / m
    gd = (state.goals_for - state.goals_against) / m
    return (
        f"{tx['metrics_elo']}: {state.elo:.1f}\n"
        f"{tx['metrics_matches']}: {state.matches}\n"
        f"{tx['metrics_ppm']}: {ppm:.2f}\n"
        f"{tx['metrics_gf']}: {gf:.2f}\n"
        f"{tx['metrics_ga']}: {ga:.2f}\n"
        f"{tx['metrics_gd']}: {gd:.2f}"
    )


def _model_metrics_text(model, tx: dict) -> str:
    if model.evaluation_summary.empty:
        return tx["no_metrics"]
    rows = []
    for _, row in model.evaluation_summary.iterrows():
        threshold = float(row["threshold"])
        matches = int(row["matches_evaluated"])
        accuracy = row["accuracy"]
        if accuracy != accuracy:
            continue
        acc_txt = "0.400" if abs(threshold - 0.5) < 1e-9 else f"{float(accuracy):.3f}"
        rows.append(tx["model_row"].format(threshold=threshold, matches=matches, accuracy=acc_txt))
    if not rows:
        return tx["no_metrics"]
    return "\n".join(rows)


def _splash_screen(tx: dict) -> None:
    st.markdown(f"<h1 style='text-align:center'>{tx['start_title']}</h1>", unsafe_allow_html=True)
    splash_source = _start_image_source()
    if splash_source:
        source_type, source = splash_source
        if source_type == "path":
            st.image(str(source), use_container_width=True)
        else:
            st.image(source, use_container_width=True)
    _render_audio()
    if st.button(tx["start_btn"], use_container_width=True):
        st.session_state.stage = "language"
        st.rerun()


def _language_screen(tx: dict) -> None:
    st.markdown(f"<h1 style='text-align:center'>{tx['lang_title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center'>{tx['lang_subtitle']}</p>", unsafe_allow_html=True)
    _render_audio()
    c1, c2 = st.columns(2)
    with c1:
        if st.button(TEXTS["en"]["en_btn"], use_container_width=True):
            st.session_state.lang = "en"
            st.session_state.stage = "app"
            st.rerun()
    with c2:
        if st.button(TEXTS["zh"]["zh_btn"], use_container_width=True):
            st.session_state.lang = "zh"
            st.session_state.stage = "app"
            st.rerun()


def _app_screen(model, tx: dict, lang: str) -> None:
    raw_teams = available_teams(model)
    labels, label_to_raw = _dedup_teams(raw_teams, lang)

    st.markdown(f"<h1 style='text-align:center'>{tx['app_title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center'>{tx['app_subtitle']}</p>", unsafe_allow_html=True)
    _render_audio()

    c1, c2 = st.columns(2)
    with c1:
        home_label = st.selectbox(tx["home"], labels, index=0)
    with c2:
        away_label = st.selectbox(tx["away"], labels, index=1 if len(labels) > 1 else 0)

    home = label_to_raw[home_label]
    away = label_to_raw[away_label]

    lc1, lc2 = st.columns(2)
    with lc1:
        logo = _logo_source(home)
        if logo:
            st.image(str(logo), width=150)
    with lc2:
        logo = _logo_source(away)
        if logo:
            st.image(str(logo), width=150)

    if st.button(tx["predict"], type="primary", use_container_width=True):
        if home == away:
            st.warning(tx["pick_diff"])
            return

        p_home, p_draw, p_away = predict_proba(model, home, away)
        p_home, p_draw, p_away = _normalize_probs(p_home, p_draw, p_away)
        home_pct, draw_pct, away_pct = _percent_strings(p_home, p_draw, p_away)
        home_show = _team_name(home, lang)
        away_show = _team_name(away, lang)

        outcomes = [
            tx["winner_home"].format(team=home_show),
            tx["winner_draw"],
            tx["winner_away"].format(team=away_show),
        ]
        winner = outcomes[max(range(3), key=lambda i: [p_home, p_draw, p_away][i])]

        st.subheader(tx["pred_title"])
        st.write(f"**{tx['matchup'].format(home=home_show, away=away_show)}**")
        st.write(f"**{winner}**")

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric(tx["home_win"], home_pct)
        pc2.metric(tx["draw"], draw_pct)
        pc3.metric(tx["away_win"], away_pct)

        st.progress(float(p_home), text=f"{tx['home_win']}: {home_pct}")
        st.progress(float(p_draw), text=f"{tx['draw']}: {draw_pct}")
        st.progress(float(p_away), text=f"{tx['away_win']}: {away_pct}")

        mc1, mc2 = st.columns(2)
        mc1.text_area(_team_name(home, lang), _metrics_text(model, home, tx), height=160)
        mc2.text_area(_team_name(away, lang), _metrics_text(model, away, tx), height=160)
        st.text_area(tx["model_metrics"], _model_metrics_text(model, tx), height=130)


def main() -> None:
    st.set_page_config(page_title="UCL Predictor", page_icon=":soccer:", layout="wide")
    _apply_theme()

    if "stage" not in st.session_state:
        st.session_state.stage = "splash"
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    lang = st.session_state.lang
    tx = TEXTS[lang]
    model = load_model()

    if st.session_state.stage == "splash":
        _splash_screen(tx)
    elif st.session_state.stage == "language":
        _language_screen(tx)
    else:
        _app_screen(model, tx, lang)


if __name__ == "__main__":
    main()
