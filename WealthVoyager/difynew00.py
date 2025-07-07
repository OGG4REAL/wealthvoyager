"""
WealthVoyager / BDI 系统
~~~~~~~~~~~~~~~~~~~~~~~~
整合版本，包含：
1. 投资组合优化
2. 智能对话/Agent模拟
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import re
import asyncio
import platform
import ast
import requests
import datetime
import markdown as md

# 添加必要的系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

# 创建日志目录
log_dir = os.path.join(parent_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.DEBUG if os.getenv('DEBUG') == 'true' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加调试信息
logger.debug("当前工作目录: %s", os.getcwd())
logger.debug("Python路径: %s", sys.path)

# ---- 3rd‑party / project helpers -------------------------------------------
try:
    from baseagent02 import InvestmentAdvisor
    from behavior import get_behavior_metrics_by_type
    from config import (
        COV_MATRIX,
        DEFAULT_ASSETS,
        OPENAI_API_BASE,
        OPENAI_API_KEY,
        PRESET_COVARIANCE,
    )
    from firecrawl_client_as import FirecrawlClient
    from config_firecrawl import Config as FirecrawlConfig
    from infor import build_base_config, handle_send
    from portfolioptima import (
        chat,
        extract_last_entry,
        llm_profile_extract,
        portfolio_optimization,
    )
    logger.debug("所有依赖模块导入成功")
except ImportError as e:
    logger.error(f"导入模块失败: {str(e)}")
    st.error(f"导入模块失败: {str(e)}")
    st.error("请确保所有依赖模块都在正确的位置")
    st.stop()

# ---------------------------------------------------------------------------
# Global constants & settings
# ---------------------------------------------------------------------------
DEBUG = False  # set True for server‑side console logs

PRESET_RETURNS: Dict[str, float] = {
    "A股": 0.0848,
    "债券": 0.0322,
    "REITs": 0.0843,
    "港股": 0.0839,
    "美股": 0.1038,
    "黄金": 0.0433,
    "大宗商品": 0.0318,
}

# 设置页面配置
st.set_page_config(
    page_title="BDI 系统 – 智能对话 & 投资助手",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS (centralised – easier to maintain)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ===== Reset & fonts =================================================== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* ===== Sidebar ========================================================= */
section[data-testid='stSidebar'] > div:first-child {
  background: linear-gradient(180deg,#e0ecfa 0%,#b6d0f7 100%); /* 更浅的蓝色渐变 */
  color: #222 !important; /* 深色文字 */
}
section[data-testid='stSidebar'] label, section[data-testid='stSidebar'] span, section[data-testid='stSidebar'] div {
  color: #222 !important;
  font-weight: 600 !important;
  font-size: 1.08rem !important;
}
section[data-testid='stSidebar'] .css-1v0mbdj, /* radio按钮label */
section[data-testid='stSidebar'] .css-1q7i5l3 {
  color: #222 !important;
  font-weight: 700 !important;
}
section[data-testid='stSidebar'] .css-1v0mbdj[aria-checked="true"],
section[data-testid='stSidebar'] .css-1q7i5l3[aria-checked="true"] {
  background: #2563eb !important;
  color: #fff !important;
  border-radius: 8px !important;
}

/* Quick status chips */
.status-chip { display:inline-block; padding:2px 8px; margin:0 4px;
               border-radius:12px; font-size:0.75rem; }
.status-green{ background:#10b981; }
.status-red  { background:#ef4444; }
.status-blue { background:#3b82f6; }

/* ===== Cards =========================================================== */
.stCard{background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;
        box-shadow:0 4px 12px rgba(0,0,0,.08);transition:.3s ease}
.stCard:hover{transform:translateY(-2px);box-shadow:0 6px 16px rgba(0,0,0,.12)}

/* ===== Primary metric cards =========================================== */
.metric-card{background:linear-gradient(135deg,#1E3A8A 0%,#3B82F6 100%);
             color:#fff;border-radius:12px;padding:20px;margin-bottom:20px;}
.metric-card h2{margin:0;font-size:1.75rem}

/* ===== Buttons ========================================================= */
.stButton>button{background:linear-gradient(135deg,#1E3A8A 0%,#3B82F6 100%);
                 color:#fff;border:none;border-radius:8px;padding:10px 20px;}
.stButton>button:hover{transform:translateY(-2px);
                       box-shadow:0 4px 12px rgba(59,130,246,.2)}

/* ===== Error messages ================================================== */
.error-message {
    background-color: #fee2e2;
    border: 1px solid #ef4444;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    color: #991b1b;
}

/* ===== Success messages ================================================ */
.success-message {
    background-color: #dcfce7;
    border: 1px solid #22c55e;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    color: #166534;
}

/* ===== Loading spinners ================================================ */
.stSpinner {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def safe_num(val: Any, ndigits: int = 2) -> str:
    """Robust numeric formatter (returns raw string if cast fails)."""
    try:
        f = float(val)
        fmt = f"{{:,.{ndigits}f}}" if ndigits else "{{:,}}"
        return fmt.format(f)
    except Exception as e:
        logger.warning(f"数值格式化失败: {str(e)}")
        return str(val)

def render_metric_card(title: str, value: float, change: float = None, change_color: str = "#4ADE80", show_percent: bool = False) -> None:
    """渲染指标卡片"""
    value_str = f"{value:.2f}%" if show_percent else f"{value:,.2f}"
    st.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <h2>{value_str}</h2>
        {f'<p style="color: {change_color};">{change:+.1f}%</p>' if change is not None else ''}
    </div>
    """, unsafe_allow_html=True)

def render_status_card(title: str, status: str, last_update: str = None) -> None:
    """渲染状态卡片"""
    st.markdown(f"""
    <div class="stCard">
        <h3>{title}</h3>
        <p style="color: #4ADE80;">{status}</p>
        {f'<p style="color: #6B7280;">最后更新：{last_update}</p>' if last_update else ''}
    </div>
    """, unsafe_allow_html=True)

def render_pie_chart(data: Dict[str, List], title: str) -> None:
    """渲染饼图"""
    df = pd.DataFrame(data)
    fig = px.pie(df, values='比例', names='类别', color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def render_line_chart(dates: List[str], values: List[float], title: str) -> None:
    """渲染折线图"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers',
                            line=dict(color='#3B82F6', width=3),
                            marker=dict(size=8)))
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="数值",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

def strip_markdown_code_block(s: str) -> str:
    """去除 markdown 代码块包裹（如 ```json ... ```）"""
    s = s.strip()
    s = re.sub(r"^```(?:json|python|text)?\\s*", "", s)
    s = re.sub(r"\\s*```$", "", s)
    return s.strip()

def firecrawl_deep_research_sync(query: str, *, max_depth: int = 2, time_limit: int = 60, max_urls: int = 10) -> Dict[str, Any]:
    """同步执行深度研究"""
    try:
        client = FirecrawlClient(FirecrawlConfig())
        async def _run():
            await client.initialize()
            result = await client.debug_deep_research(query, max_depth, time_limit, max_urls)
            await client.close()
            return result
        return asyncio.run(_run())
    except Exception as e:
        logger.error(f"执行深度研究失败: {str(e)}")
        st.error("执行深度研究失败，请稍后重试")
        return {}

def extract_json_array_from_mcp_result(mcp_result):
    # 1. 拿到 content[0]
    if isinstance(mcp_result, dict) and 'content' in mcp_result and mcp_result['content']:
        raw = mcp_result['content'][0]
    elif isinstance(mcp_result, str):
        try:
            mcp_result_json = json.loads(mcp_result)
            raw = mcp_result_json.get('content', [''])[0]
        except Exception:
            raw = mcp_result
    else:
        raw = str(mcp_result)
    # 2. 用正则提取 text='[ ... ]'
    match = re.search(r"text='(\[.*?\])'", raw, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 兜底：直接找第一个 [ ... ] 块
        match2 = re.search(r"(\[.*\])", raw, re.DOTALL)
        if match2:
            json_str = match2.group(1)
        else:
            json_str = raw.strip()
    # 3. 去掉 markdown 代码块
    json_str = strip_markdown_code_block(json_str)
    return json_str

def convert_userconfig_to_profile(userconfig: Dict[str, Any]) -> Dict[str, Any]:
    """将用户配置转换为用户画像"""
    try:
        investment_purpose = userconfig.get("investment_purpose", "wealth_growth")
        assets = userconfig.get("assets", [])
        allocations = userconfig.get("current_allocation", [])
        asset_allocation = dict(zip(assets, allocations)) if assets and allocations else {}
        behavior_metrics = get_behavior_metrics_by_type(investment_purpose)
        
        profile = {
            "investment_purpose": investment_purpose,
            "target_amount": userconfig.get("target_amount", 1000000),
            "initial_investment": userconfig.get("initial_investment", 100000),
            "investment_years": userconfig.get("investment_years", 10),
            "volatility_tolerance": userconfig.get("acceptable_volatility", 0.2),
            "max_acceptable_loss": userconfig.get("max_acceptable_loss", 0.3),
            "asset_allocation": asset_allocation,
            "liquidity_requirement": userconfig.get("liquidity_requirement", "medium"),
            "leverage_allowed": userconfig.get("leverage_allowed", False),
            "restricted_assets": userconfig.get("restricted_assets", []),
            "investor_type": investment_purpose,
            "risk_tolerance": userconfig.get("risk_tolerance", "Medium"),
            "behavior_metrics": behavior_metrics
        }
        
        return profile
    except Exception as e:
        logger.error(f"转换用户配置失败: {str(e)}")
        st.error("转换用户配置失败，请检查输入数据")
        return {}

def process_investment_request(base_config: Dict[str, Any], user_assets: List[str], user_allocation: List[float]) -> Dict[str, Any]:
    """处理投资请求的核心逻辑"""
    try:
        # 获取资产收益率预测
        search_query = (
            "请你根据权威数据和最新研究，列出未来10年中国主要资产类别（A股、债券、REITs、港股、美股、黄金、大宗商品）的预期年化收益率及其信息来源，"
            "输出格式如下（严格用JSON数组，每个元素包含资产类别、预期收益率（百分数）、信息来源）：\n"
            "[\n"
            "  {\"资产类别\": \"A股\", \"预期收益率\": \"8%\", \"来源\": \"央视经济\"},\n"
            "  {\"资产类别\": \"债券\", \"预期收益率\": \"3%\", \"来源\": \"新浪财经\"},\n"
            "  ...\n"
            "]\n"
            "只输出JSON，不要输出其他内容。"
        )
        
        mcp_result = firecrawl_deep_research_sync(search_query)
        json_str = extract_json_array_from_mcp_result(mcp_result)
        json_str_clean = json_str.replace('\\n', '').replace('\n', '').replace('\r', '')
        
        try:
            asset_list = json.loads(json_str_clean)
        except Exception as e1:
            try:
                asset_list = ast.literal_eval(json_str_clean)
            except Exception as e2:
                mean_returns = [PRESET_RETURNS.get(asset, 0.05) for asset in user_assets]
                sources = ["模拟数据"] * len(user_assets)
                asset_list = []

        mean_returns = []
        sources = []
        for asset in user_assets:
            found = next((item for item in asset_list if item["资产类别"] == asset), None)
            if found:
                rate = float(found["预期收益率"].replace("%", "")) / 100
                mean_returns.append(rate)
                sources.append(found["信息来源"] if "信息来源" in found else found.get("来源", "未知"))
            else:
                mean_returns.append(PRESET_RETURNS.get(asset, 0.05))
                sources.append("模拟数据")

        # 构建协方差矩阵
        if user_assets == DEFAULT_ASSETS:
            cov_matrix = COV_MATRIX
        else:
            covariance = PRESET_COVARIANCE
            cov_matrix = [[covariance[asset_i][asset_j] for asset_j in user_assets] for asset_i in user_assets]

        # 执行投资组合优化
        optimization_result = portfolio_optimization(mean_returns, cov_matrix, base_config)
        
        return {
            "mean_returns": mean_returns,
            "sources": sources,
            "optimization_result": optimization_result
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "mean_returns": [PRESET_RETURNS.get(asset, 0.05) for asset in user_assets],
            "sources": ["模拟数据"] * len(user_assets),
            "optimization_result": None
        }

# ========== 结果保存/读取机制 ========== #
# 结果文件目录
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(results_dir, exist_ok=True)

def get_result_path(page: str, session_id: str = 'default') -> str:
    """生成结果文件路径，page为页面标识，session_id为用户/会话标识（默认单用户）"""
    return os.path.join(results_dir, f"{page}_{session_id}.json")

def save_result(page: str, data: dict, session_id: str = 'default'):
    """保存结果到本地json文件，自动将所有numpy类型和自定义类型转换为标准Python类型，避免序列化报错（兼容numpy 2.0及以上）"""
    def convert(obj):
        # 递归转换所有numpy类型和自定义类型为标准Python类型
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (str, float, int, bool)) or obj is None:
            return obj
        else:
            # 兜底：所有非标准类型直接转为字符串
            return str(obj)
    path = get_result_path(page, session_id)
    data = convert(data)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已成功保存到: {path}")
    except Exception as e:
        logger.error(f"保存结果到 {path} 失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        try:
            import streamlit as st
            st.error(f"保存结果到 {path} 失败: {e}")
        except Exception:
            pass


def load_result(page: str, session_id: str = 'default'):
    """从本地json文件读取结果，若不存在则返回None"""
    path = get_result_path(page, session_id)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None
# ========== 结果保存/读取机制 END ========== #

# ---------------------------------------------------------------------------
# 新增：市场新闻与解读页面
# ---------------------------------------------------------------------------

def deepseek_news_interpretation(raw_content, profile, api_key, api_base, max_tokens=2048):
    """调用deepseek API，对新闻分段并结合用户画像生成个性化解读，返回结构化JSON数组（含url）"""
    prompt = f"""
你是一个专业的金融投资顾问。请阅读以下市场新闻原文（内容由MCP自动抓取，格式可能包含markdown链接、括号内网址或直接URL），请自动识别每条新闻的正文和原文网址（如有），并结合给定的用户画像，对每条新闻做个性化解读。输出格式为JSON数组，每个元素包含"news"（新闻内容）、"interpretation"（个性化解读）、"url"（原文网址，如无则留空）三个字段。不要输出多余内容。

【市场新闻原文】
{raw_content}

【用户画像】
{profile}

【输出格式示例】
[
  {{"news": "新闻内容1", "interpretation": "个性化解读1", "url": "https://xxx.com"}},
  {{"news": "新闻内容2", "interpretation": "个性化解读2", "url": ""}}
]
"""
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    import streamlit as st
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        import re, json
        st.write("[DEBUG] deepseek原始返回:")
        st.write(content)
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            arr = match.group(0)
            st.write("[DEBUG] 正则提取到的JSON片段:")
            st.write(arr)
            try:
                return json.loads(arr)
            except Exception as e:
                st.write("[DEBUG] JSON解析异常:")
                st.write(str(e))
                raise
        else:
            st.write("[DEBUG] 未匹配到JSON数组")
            return []
    except Exception as e:
        st.write("[DEBUG] deepseek调用或解析异常:")
        st.write(str(e))
        raise

def page_market_news():
    """市场新闻与解读页面"""
    import streamlit as st
    import os
    import json
    st.markdown('<div class="main-title" style="font-size:2.1rem;font-weight:700;margin-bottom:18px;">Voyager • NewsCrawler</div>', unsafe_allow_html=True)
    # 顶部加今日日期
    today = datetime.date.today().strftime('%Y-%m-%d')
    st.markdown(f"<div style='font-size:1.1rem;color:#6B7280;margin-bottom:10px;'>今日日期：{today}</div>", unsafe_allow_html=True)
    # 1. 读取结构化新闻解读结果
    result_path = os.path.join(results_dir, 'news_interpretation_default.json')
    agent_path = os.path.join(results_dir, 'agent_default.json')
    profile_path = os.path.join(results_dir, 'profile_default.json')
    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            news_list = json.load(f)
    else:
        # 2. 若无结构化结果，读取原始新闻和profile，调用deepseek生成
        if not os.path.exists(agent_path) or not os.path.exists(profile_path):
            st.info("请先完成agent模拟和用户画像生成。")
            return
        with open(agent_path, 'r', encoding='utf-8') as f:
            agent_data = json.load(f)
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        # 提取raw_content
        logs = agent_data.get('logs', [])
        raw_content = None
        for entry in logs:
            if entry.get('step') == 'raw_content':
                content = entry.get('content', [])
                if content:
                    raw_content = content[0]
                    break
        if not raw_content:
            st.warning("未找到原始市场新闻内容。")
            return
        # deepseek API参数
        from config import OPENAI_API_KEY, OPENAI_API_BASE
        with st.spinner('正在调用大模型分段并生成个性化解读...'):
            news_list = deepseek_news_interpretation(raw_content, profile, OPENAI_API_KEY, OPENAI_API_BASE)
        if not news_list:
            st.error("大模型返回内容解析失败，请稍后重试。")
            return
        # 保存结构化结果
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(news_list, f, ensure_ascii=False, indent=2)
        st.success("已生成结构化新闻解读，页面将自动刷新。请稍候...")
        st.rerun()
    # 3. 展示每条新闻及解读
    if not news_list:
        st.info("暂无可展示的新闻解读。")
        return
    for item in news_list:
        news = item.get('news', '').strip()
        interp = item.get('interpretation', '').strip()
        url = item.get('url', '').strip()
        interp_html = f"<div style='color:#2563eb;font-size:0.98rem;font-weight:500;margin-bottom:2px;'><b>个性化解读：</b>{interp}</div>" if interp else ""
        url_html = f"<div style='margin-top:6px;'><a href='{url}' target='_blank' style='color:#2563eb;font-size:0.95rem;'>🔗 原文链接</a></div>" if url else ""
        st.markdown(f"""
        <div class='stCard' style='background:#f3f4f6;margin-bottom:14px;padding:16px 18px 12px 18px;border-radius:10px;'>
            <div style='font-weight:600;font-size:1.05rem;margin-bottom:4px;color:#222;'>{news}</div>
            {interp_html}
            {url_html}
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Core Streamlit tabs
# ---------------------------------------------------------------------------

def main():
    """主函数"""
    try:
        # 侧边栏页面选择
        page = st.sidebar.radio(
            "选择功能",
            ["🏠 首页", "📈α 收益工坊", "🤖双智投对话引擎", "📰 市场新闻与解读"],
            index=0
        )

        # 顶部主标题区
        if page == "🏠 首页":
            st.markdown("""
            <div style='text-align: center; padding: 32px 0 18px 0;'>
                <h1 style='font-size:2.3rem;font-weight:800;margin-bottom:8px;'>WealthVoyager AI 投资助手</h1>
                <div style='color:#374151;font-size:1.18rem;font-weight:500;'>智能投资决策支持系统</div>
            </div>
            """, unsafe_allow_html=True)
        elif page == "📈α 收益工坊":
            st.markdown("""
            <div style='text-align: center; padding: 32px 0 18px 0;'>
                <h1 style='font-size:2.3rem;font-weight:800;margin-bottom:8px;'>Voyager • AlphaForge</h1>
                <div style='color:#374151;font-size:1.13rem;font-weight:500;'>智能资产配置与收益优化工坊</div>
            </div>
            """, unsafe_allow_html=True)
        elif page == "🤖双智投对话引擎":
            st.markdown("""
            <div style='text-align: center; padding: 32px 0 18px 0;'>
                <h1 style='font-size:2.3rem;font-weight:800;margin-bottom:8px;'>Voyager • DualAdvisor</h1>
                <div style='color:#374151;font-size:1.13rem;font-weight:500;'>AI双智能顾问对话与模拟</div>
            </div>
            """, unsafe_allow_html=True)
        elif page == "📰 市场新闻与解读":
            st.markdown("""
            <div style='text-align: center; padding: 32px 0 18px 0;'>
                <h1 style='font-size:2.3rem;font-weight:800;margin-bottom:8px;'>Voyager • NewsCrawler</h1>
                <div style='color:#374151;font-size:1.13rem;font-weight:500;'>市场新闻智能解读与个性化分析</div>
            </div>
            """, unsafe_allow_html=True)

        # 根据选择渲染对应页面
        if page == "🏠 首页":
            page_home()
        elif page == "📈α 收益工坊":
            page_portfolio_optimization()
        elif page == "🤖双智投对话引擎":
            page_agent_simulation()
        else:
            page_market_news()
        
    except Exception as e:
        logger.error(f"主程序运行失败: {str(e)}")
        st.error("程序运行出错，请刷新页面重试")

    # 在侧边栏左下角添加AssistHub客服标记
    st.sidebar.markdown("""
    <style>
    #custom-assisthub-sidebar {
        position: fixed;
        left: 0;
        bottom: 24px;
        width: 260px;
        z-index: 999;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        padding-left: 18px;
    }
    #custom-assisthub-bubble {
        background: linear-gradient(90deg,#e0ecfa 0%,#f3f8fe 100%);
        border: 1.5px solid #2563eb;
        border-radius: 18px 18px 18px 4px;
        box-shadow: 0 2px 8px rgba(59,130,246,0.08);
        padding: 14px 18px 12px 16px;
        margin-bottom: 6px;
        color: #1e293b;
        font-size: 1.08rem;
        line-height: 1.7;
        max-width: 220px;
        font-weight: 600;
        display: flex;
        align-items: center;
    }
    #custom-assisthub-sidebar .icon {
        font-size: 1.45rem;
        margin-right: 10px;
        vertical-align: -2px;
    }
    </style>
    <div id="custom-assisthub-sidebar">
        <div id="custom-assisthub-bubble">
            <span class="icon">💬</span>
            有问题？随时咨询AssistHub智能客服
        </div>
    </div>
    """, unsafe_allow_html=True)

def page_portfolio_optimization():
    """投资组合优化页面"""
    try:
        st.markdown('<div class="main-title" style="font-size:2.1rem;font-weight:700;margin-bottom:18px;">Voyager • AlphaForge</div>', unsafe_allow_html=True)
        
        # 初始化会话状态
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = None
        if "dify_response" not in st.session_state:
            st.session_state.dify_response = ""
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        if "base_config_ready" not in st.session_state:
            st.session_state.base_config_ready = False
        if "optimization_results" not in st.session_state:
            st.session_state.optimization_results = None
        if "plan_confirmed" not in st.session_state:
            st.session_state.plan_confirmed = False
        if "selected_config" not in st.session_state:
            st.session_state.selected_config = None
        if "selected_success_rate" not in st.session_state:
            st.session_state.selected_success_rate = None

        # 创建两列布局
        col1, col2 = st.columns([1, 2])

        with col1:
            import os
            import json
            # 优先用 Dify API 获取历史
            conversation_id = st.session_state.get('conversation_id', None)
            history = []
            if conversation_id:
                api_history = get_dify_conversation_history(conversation_id)
                if api_history:
                    history = api_history
            # fallback 到本地文件方案
            if not history:
                log_path = os.path.join(os.path.dirname(__file__), "conversation_log.json")
                if os.path.exists(log_path):
                    try:
                        with open(log_path, "r", encoding="utf-8") as f:
                            log_data = json.load(f)
                        if isinstance(log_data, list) and log_data:
                            latest_id = log_data[-1].get("conversation_id", None)
                            history = [item for item in log_data if item.get("conversation_id") == latest_id]
                    except Exception as e:
                        history = [{"role": "系统", "content": f"读取对话历史失败: {e}"}]
                else:
                    history = [{"role": "系统", "content": "未找到对话历史文件"}]

            st.markdown("""
            <div style='background:#fff;border-radius:14px;padding:20px 18px 16px 18px;box-shadow:0 4px 16px rgba(0,0,0,0.08);margin-bottom:20px;'>
              <div style='display:flex;align-items:center;margin-bottom:12px;'>
                <span style='font-size:1.5rem;margin-right:10px;'>🤖</span>
                <span style='font-size:1.18rem;font-weight:700;color:#2563eb;'>AssistHub 客服</span>
              </div>
              <div style='color:#374151;font-size:1.05rem;margin-bottom:8px;'>
                请输入您的投资问题，AssistHub会为您提供智能解答。<br>
                如需自定义资产类别和持仓比例，请在下方输入，留空则使用默认资产配置。
              </div>
            """, unsafe_allow_html=True)

            # 聊天历史折叠区
            with st.expander("展开历史记录", expanded=False):
                if history:
                    for msg in history:
                        if msg.get('role') == 'user':
                            st.markdown(
                                f"<div style='text-align:left;margin-bottom:8px;'>"
                                f"<span style='display:inline-block;background:#2563eb;color:#fff;padding:7px 16px;border-radius:16px;font-size:1rem;max-width:80%;word-break:break-all;'>我：{msg.get('content','')}</span>"
                                "</div>",
                                unsafe_allow_html=True,
                            )
                        elif msg.get('role') == 'assistant':
                            st.markdown(
                                f"<div style='text-align:right;margin-bottom:8px;'>"
                                f"<span style='display:inline-block;background:#10b981;color:#fff;padding:7px 16px;border-radius:16px;font-size:1rem;max-width:80%;word-break:break-all;'>AssistHub：{msg.get('content','')}</span>"
                                "</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.markdown("<div style='color:#6B7280;text-align:center;'>暂无历史对话</div>", unsafe_allow_html=True)

            # 保留原有两个输入框和发送按钮
            st.text_input("请输入您的投资问题:", key="user_input_key")
            st.text_input(
                "请输入您的资产类别和持仓比例（允许范围：A股, 债券, REITs, 港股, 美股, 黄金, 大宗商品；若留空则使用默认资产：股票, 债券, 房地产信托）：",
                key="current_allocation"
            )
            st.button("发送", on_click=handle_send)

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("### 📊 优化结果")
            # 优先尝试读取本地保存的结果，避免重复计算
            saved_result = load_result('portfolio')
            if saved_result:
                st.write("已加载本地保存的优化结果（如需重新计算请删除results/portfolio_default.json）")
                mcp_data = saved_result.get('mcp_data', None)
                opt = saved_result.get('optimization', None)
                if opt:
                    render_portfolio_optimization_result(opt, mcp_data)
                else:
                    st.info("未找到优化结果数据。")
                return  # 已有结果直接返回

            # 提取数据
            extracted_data = extract_last_entry(st.session_state.conversation_id)
            required_keys = ["目标金额", "投资年限", "初始资金", "可接受的资产波动率"]
            missing_keys = [k for k in required_keys if k not in extracted_data]
            if not extracted_data or missing_keys:
                st.info("请先在左侧输入您的投资问题和相关选项，我们将为您生成个性化的投资方案。")
                return

            # 生成基础配置
            if not st.session_state.get('base_config_ready', False):
                history_text = "\n".join([entry["input"] + "\n" + entry["response"] for entry in st.session_state.conversation_history])
                profile = llm_profile_extract(history_text)
                for k, v in profile.items():
                    extracted_data[k] = v
                base_config = build_base_config(extracted_data)
                st.session_state.base_config = base_config
                st.session_state.base_config_ready = True
                # 新增：保存base_config到本地，便于后续页面读取
                save_result('profile', base_config)
            else:
                base_config = st.session_state.base_config

            # 资产分析范围选择
            raw_prob = extracted_data.get("成功概率", "0%")
            try:
                prob = float(raw_prob.strip('%')) / 100
            except ValueError:
                prob = 0

            if prob < 0.6:
                st.markdown("#### 📌 请选择资产分析范围")
                asset_analysis_mode = st.radio(
                    "您希望如何进行资产分析和优化？",
                    ["使用我输入的资产类别", "使用所有可选资产类别（A股, 债券, REITs, 港股, 美股, 黄金, 大宗商品）"],
                    key="asset_analysis_mode_radio"
                )
                if asset_analysis_mode == "使用所有可选资产类别（A股, 债券, REITs, 港股, 美股, 黄金, 大宗商品）":
                    full_assets = ["A股", "债券", "REITs", "港股", "美股", "黄金", "大宗商品"]
                    extracted_data["资产类别"] = full_assets
                    extracted_data["当前资产配置"] = [round(1/7, 3)] * 7
                base_config = build_base_config(extracted_data)

            # 方案生成与选择
            if st.session_state.dify_response and "start_optimization" not in st.session_state:
                st.write("### 已获取到您的投资信息")
                col1, col2 = st.columns(2)
                with col1:
                    render_metric_card("目标金额", base_config.get('target_amount', 0), show_percent=False)
                    render_metric_card("投资年限", base_config.get('investment_years', 0), show_percent=False)
                with col2:
                    render_metric_card("初始投资", base_config.get('initial_investment', 0), show_percent=False)
                    render_metric_card("可接受波动率", base_config.get('acceptable_volatility', 0)*100, show_percent=True)
                if st.button("开始生成投资方案", key="start_optimization_button"):
                    st.session_state.start_optimization = True
                    st.rerun()

            if st.session_state.dify_response and st.session_state.get("start_optimization", False):
                advisor = InvestmentAdvisor(base_config, OPENAI_API_KEY, OPENAI_API_BASE)
                if st.session_state.optimization_results is None:
                    with st.spinner('正在生成投资方案，请稍候...'):
                        results = {
                            "A": advisor.run_optimization(["target_amount"], max_rounds=3),
                            "B": advisor.run_optimization(["investment_years"], max_rounds=3),
                            "C": advisor.run_optimization(["investment_years", "initial_investment"], max_rounds=3)
                        }
                        st.session_state.optimization_results = results
                results = st.session_state.optimization_results

                st.write("\n📊 **所有可选方案：**")
                for key, label in zip(["A", "B", "C"], ["方案A：调整目标金额", "方案B：调整投资年限", "方案C：调整年限和初始投资"]):
                    with st.expander(label, expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            render_metric_card("目标金额", results[key][0]['target_amount'], show_percent=False)
                            render_metric_card("投资年限", results[key][0]['investment_years'], show_percent=False)
                        with col2:
                            render_metric_card("初始投资", results[key][0]['initial_investment'], show_percent=False)
                            render_metric_card("成功概率", results[key][1]*100, show_percent=True)
                        st.markdown(f"**达标状态**: {'✅ 已达标' if results[key][1] >= 0.6 else '⚠️ 未达标'}")

                selected_plan = st.radio(
                    "请选择您偏好的投资方案：",
                    ["A", "B", "C"],
                    format_func=lambda x: {"A": "方案 A：调整目标金额", "B": "方案 B：调整投资年限", "C": "方案 C：调整年限和初始投资"}[x],
                    key="plan_selection"
                )

                if st.button("确认选择", key="confirm_plan"):
                    st.session_state.selected_config = results[selected_plan][0]
                    st.session_state.selected_success_rate = results[selected_plan][1]
                    st.session_state.plan_confirmed = True
                    st.success(f"您已选择{selected_plan}方案！")
                    st.rerun()

            # 资产配置优化
            if st.session_state.get("plan_confirmed", False):
                st.write("\n## 📈 基于选定方案的资产配置优化")
                selected_config = st.session_state.selected_config
                user_assets = base_config["assets"]

                # MCP数据获取
                with st.spinner('正在通过MCP（firecrawl）获取资产未来收益率...'):
                    try:
                        search_query = (
                            "请你根据权威数据和最新研究，列出未来10年中国主要资产类别（A股、债券、REITs、港股、美股、黄金、大宗商品）的预期年化收益率及其信息来源，"
                            "输出格式如下（严格用JSON数组，每个元素包含资产类别、预期收益率（百分数）、信息来源）：\n"
                            "[\n"
                            "  {\"资产类别\": \"A股\", \"预期收益率\": \"8%\", \"来源\": \"央视经济\"},\n"
                            "  {\"资产类别\": \"债券\", \"预期收益率\": \"3%\", \"来源\": \"新浪财经\"},\n"
                            "  ...\n"
                            "]\n"
                            "只输出JSON，不要输出其他内容。"
                        )
                        mcp_result = firecrawl_deep_research_sync(search_query)
                        json_str = extract_json_array_from_mcp_result(mcp_result)
                        json_str_clean = json_str.replace('\\n', '').replace('\n', '').replace('\r', '')
                        try:
                            asset_list = json.loads(json_str_clean)
                        except Exception as e1:
                            try:
                                asset_list = ast.literal_eval(json_str_clean)
                            except Exception as e2:
                                st.error(f"MCP返回内容解析失败: {e1} / {e2}")
                                st.write("原始内容：", json_str_clean)
                                asset_list = []

                        mean_returns = []
                        sources = []
                        for asset in user_assets:
                            found = next((item for item in asset_list if item["资产类别"] == asset), None)
                            if found:
                                rate = float(found["预期收益率"].replace("%", "")) / 100
                                mean_returns.append(rate)
                                sources.append(found["信息来源"] if "信息来源" in found else found.get("来源", "未知"))
                            else:
                                mean_returns.append(PRESET_RETURNS.get(asset, 0.05))
                                sources.append("模拟数据")
                        # 保存mcp抓取数据
                        mcp_data = {
                            "资产类别": user_assets,
                            "预期收益率": [f"{r:.2%}" for r in mean_returns],
                            "信息来源": sources
                        }
                    except Exception as e:
                        st.error(f"MCP返回内容解析失败: {e}")
                        mean_returns = [PRESET_RETURNS.get(asset, 0.05) for asset in user_assets]
                        sources = ["模拟数据"] * len(user_assets)
                        mcp_data = []

                # 可视化
                try:
                    df = pd.DataFrame({
                        "资产类别": user_assets,
                        "预期收益率": [f"{r:.2%}" for r in mean_returns],
                        "信息来源": sources
                    })
                    st.write("📊 **预测数据及来源**")
                    st.table(df)
                except Exception as e:
                    st.write("收益率可视化失败", e)

                # 协方差矩阵
                if user_assets == DEFAULT_ASSETS:
                    cov_matrix = COV_MATRIX
                else:
                    covariance = PRESET_COVARIANCE
                    cov_matrix = [[covariance[asset_i][asset_j] for asset_j in user_assets] for asset_i in user_assets]

                # 执行投资组合优化
                with st.spinner('正在优化投资组合...'):
                    full_cfg = {**base_config, **selected_config}
                    optimization_result = portfolio_optimization(mean_returns, cov_matrix, full_cfg)

                if optimization_result:
                    weights = optimization_result["weights"]
                    exp_return = optimization_result["expected_return"]
                    exp_vol = optimization_result["expected_volatility"]
                    final_amt = optimization_result["final_amount"]
                    max_drawdown_est = optimization_result["max_drawdown"]

                    st.write("\n### 🎯 最优资产配置建议")
                    # 使用plotly绘制饼图
                    fig = px.pie(
                        names=user_assets,
                        values=[w*100 for w in weights],
                        title="资产配置比例（%）"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("\n### 📊 投资组合详细信息")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        render_metric_card("预期年化收益率", exp_return*100, show_percent=True)
                    with col2:
                        render_metric_card("预期波动率", exp_vol*100, show_percent=True)
                    with col3:
                        render_metric_card("最大回撤", max_drawdown_est*100, show_percent=True)

                    st.write("\n### 💡 具体配置建议")
                    for asset, weight in zip(user_assets, weights):
                        st.write(f"- {asset}: {weight*100:.1f}%")

                    # 保存最终结果到本地，便于后续页面直接读取
                    save_result('portfolio', {
                        'mcp_data': mcp_data,
                        'optimization': {
                            'assets': user_assets,
                            'weights': weights,
                            'expected_return': exp_return,
                            'expected_volatility': exp_vol,
                            'final_amount': final_amt,
                            'max_drawdown': max_drawdown_est
                        }
                    })
                    # 美观展示
                    render_portfolio_optimization_result({
                        'assets': user_assets,
                        'weights': weights,
                        'expected_return': exp_return,
                        'expected_volatility': exp_vol,
                        'final_amount': final_amt,
                        'max_drawdown': max_drawdown_est
                    }, mcp_data)
                    # === 新增：同步portfolio资产配置到profile ===
                    import traceback
                    profile_path = os.path.join(results_dir, 'profile_default.json')
                    try:
                        print('[DEBUG] profile_path:', profile_path)
                        print('[DEBUG] user_assets:', user_assets)
                        print('[DEBUG] weights:', weights)
                        if os.path.exists(profile_path):
                            with open(profile_path, 'r', encoding='utf-8') as f:
                                profile = json.load(f)
                            print('[DEBUG] profile同步前:', profile)
                            profile['assets'] = user_assets
                            profile['current_allocation'] = [round(w, 6) for w in weights]
                            profile['asset_allocation'] = {a: round(w, 6) for a, w in zip(user_assets, weights)}
                            # 新增：同步目标金额、投资年限、初始资金
                            for k in ['target_amount', 'investment_years', 'initial_investment']:
                                if k in selected_config:
                                    profile[k] = selected_config[k]
                            with open(profile_path, 'w', encoding='utf-8') as f:
                                json.dump(profile, f, ensure_ascii=False, indent=2)
                            print('[DEBUG] profile同步后:', profile)
                        else:
                            print('[DEBUG] profile_path不存在:', profile_path)
                    except Exception as e:
                        print('[DEBUG] profile同步异常:', str(e))
                        print(traceback.format_exc())
                    # === END ===
                else:
                    st.error("无法找到满足条件的投资组合，请调整投资参数或放宽限制条件。")

    except Exception as e:
        logger.error(f"投资组合优化页面渲染失败: {str(e)}")
        st.error("投资组合优化页面加载失败，请刷新页面重试")

def render_user_profile(profile: dict):
    """美化后的用户画像展示"""
    import plotly.express as px
    import pandas as pd
    import streamlit as st
    # 1. 顶部标题
    st.markdown("""
    <div style='font-size:2rem;font-weight:600;margin-bottom:10px;'>👤 用户画像</div>
    """, unsafe_allow_html=True)

    # 2. 第一行：核心参数卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h4>目标金额</h4><h2>{profile.get('target_amount',0):,.0f} 元</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h4>初始投资</h4><h2>{profile.get('initial_investment',0):,.0f} 元</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h4>投资年限</h4><h2>{profile.get('investment_years',0)} 年</h2></div>", unsafe_allow_html=True)
    with col4:
        vol = profile.get('volatility_tolerance', profile.get('acceptable_volatility', 0))
        if vol > 1: vol = vol / 100
        st.markdown(f"<div class='metric-card'><h4>可接受波动率</h4><h2>{vol*100:.2f}%</h2></div>", unsafe_allow_html=True)

    # 3. 资产配置饼图
    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
    st.markdown("#### 资产配置")
    asset_allocation = profile.get('asset_allocation', {})
    if asset_allocation:
        df = pd.DataFrame({
            '资产': list(asset_allocation.keys()),
            '配置比例': [v*100 for v in asset_allocation.values()]
        })
        fig = px.pie(df, names='资产', values='配置比例', title='', color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无资产配置数据")

    # 4. 其他约束与信息
    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
    st.markdown("#### 投资目标与约束")
    info_map = [
        ("投资目的", profile.get('investment_purpose', '')), 
        ("流动性要求", profile.get('liquidity_requirement', '')), 
        ("允许杠杆", '是' if profile.get('leverage_allowed', False) else '否'),
        ("最大可接受回撤", f"{profile.get('max_acceptable_loss',0)*100:.1f}%"),
        ("厌恶资产", '、'.join(profile.get('restricted_assets', [])) or '无'),
        ("风险偏好", profile.get('risk_tolerance', ''))
    ]
    # 自定义美观表格
    table_html = """
    <div style='color:#6B7280;font-size:1.05rem;margin-bottom:6px;'>其他信息</div>
    <table style='width:100%;border-collapse:collapse;'>
    """
    for k, v in info_map:
        table_html += f"<tr style='border-bottom:1px solid #f3f4f6;'><td style='padding:6px 8px 6px 0;width:38%;color:#6B7280;'>{k}</td><td style='padding:6px 0 6px 8px;font-weight:600;color:#222;'>{v}</td></tr>"
    table_html += "</table>"
    st.markdown(table_html, unsafe_allow_html=True)

    # 5. 行为特征（可选，折叠）
    behavior = profile.get('behavior_metrics', {})
    if behavior:
        # 英文key转中文
        behavior_name_map = {
            "loss_aversion": "损失厌恶",
            "news_policy_sensitivity": "政策敏感度",
            "investment_experience": "投资经验",
            "real_time_emotion": "情绪波动",
            "herding_tendency": "从众倾向",
            "regret_aversion": "懊悔厌恶",
            "overconfidence": "过度自信",
            "illusion_of_control": "控制错觉",
            "decision_delay": "决策拖延"
        }
        with st.expander("行为特征（可选）", expanded=False):
            import pandas as pd
            beh_df = pd.DataFrame([
                {
                    "特征": behavior_name_map.get(k, k),
                    "分值": f"{round(v, 1)}"
                }
                for k, v in behavior.items()
            ])
            beh_df.index = [''] * len(beh_df)  # 关键：去掉index显示
            st.table(beh_df)

def extract_and_format_llm_contents(obj):
    """递归收集所有content字段文本，或字符串内所有content='...'片段和分段内容，去重、顺序拼接，格式化，返回list[str]"""
    import re
    contents = []
    def _collect(o):
        if isinstance(o, dict):
            if 'content' in o and isinstance(o['content'], str):
                contents.append(o['content'])
            for v in o.values():
                _collect(v)
        elif isinstance(o, list):
            for v in o:
                _collect(v)
        elif isinstance(o, str):
            # 1. 提取所有content='...'片段
            found = re.findall(r"content='([^']+)'", o)
            contents.extend(found)
            # 2. 去掉所有content='...'片段后的剩余内容
            s = re.sub(r"content='[^']+'", '', o)
            # 3. 按分段符号切分剩余内容
            # 支持 1. 2. 3.、\n\n、\n【 等
            segs = re.split(r'(?:\n\d+\.\s*)|(?:\n{2,})|(?:\n【)', s)
            for seg in segs:
                seg = seg.strip()
                # 过滤掉LLM对象repr段落
                if seg and not ("ChatCompletion(" in seg or "Choice(" in seg):
                    contents.append(seg)
    _collect(obj)
    # 去重，保持顺序
    seen = set()
    result = []
    for c in contents:
        c = c.strip()
        if c and c not in seen:
            # 转义字符还原
            c = c.replace('\\n', '\n').replace('\\t', '    ').replace('\n', '\n').replace('\t', '    ')
            # 去除多余空行
            c = re.sub(r'\n{3,}', '\n\n', c)
            result.append(c)
            seen.add(c)
    return result

def page_agent_simulation():
    """智能对话/Agent模拟页面"""
    try:
        st.markdown('<div class="main-title" style="font-size:2.1rem;font-weight:700;margin-bottom:18px;">Voyager • DualAdvisor</div>', unsafe_allow_html=True)
        # 每次都强制从本地读取profile（base_config），保证与profile_default.json同步
        loaded_profile = load_result('profile')
        if loaded_profile:
            st.session_state.base_config = loaded_profile
            st.info("已自动从本地加载最新用户画像。")
        else:
            st.info("请先在主流程输入投资参数，生成用户画像后再切换到本页。")
            return
        userconfig = st.session_state.base_config
        profile = convert_userconfig_to_profile(userconfig)
        # 创建两列布局
        col1, col2 = st.columns([1, 2])
        with col1:
            render_user_profile(profile)
            if st.button("开始Agent模拟", key="start_agent_sim_btn"):
                with st.spinner("正在运行完整Agent模拟流程..."):
                    try:
                        from WealthVoyager.investment_dialogue.main import main_async
                        logs, daily_report = main_async(profile)
                        # 保存模拟日志和日报到本地，便于后续页面直接读取
                        save_result('agent', {
                            'logs': logs,
                            'daily_report': daily_report
                        })
                    except Exception as e:
                        import traceback
                        st.error(f"Agent模拟出错: {e}\n{traceback.format_exc()}")
        # 只保留底部展示每日简报
        agent_result = load_result('agent')
        if agent_result and 'daily_report' in agent_result:
            with col2:
                st.markdown("### 📝 每日简报")
                daily_report = agent_result['daily_report']
                contents = extract_and_format_llm_contents(daily_report)
                for i, text in enumerate(contents):
                    st.markdown(text, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Agent模拟页面渲染失败: {str(e)}")
        st.error("Agent模拟页面加载失败，请刷新页面重试")

def page_home():
    import streamlit as st
    import os
    import json
    import datetime
    import plotly.express as px
    st.markdown('<div class="main-title">🏠 欢迎使用WealthVoyager AI 投资助手</div>', unsafe_allow_html=True)
    today = datetime.date.today().strftime('%Y-%m-%d')
    st.markdown(f"<div style='font-size:1.1rem;color:#6B7280;margin-bottom:10px;'>今日日期：{today}</div>", unsafe_allow_html=True)
    # 读取用户画像和资产优化结果
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    profile_path = os.path.join(results_dir, 'profile_default.json')
    portfolio_path = os.path.join(results_dir, 'portfolio_default.json')
    agent_path = os.path.join(results_dir, 'agent_default.json')
    news_path = os.path.join(results_dir, 'news_interpretation_default.json')
    # 加载数据
    profile = None
    portfolio = None
    agent = None
    news_list = None
    if os.path.exists(profile_path):
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)
    if os.path.exists(portfolio_path):
        with open(portfolio_path, 'r', encoding='utf-8') as f:
            portfolio = json.load(f)
    if os.path.exists(agent_path):
        with open(agent_path, 'r', encoding='utf-8') as f:
            agent = json.load(f)
    if os.path.exists(news_path):
        with open(news_path, 'r', encoding='utf-8') as f:
            news_list = json.load(f)
    # ========== 卡片样式 ========== #
    CARD_STYLE = """
    background: #fff;
    border-radius: 14px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    padding: 28px 24px 20px 24px;
    margin-bottom: 24px;
    min-height: 180px;
    transition: box-shadow .2s;
    """
    # ========== 布局 ========== #
    col_left, col_right = st.columns([1, 1])
    # --- 左侧 --- #
    with col_left:
        # 左上：今日热点新闻（蓝色卡片）
        st.markdown(f"""
        <div style='background:#e0f2fe;border-radius:14px 14px 14px 14px;box-shadow:0 4px 16px rgba(0,0,0,0.08);padding:28px 24px 20px 24px;margin-bottom:0;position:relative;'>
            <div style='display:flex;align-items:center;margin-bottom:10px;'>
                <span style='font-size:1.5rem;margin-right:10px;'>📰</span>
                <span style='font-size:1.15rem;font-weight:600;'>今日热点新闻</span>
            </div>
            <div style='color:#6B7280;font-size:0.95rem;margin-bottom:8px;'>如需查看详细解读请前往新闻解读页</div>
        """, unsafe_allow_html=True)
        if news_list and len(news_list) > 0:
            for item in news_list:
                news = item.get('news', '').strip()
                interp = item.get('interpretation', '').strip()
                url = item.get('url', '').strip()
                interp_html = f"<div style='color:#2563eb;font-size:0.98rem;font-weight:500;margin-bottom:2px;'><b>个性化解读：</b>{interp}</div>" if interp else ""
                url_html = f"<div style='margin-top:6px;'><a href='{url}' target='_blank' style='color:#2563eb;font-size:0.95rem;'>🔗 原文链接</a></div>" if url else ""
                st.markdown(f"""
                <div class='stCard' style='background:#f3f4f6;margin-bottom:14px;padding:16px 18px 12px 18px;border-radius:10px;'>
                    <div style='font-weight:600;font-size:1.05rem;margin-bottom:4px;color:#222;'>{news}</div>
                    {interp_html}
                    {url_html}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("暂无新闻摘要。")
        st.markdown("</div>", unsafe_allow_html=True)
        # 左下：智能顾问·专属动态建议（淡绿色卡片）
        st.markdown(f"""
        <div style='background:#dcfce7;border-radius:14px 14px 14px 14px;box-shadow:0 4px 16px rgba(0,0,0,0.08);padding:28px 24px 20px 24px;margin-top:0;margin-bottom:24px;min-height:180px;transition:box-shadow .2s;'>
            <div style='display:flex;align-items:center;margin-bottom:2px;'>
                <span style='font-size:1.5rem;margin-right:10px;'>🤖</span>
                <span style='font-size:1.15rem;font-weight:600;'>智能顾问·专属动态建议</span>
            </div>
            <div style='color:#6B7280;font-size:0.98rem;margin-bottom:10px;'>基于您的画像与最新市场动态智能生成</div>
        """, unsafe_allow_html=True)
        # 新增：大模型摘要优先展示
        summary = None
        if agent and 'daily_report' in agent and profile:
            import streamlit as st
            if 'advisor_summary' not in st.session_state:
                with st.spinner('AI智能顾问正在为您总结专属建议...'):
                    summary = summarize_advisor_suggestion(profile, agent['daily_report'])
                    st.session_state['advisor_summary'] = summary
            else:
                summary = st.session_state['advisor_summary']
        if summary:
            import json
            try:
                suggestions = json.loads(summary)
                if isinstance(suggestions, list) and all(isinstance(item, dict) and '建议' in item for item in suggestions):
                    st.markdown("<ol style='font-size:1.08rem;line-height:1.7;color:#222;margin:0 0 0 18px;'>" + ''.join([f"<li style='margin-bottom:6px;'>{item['建议']}</li>" for item in suggestions]) + "</ol>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color:#991b1b;'>AI建议解析失败，请稍后重试。</div>", unsafe_allow_html=True)
            except Exception:
                st.markdown("<div style='color:#991b1b;'>AI建议解析失败，请稍后重试。</div>", unsafe_allow_html=True)
        else:
            # 兜底：原有关键词筛选逻辑
            if agent and 'daily_report' in agent:
                try:
                    contents = extract_and_format_llm_contents(agent['daily_report'])
                    keywords = ['建议', '配置', '投资方案']
                    filtered = [c for c in contents if any(k in c for k in keywords)]
                    if filtered:
                        preview = '\n'.join(filtered[:2])
                        if len(preview) > 200:
                            preview = preview[:200] + '...'
                        st.markdown(preview)
                    else:
                        st.markdown("<span style='color:#6B7280;'>暂无个性化投资建议</span>", unsafe_allow_html=True)
                except Exception:
                    st.markdown("<span style='color:#6B7280;'>暂无个性化投资建议</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#6B7280;'>暂无个性化投资建议</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    # --- 右侧 --- #
    with col_right:
        # 右上：资产配置
        st.markdown(f"""
        <div style='background:#fff;border-radius:14px 14px 14px 14px;box-shadow:0 4px 16px rgba(0,0,0,0.08);padding:28px 24px 20px 24px;margin-bottom:24px;'>
            <div style='display:flex;align-items:center;margin-bottom:10px;'>
                <span style='font-size:1.5rem;margin-right:10px;'>💹</span>
                <span style='font-size:1.25rem;font-weight:600;'>资产配置</span>
            </div>
        """, unsafe_allow_html=True)
        asset_allocation = profile.get('asset_allocation', {}) if profile else {}
        if asset_allocation:
            df = {
                '资产': list(asset_allocation.keys()),
                '配置比例': [v*100 for v in asset_allocation.values()]
            }
            fig = px.pie(df, names='资产', values='配置比例', title='', color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无资产配置数据")
        st.markdown("</div>", unsafe_allow_html=True)
        # 新增：用户画像三句话总结（卡片）
        if profile:
            if 'profile_brief' not in st.session_state:
                with st.spinner('AI正在为您总结用户画像...'):
                    brief = summarize_user_profile_brief(profile)
                    st.session_state['profile_brief'] = brief
            else:
                brief = st.session_state['profile_brief']
            if brief:
                st.markdown(f"""
                <div style='background:#f3f4f6;border-radius:12px;padding:18px 18px 12px 18px;margin-bottom:10px;box-shadow:0 2px 8px rgba(59,130,246,0.06);font-size:1.08rem;line-height:1.9;color:#222;'>
                {brief.replace('\n', '<br>')}
                </div>
                """, unsafe_allow_html=True)
        # 右下：用户画像（可下拉展开，默认收起）
        with st.expander('👤 用户画像', expanded=False):
            st.markdown(f"""
            <div style='background:#fff;border-radius:14px 14px 14px 14px;box-shadow:0 4px 16px rgba(0,0,0,0.08);padding:28px 24px 20px 24px;margin-top:0;'>
            """, unsafe_allow_html=True)
            if profile:
                total_amt = profile.get('initial_investment', 0)
                target_amt = profile.get('target_amount', 0)
                try:
                    rate = (total_amt / target_amt) if target_amt else 0
                except:
                    rate = 0
                exp_return = portfolio['optimization'].get('expected_return', 0) if portfolio and 'optimization' in portfolio else 0
                exp_vol = portfolio['optimization'].get('expected_volatility', 0) if portfolio and 'optimization' in portfolio else profile.get('acceptable_volatility', 0)
                max_drawdown = portfolio['optimization'].get('max_drawdown', 0) if portfolio and 'optimization' in portfolio else profile.get('max_acceptable_loss', 0)
                # 六大指标2行3列排版
                st.markdown(f"""
                <div style='width:100%;display:flex;flex-direction:column;gap:0;margin-bottom:18px;'>
                  <div style='display:flex;gap:0;'>
                    <div style='flex:1;min-width:0;text-align:center;'>
                      <div style='color:#6B7280;font-size:0.98rem;'>初始投资</div>
                      <div style='font-size:1.35rem;font-weight:700;'>{total_amt:,.0f} 元</div>
                    </div>
                    <div style='flex:1;min-width:0;text-align:center;'>
                      <div style='color:#6B7280;font-size:0.98rem;'>目标金额</div>
                      <div style='font-size:1.35rem;font-weight:700;'>{target_amt:,.0f} 元</div>
                    </div>
                    <div style='flex:1;min-width:0;text-align:center;'>
                      <div style='color:#6B7280;font-size:0.98rem;'>投资进度</div>
                      <div style='font-size:1.35rem;font-weight:700;'>{rate*100:.1f}%</div>
                    </div>
                  </div>
                  <div style='display:flex;gap:0;margin-top:8px;'>
                    <div style='flex:1;min-width:0;text-align:center;'>
                      <div style='color:#6B7280;font-size:0.98rem;'>预期收益率</div>
                      <div style='font-size:1.35rem;font-weight:700;'>{exp_return*100:.2f}%</div>
                    </div>
                    <div style='flex:1;min-width:0;text-align:center;'>
                      <div style='color:#6B7280;font-size:0.98rem;'>预期波动率</div>
                      <div style='font-size:1.35rem;font-weight:700;'>{exp_vol*100:.2f}%</div>
                    </div>
                    <div style='flex:1;min-width:0;text-align:center;'>
                      <div style='color:#6B7280;font-size:0.98rem;'>最大回撤</div>
                      <div style='font-size:1.35rem;font-weight:700;'>{max_drawdown*100:.2f}%</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                # 其他信息表格加回
                st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
                info_map = [
                    ("投资目的", profile.get('investment_purpose', '')), 
                    ("流动性要求", profile.get('liquidity_requirement', '')), 
                    ("允许杠杆", '是' if profile.get('leverage_allowed', False) else '否'),
                    ("最大可接受回撤", f"{profile.get('max_acceptable_loss',0)*100:.1f}%"),
                    ("厌恶资产", '、'.join(profile.get('restricted_assets', [])) or '无'),
                    ("风险偏好", profile.get('risk_tolerance', ''))
                ]
                table_html = """
                <div style='color:#6B7280;font-size:1.05rem;margin-bottom:6px;'>其他信息</div>
                <table style='width:100%;border-collapse:collapse;'>
                """
                for k, v in info_map:
                    table_html += f"<tr style='border-bottom:1px solid #f3f4f6;'><td style='padding:6px 8px 6px 0;width:38%;color:#6B7280;'>{k}</td><td style='padding:6px 0 6px 8px;font-weight:600;color:#222;'>{v}</td></tr>"
                table_html += "</table>"
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.info("暂无用户画像数据")
            st.markdown("</div>", unsafe_allow_html=True)
    # 风险提示
    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#991b1b;font-size:1.05rem;text-align:center;padding:10px 0 0 0;'>⚠️ 投资有风险，决策需谨慎。市场有不确定性，建议结合自身风险承受能力理性决策。</div>", unsafe_allow_html=True)

# 新增：优化结果美观展示函数

def render_portfolio_optimization_result(opt_result, mcp_data=None):
    import streamlit as st
    import plotly.express as px
    import pandas as pd
    import numpy as np
    # 资产、权重
    assets = opt_result.get('assets', [])
    weights = opt_result.get('weights', [])
    # 修复：确保 weights 是 list
    if isinstance(weights, np.ndarray):
        weights = weights.tolist()
    exp_return = opt_result.get('expected_return', 0)
    exp_vol = opt_result.get('expected_volatility', 0)
    final_amt = opt_result.get('final_amount', 0)
    max_drawdown = opt_result.get('max_drawdown', 0)
    # MCP数据表 - 显示为note形式
    if mcp_data and isinstance(mcp_data, dict) and mcp_data:
        try:
            with st.expander("ℹ️ 预测数据来源", expanded=False):
                df = pd.DataFrame(mcp_data)
                for _, row in df.iterrows():
                    st.markdown(f"- {row['资产类别']}: [{row['预期收益率']}]({row['信息来源']})")
        except Exception:
            pass
    # 饼图
    if assets and weights and len(assets) == len(weights):
        fig = px.pie(
            names=assets,
            values=[w*100 for w in weights],
            title="资产配置比例（%）",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    # 指标卡片
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("预期年化收益率", exp_return*100, show_percent=True)
    with col2:
        render_metric_card("预期波动率", exp_vol*100, show_percent=True)
    with col3:
        render_metric_card("最大回撤", max_drawdown*100, show_percent=True)
    # 详细配置
    st.write("\n### 💡 具体配置建议")
    for asset, weight in zip(assets, weights):
        st.write(f"- {asset}: {weight*100:.1f}%")
    # 原始JSON可选折叠
    with st.expander("查看原始优化结果JSON", expanded=False):
        st.json(opt_result)

# 新增：大模型摘要函数
def summarize_advisor_suggestion(profile, daily_report, api_key=None, api_base=None, model="deepseek-chat", max_tokens=512):
    """调用大模型API对daily_report和profile生成精炼摘要，输出严格JSON格式"""
    prompt = f"""
你是专业的智能理财顾问。请根据以下【用户画像】和【对话内容】，为该用户生成最多5条最重要的投资建议。输出格式必须为严格的JSON数组，每条建议为一个对象，字段名为"建议"，不要输出任何多余内容、标题或说明，也不要输出代码块标记。

【用户画像】
{profile}

【对话内容】
{daily_report}

【输出格式示例】
[
  {{"建议": "建议一..."}},
  {{"建议": "建议二..."}}
]
"""
    if api_key is None or api_base is None:
        try:
            from config import OPENAI_API_KEY, OPENAI_API_BASE
            api_key = api_key or OPENAI_API_KEY
            api_base = api_base or OPENAI_API_BASE
        except Exception:
            return None
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.5
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        return content.strip()
    except Exception as e:
        import streamlit as st
        st.write(f"[DEBUG] 大模型摘要API异常: {e}")
        return None

# 新增：用户画像三句话总结（deepseek大模型）
def summarize_user_profile_brief(profile, api_key=None, api_base=None, model="deepseek-chat", max_tokens=256):
    """调用大模型API对用户画像生成三句话总结，分别为投资目标、财富状况、风险偏好，风格通俗友好。"""
    prompt = f"""
<task>用生活化语言总结用户的投资目标、财富状况和风险偏好</task>

<context>
请用通俗、生活化、非专业的语言，分别用一句话总结用户的投资目标、财富状况和风险偏好。每句话要像理财顾问和客户交流时的表达，不要直接罗列数据或字段，可以适当归纳和润色，让内容简明友好、易于理解。
【用户画像】
{profile}
【输出格式示例】
投资目标：希望8年后顺利退休，资产实现稳步增长。
财富状况：目前拥有较为充裕的可投资资产，整体财务状况乐观。
风险偏好：倾向于稳健中求进，愿意承担适度风险以追求财富增长。
</context>

<instructions>
1. 分析用户画像数据，提取关键信息：
   - 投资目标相关要素：如退休年限、资产增值需求等
   - 财富状况核心指标：可投资资产规模、负债情况等
   - 风险偏好表现：历史投资行为、风险承受问卷结果等
2. 用自然对话语言转述关键信息：
   - 避免专业术语，使用"存钱""过日子"等生活化表达
   - 保持理财顾问对客户说话的语气，如"您目前..."
   - 每类总结限一句话，不超过30字
3. 参照输出格式示例进行润色：
   - 投资目标：突出时间规划和期望效果
   - 财富状况：描述当前资金充裕程度
   - 风险偏好：说明风险承受态度和期望回报
4. 确保内容简明友好：
   - 用积极词汇如"稳步""乐观""适度"
   - 避免数字和金融术语
   - 保持语句流畅自然
</instructions>

<output_format>
输出必须严格按以下三行格式：
投资目标：[通俗总结语句]
财富状况：[通俗总结语句]
风险偏好：[通俗总结语句]
示例：
投资目标：打算10年后安心退休，让存款慢慢变多。
财富状况：手头闲钱不少，没什么债务压力。
风险偏好：想稳当赚钱，也能接受小波动。
</output_format>
"""
    if api_key is None or api_base is None:
        try:
            from config import OPENAI_API_KEY, OPENAI_API_BASE
            api_key = api_key or OPENAI_API_KEY
            api_base = api_base or OPENAI_API_BASE
        except Exception:
            return None
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.4
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        return content.strip()
    except Exception as e:
        import streamlit as st
        st.write(f"[DEBUG] 用户画像三句话API异常: {e}")
        return None

def get_dify_conversation_history(conversation_id):
    """通过 Dify API 拉取指定 conversation_id 的全部历史消息"""
    import requests
    from config import DIFY_API_KEY
    try:
        api_url = f"https://api.dify.ai/v1/messages?conversation_id={conversation_id}&user=abc-123"
        headers = {
            'Authorization': f'Bearer {DIFY_API_KEY}',
            'Content-Type': 'application/json',
        }
        resp = requests.get(api_url, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        # 解析消息列表，按 query/answer 轮次拼接
        messages = []
        for item in data.get('data', []):
            if item.get('query'):
                messages.append({'role': 'user', 'content': item['query']})
            if item.get('answer'):
                messages.append({'role': 'assistant', 'content': item['answer']})
        return messages
    except Exception as e:
        logger.error(f"拉取 Dify 对话历史失败: {e}")
        return None

if __name__ == "__main__":
    main()
