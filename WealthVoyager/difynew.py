import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import json
import matplotlib.pyplot as plt
import matplotlib
from typing import *
import re
import asyncio
import platform
import pandas as pd
import ast

# ========== 工具函数：去除 markdown 代码块 ==========
def strip_markdown_code_block(s):
    """去除 markdown 代码块包裹（如 ```json ... ```）"""
    s = s.strip()
    # 去掉开头的 ```json、```python、```text 或 ```
    s = re.sub(r"^```(?:json|python|text)?\\s*", "", s)
    # 去掉结尾的 ```
    s = re.sub(r"\\s*```$", "", s)
    return s.strip()

from baseagent02 import InvestmentAdvisor
from config import  OPENAI_API_KEY, OPENAI_API_BASE, DEFAULT_ASSETS, PRESET_COVARIANCE, COV_MATRIX
from infor import handle_send, build_base_config
from portfolioptima import extract_last_entry, portfolio_optimization, chat, llm_profile_extract
from firecrawl_client_as import FirecrawlClient
from config_firecrawl import Config
from behavior import get_behavior_metrics_by_type

# 在主流程顶部统一定义 PRESET_RETURNS
PRESET_RETURNS = {
    "A股": 0.0848, "债券": 0.0322, "REITs": 0.0843,
    "港股": 0.0839, "美股": 0.1038, "黄金": 0.0433,
    "大宗商品": 0.0318
}

# 健壮的数值字段格式化工具
def safe_num(val, ndigits=2, default=None):
    try:
        f = float(val)
        if ndigits == 0:
            return f"{f:,.0f}"
        else:
            return f"{f:,.{ndigits}f}"
    except Exception as e:
        return f"{val}（类型异常）"

def set_chinese_font():
    system = platform.system()
    if system == "Darwin":  # macOS
        font_list = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
    elif system == "Windows":
        font_list = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    else:  # Linux
        font_list = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
    # 检查系统字体库，优先用可用的
    available_fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
    for font in font_list:
        if font in available_fonts:
            matplotlib.pyplot.rcParams['font.sans-serif'] = [font]
            break
    else:
        matplotlib.pyplot.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.pyplot.rcParams['axes.unicode_minus'] = False

set_chinese_font()
# ----------------------------
# Streamlit 主界面
# ----------------------------
# 在初始化会话状态处添加历史记录存储
def main():
    st.title("📊 AI 投资助手")
    st.sidebar.header("💬 AI 交互")
    st.sidebar.markdown(
        """
        **您好！我是您的 GLAD 智能财富管理助理，很高兴为您服务！**  
        为了更好地帮助您规划投资目标，我们需要先了解一些基本信息，包括您的 **目标金额**、**投资年限** 以及 **初始资金**。  
        这些信息将帮助我们为您制定更精准的投资计划。  
        
        💡 请输入您的投资目标或问题。
        """
    )

    # 如果还没有 conversation_id，就初始化
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    if "dify_response" not in st.session_state:
        st.session_state.dify_response = ""

    # ========= 核心修改：回调函数 + text_input(key=...) + button(on_click=...) ============


    # 创建输入框，用户输入投资问题
    st.sidebar.text_input("请输入您的投资问题:", key="user_input_key")
    st.sidebar.text_input(
        "请输入您的资产类别和持仓比例（允许范围：A股, 债券, REITs, 港股, 美股, 黄金, 大宗商品；若留空则使用默认资产：股票, 债券, 房地产信托）:",
        key="current_allocation"
    )
    st.sidebar.button("发送", on_click=handle_send)

    st.sidebar.markdown(f"**Dify 回复:**  \n{st.session_state.dify_response}")
    
    # 初始化会话状态（添加在文件开头部分）
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # 在 handle_send 回调函数后添加历史记录显示
    with st.expander("🗨️ 对话历史", expanded=True):
        for entry in st.session_state.conversation_history:
            st.markdown(f"**You**: {entry['input']}")  
            st.markdown(f"**AI**: {entry['response']}")
            st.markdown("---")


    extracted_data = extract_last_entry(st.session_state.conversation_id)
    print("[DEBUG] 当前extracted_data:", extracted_data)
    required_keys = ["目标金额", "投资年限", "初始资金", "可接受的资产波动率"]
    missing_keys = [k for k in required_keys if k not in extracted_data]
    print(f"[DEBUG] 缺失字段: {missing_keys}")
    if not extracted_data or missing_keys:
        st.info("请先在左侧输入您的投资问题和相关选项，我们将为您生成个性化的投资方案。")
        st.stop()
    print("[DEBUG] extracted_data已满足所有必需字段，准备生成base_config...")

    # 只在首次满足条件时做一次 LLM 识别和 base_config 生成
    if not st.session_state.get('base_config_ready', False):
        history_text = "\n".join([entry["input"] + "\n" + entry["response"] for entry in st.session_state.conversation_history])
        profile = llm_profile_extract(history_text)
        print(f"[DEBUG] llm_profile_extract识别结果: {profile}")
        for k, v in profile.items():
            extracted_data[k] = v  # 用大模型结果覆盖
        base_config = build_base_config(extracted_data)
        print("[DEBUG] 生成的base_config:", base_config)
        st.session_state.base_config = base_config
        st.session_state.base_config_ready = True
    else:
        base_config = st.session_state.base_config
        print("[DEBUG] 复用缓存base_config:", base_config)
    st.session_state.base_config = base_config
    print("[DEBUG] 已写入st.session_state.base_config:", st.session_state.base_config)
    # === 加在 base_config 之前：根据用户选择覆盖资产范围 ===
    raw_prob = extracted_data.get("成功概率", "0%")  # 比如 "48.27%"
    # 去掉百分号并转为小数
    try:
        prob = float(raw_prob.strip('%')) / 100
    except ValueError:
        prob = 0  # 无法转换时默认 0

    if prob < 0.6:
        # 提示用户选择资产分析范围（只提示一次）
        st.markdown("#### 📌 请选择资产分析范围")
        asset_analysis_mode  = st.radio(
                "您希望如何进行资产分析和优化？",
                ["使用我输入的资产类别", "使用所有可选资产类别（A股, 债券, REITs, 港股, 美股, 黄金, 大宗商品）"],
                key="asset_analysis_mode_radio"
            )
        # 根据选择覆盖 extracted_data 中的资产
        if asset_analysis_mode  == "使用所有可选资产类别（A股, 债券, REITs, 港股, 美股, 黄金, 大宗商品）":
            # 覆盖 asset + 给一个均匀/随机配置比例
            full_assets = ["A股", "债券", "REITs", "港股", "美股", "黄金", "大宗商品"]
            extracted_data["资产类别"] = full_assets
            extracted_data["当前资产配置"] = [round(1/7, 3)] * 7  # 均匀分配；你也可以改成随机生成
        
        base_config = build_base_config(extracted_data)

        # 添加一个按钮，让用户确认开始优化
        if st.session_state.dify_response and "start_optimization" not in st.session_state:
            st.write("### 已获取到您的投资信息")
            st.write(f"- 目标金额: {safe_num(base_config.get('target_amount'), 0)} 元")
            st.write(f"- 投资年限: {safe_num(base_config.get('investment_years'), 0)} 年")
            st.write(f"- 初始投资: {safe_num(base_config.get('initial_investment'), 0)} 元")
            st.write(f"- 可接受波动率: {safe_num(base_config.get('acceptable_volatility'), 2)}%")
            
            if st.button("开始生成投资方案", key="start_optimization_button"):
                st.session_state.start_optimization = True
                # 设置一个标志，表示用户已确认开始优化
                st.rerun()  # 使用 st.rerun() 替代 st.experimental_rerun()

        # 修改 InvestmentAdvisor 调用部分
        # 确保有用户输入时且用户已确认开始优化时才进行优化
        if st.session_state.dify_response and st.session_state.get("start_optimization", False):      
            advisor = InvestmentAdvisor(base_config, OPENAI_API_KEY, OPENAI_API_BASE)
            
            # 调用 advisor 进行优化
            if "optimization_results" not in st.session_state:
                # 只在第一次运行时执行优化
                with st.spinner('正在生成投资方案，请稍候...'):
                    results = {
                        "A": advisor.run_optimization(["target_amount"], max_rounds=3),
                        "B": advisor.run_optimization(["investment_years"], max_rounds=3),
                        "C": advisor.run_optimization(["investment_years", "initial_investment"], max_rounds=3)
                    }
                    st.session_state.optimization_results = results
            
            # 从 session_state 获取结果
            results = st.session_state.optimization_results

            # 显示所有方案
            st.write("\n📊 **所有可选方案：**")
            
            st.write("\nA. 调整目标金额方案:")
            st.write(f"   - 目标金额: {safe_num(results['A'][0]['target_amount'], 0)} 元")
            st.write(f"   - 投资年限: {safe_num(results['A'][0]['investment_years'], 0)} 年")
            st.write(f"   - 初始投资: {safe_num(results['A'][0]['initial_investment'], 0)} 元")
            st.write(f"   - 成功概率: {safe_num(results['A'][1], 2)}%")
            st.write(f"   - 达标状态: {'✅ 已达标' if results['A'][1] >= 0.6 else '⚠️ 未达标'}")

            st.write("\nB. 调整投资年限方案:")
            st.write(f"   - 目标金额: {safe_num(results['B'][0]['target_amount'], 0)} 元")
            st.write(f"   - 投资年限: {safe_num(results['B'][0]['investment_years'], 0)} 年")
            st.write(f"   - 初始投资: {safe_num(results['B'][0]['initial_investment'], 0)} 元")
            st.write(f"   - 成功概率: {safe_num(results['B'][1], 2)}%")
            st.write(f"   - 达标状态: {'✅ 已达标' if results['B'][1] >= 0.6 else '⚠️ 未达标'}")

            st.write("\nC. 调整年限和初始投资方案:")
            st.write(f"   - 目标金额: {safe_num(results['C'][0]['target_amount'], 0)} 元")
            st.write(f"   - 投资年限: {safe_num(results['C'][0]['investment_years'], 0)} 年")
            st.write(f"   - 初始投资: {safe_num(results['C'][0]['initial_investment'], 0)} 元")
            st.write(f"   - 成功概率: {safe_num(results['C'][1], 2)}%")
            st.write(f"   - 达标状态: {'✅ 已达标' if results['C'][1] >= 0.6 else '⚠️ 未达标'}")

            # 用户选择方案
            selected_plan = st.radio(
                "请选择您偏好的投资方案：",
                ["A", "B", "C"],
                format_func=lambda x: {
                    "A": "方案 A：调整目标金额",
                    "B": "方案 B：调整投资年限",
                    "C": "方案 C：调整年限和初始投资"
                }[x],
                key="plan_selection"
            )

            # 确认按钮
            if st.button("确认选择", key="confirm_plan"):
                st.session_state.selected_config = results[selected_plan][0]
                st.session_state.selected_success_rate = results[selected_plan][1]
                st.session_state.plan_confirmed = True
                st.success(f"您已选择{selected_plan}方案！")
                
                # 显示选中的方案详情
                st.write("\n🎯 **已选方案详情：**")
                st.write(f"- 目标金额：{safe_num(st.session_state.selected_config['target_amount'], 0)} 元")
                st.write(f"- 投资年限：{safe_num(st.session_state.selected_config['investment_years'], 0)} 年")
                st.write(f"- 初始投资：{safe_num(st.session_state.selected_config['initial_investment'], 0)} 元")
                st.write(f"- 预期成功率：{safe_num(st.session_state.selected_success_rate, 2)}%")
                
                # 添加重新运行，确保状态更新
                st.rerun()
        # 继续后面的资产优化部分
        # 修改：只有在确认选择后才执行后续流程
        if "plan_confirmed" in st.session_state and st.session_state.plan_confirmed:
            st.write("\n## 📈 基于选定方案的资产配置优化")
            
            user_assets = base_config["assets"]
            user_allocation = base_config["current_allocation"]

            print(f"DEBUG - 使用 base_config 中的资产类别: {user_assets}")
            print(f"DEBUG - 使用 base_config 中的配置比例: {user_allocation}")

            # 使用选定配置中的参数继续后面的优化流程
            selected_config = st.session_state.selected_config
            print(f"DEBUG - 选定的配置: {selected_config}")
            x = selected_config['investment_years']
            print(f"DEBUG - 投资年限: {x}")

            # === 替换大模型function为MCP网页搜索 ===
            with st.spinner('正在通过MCP（firecrawl）获取资产未来收益率...'):
                try:
                    # 构造结构化 prompt
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
                    # firecrawl_deep_research_sync 查询
                    mcp_result = firecrawl_deep_research_sync(search_query)
                    print("DEBUG - MCP原始返回：", mcp_result)
                    st.write("DEBUG - MCP原始返回：", mcp_result)

                    # === 修正：先解析 MCP 返回的 JSON，再提取 text='[...]' 里的内容 ===
                    json_str = extract_json_array_from_mcp_result(mcp_result)
                    print("DEBUG - 尝试解析的 JSON 字符串：", json_str[:500])
                    st.write("DEBUG - 尝试解析的 JSON 字符串：", json_str)

                    # 新增：去掉所有 \n 和实际换行符
                    json_str_clean = json_str.replace('\\n', '').replace('\n', '').replace('\r', '')

                    # 先用 json.loads，失败再 ast.literal_eval
                    try:
                        asset_list = json.loads(json_str_clean)
                    except Exception as e1:
                        try:
                            asset_list = ast.literal_eval(json_str_clean)
                        except Exception as e2:
                            st.error(f"MCP返回内容解析失败: {e1} / {e2}")
                            st.write("原始内容：", json_str_clean)
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

                except Exception as e:
                    st.error(f"MCP返回内容解析失败: {e}")
                    mean_returns = [PRESET_RETURNS.get(asset, 0.05) for asset in user_assets]
                    sources = ["模拟数据"] * len(user_assets)

            # 前端可视化
            try:
                df = pd.DataFrame({
                    "资产类别": user_assets,
                    "预期收益率": [f"{r:.2%}" for r in mean_returns],
                    "信息来源": sources
                })
                st.write("📌 **预测数据及来源**")
                st.table(df)
            except Exception as e:
                st.write("收益率可视化失败", e)

            # 构建协方差矩阵
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
                weights          = optimization_result["weights"]
                exp_return       = optimization_result["expected_return"]
                exp_vol          = optimization_result["expected_volatility"]
                final_amt        = optimization_result["final_amount"]
                max_drawdown_est = optimization_result["max_drawdown"]
                
                # 显示优化结果
                st.write("\n### 🎯 最优资产配置建议")
                
                # 创建资产配置饼图
                fig, ax = plt.subplots(figsize=(10, 6))
                assets = user_assets if user_assets != DEFAULT_ASSETS else ["A股", "债券", "REITs"]
                plt.pie(weights, labels=assets, autopct='%1.1f%%')
                plt.title("资产配置比例")
                st.pyplot(fig)
                
                # 显示详细数据
                st.write("\n### 📊 投资组合详细信息")
                st.write(f"- 预期年化收益率: {safe_num(exp_return, 2)}%")
                st.write(f"- 预期波动率: {safe_num(exp_vol, 2)}%")
                st.write(f"- {safe_num(selected_config['investment_years'], 0)}年后预期金额: {safe_num(final_amt, 2)} 元")
                
                # 显示具体配置建议
                st.write("\n### 💡 具体配置建议")
                for asset, weight in zip(assets, weights):
                    st.write(f"- {asset}: {safe_num(weight, 1)}%")
            else:
                st.error("无法找到满足条件的投资组合，请调整投资参数或放宽限制条件。")






#分段操作
    else:
        # 继续后面的资产优化部分
        st.write("\n## 📈 基于选定方案的资产配置优化")
        base_config = build_base_config(extracted_data)

        user_assets = base_config["assets"]
        user_allocation = base_config["current_allocation"]
        x = base_config['investment_years']
        print(f"DEBUG - 使用 base_config 中的资产类别: {user_assets}")
        print(f"DEBUG - 使用 base_config 中的配置比例: {user_allocation}")


        # === 替换大模型function为MCP网页搜索 ===
        with st.spinner('正在通过MCP（firecrawl）获取资产未来收益率...'):
            try:
                # 构造结构化 prompt
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
                # firecrawl_deep_research_sync 查询
                mcp_result = firecrawl_deep_research_sync(search_query)
                print("DEBUG - MCP原始返回：", mcp_result)
                st.write("DEBUG - MCP原始返回：", mcp_result)

                # === 修正：先解析 MCP 返回的 JSON，再提取 text='[...]' 里的内容 ===
                json_str = extract_json_array_from_mcp_result(mcp_result)
                print("DEBUG - 尝试解析的 JSON 字符串：", json_str[:500])
                st.write("DEBUG - 尝试解析的 JSON 字符串：", json_str)

                # 新增：去掉所有 \n 和实际换行符
                json_str_clean = json_str.replace('\\n', '').replace('\n', '').replace('\r', '')

                # 先用 json.loads，失败再 ast.literal_eval
                try:
                    asset_list = json.loads(json_str_clean)
                except Exception as e1:
                    try:
                        asset_list = ast.literal_eval(json_str_clean)
                    except Exception as e2:
                        st.error(f"MCP返回内容解析失败: {e1} / {e2}")
                        st.write("原始内容：", json_str_clean)
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

            except Exception as e:
                st.error(f"MCP返回内容解析失败: {e}")
                mean_returns = [PRESET_RETURNS.get(asset, 0.05) for asset in user_assets]
                sources = ["模拟数据"] * len(user_assets)

        # 前端可视化
        try:
            df = pd.DataFrame({
                "资产类别": user_assets,
                "预期收益率": [f"{r:.2%}" for r in mean_returns],
                "信息来源": sources
            })
            st.write("📌 **预测数据及来源**")
            st.table(df)
        except Exception as e:
            st.write("收益率可视化失败", e)

        # 构建协方差矩阵
        if user_assets == DEFAULT_ASSETS:
            cov_matrix = COV_MATRIX
        else:
            covariance = PRESET_COVARIANCE
            cov_matrix = [[covariance[asset_i][asset_j] for asset_j in user_assets] for asset_i in user_assets]

        # 执行投资组合优化
        with st.spinner('正在优化投资组合...'):
            optimization_result = portfolio_optimization(mean_returns, cov_matrix, base_config)

        if optimization_result:
            weights          = optimization_result["weights"]
            exp_return       = optimization_result["expected_return"]
            exp_vol          = optimization_result["expected_volatility"]
            final_amt        = optimization_result["final_amount"]
            max_drawdown_est = optimization_result["max_drawdown"]
            
            # 显示优化结果
            st.write("\n### 🎯 最优资产配置建议")
            
            # 创建资产配置饼图
            fig, ax = plt.subplots(figsize=(10, 6))
            assets = user_assets if user_assets != DEFAULT_ASSETS else ["A股", "债券", "REITs"]
            plt.pie(weights, labels=assets, autopct='%1.1f%%')
            plt.title("资产配置比例")
            st.pyplot(fig)
            
            # 显示详细数据
            st.write("\n### 📊 投资组合详细信息")
            st.write(f"- 预期年化收益率: {safe_num(exp_return, 2)}%")
            st.write(f"- 预期波动率: {safe_num(exp_vol, 2)}%")
            st.write(f"- {safe_num(base_config['investment_years'], 0)}年后预期金额: {safe_num(final_amt, 2)} 元")
            
            # 显示具体配置建议
            st.write("\n### 💡 具体配置建议")
            for asset, weight in zip(assets, weights):
                st.write(f"- {asset}: {safe_num(weight, 1)}%")
        else:
            st.error("无法找到满足条件的投资组合，请调整投资参数或放宽限制条件。")


# 替换所有原本的 async_firecrawl_query 调用为如下同步调用：
def firecrawl_deep_research_sync(query: str, maxDepth=2, timeLimit=60, maxUrls=10):
    client = FirecrawlClient(Config())
    async def _run():
        await client.initialize()
        result = await client.debug_deep_research(query, maxDepth, timeLimit, maxUrls)
        await client.close()
        return result
    return asyncio.run(_run())

def extract_asset_returns_from_report(report_text):
    # 资产类别列表
    asset_names = ["A股", "债券", "REITs", "港股", "美股", "黄金", "大宗商品"]
    result = {}
    for asset in asset_names:
        # 匹配如"预期年化收益率有望维持在约8%~10%左右"或"年化收益率预期在约3%~5%左右"
        pattern = rf"{asset}.*?([0-9]+\.?[0-9]*)%[~～-]([0-9]+\.?[0-9]*)%"
        match = re.search(pattern, report_text, re.DOTALL)
        if match:
            low = float(match.group(1)) / 100
            high = float(match.group(2)) / 100
            result[asset] = (low, high)
        else:
            # 兜底：尝试只匹配一个百分数
            pattern2 = rf"{asset}.*?([0-9]+\.?[0-9]*)%"
            match2 = re.search(pattern2, report_text, re.DOTALL)
            if match2:
                val = float(match2.group(1)) / 100
                result[asset] = (val, val)
            else:
                result[asset] = None
    return result

# ========== 新增：行为指标查表赋值函数 ==========
def get_behavior_metrics_by_type(investment_purpose):
    mapping = {
        "retirement":      [0.8, 0.5, 0.8, 0.5, 0.4, 0.8, 0.3, 0.2, 0.6],
        "child_education": [0.7, 0.6, 0.5, 0.5, 0.5, 0.7, 0.4, 0.4, 0.5],
        "house_purchase":  [0.6, 0.7, 0.4, 0.5, 0.6, 0.6, 0.3, 0.3, 0.7],
        "wealth_growth":   [0.3, 0.8, 0.6, 0.5, 0.7, 0.3, 0.8, 0.7, 0.3],
    }
    keys = [
        "loss_aversion", "news_policy_sensitivity", "investment_experience", "real_time_emotion",
        "herding_tendency", "regret_aversion", "overconfidence", "illusion_of_control", "decision_delay"
    ]
    values = mapping.get(investment_purpose, mapping["wealth_growth"])
    values[3] = 0.5  # 实时情绪统一初始化为0.5
    return dict(zip(keys, values))

# ========== 新增：userconfig转profile唯一入口 ==========
def convert_userconfig_to_profile(userconfig):
    investment_purpose = userconfig.get("investment_purpose", "wealth_growth")
    asset_allocation = {k: v for k, v in zip(userconfig.get("assets", []), userconfig.get("current_allocation", []))}
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

# ========== 工具函数：从 firecrawl/mcp 返回内容中提取 JSON 数组 ==========
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

# ========== Streamlit多tab主界面 ==========
tab1, tab2 = st.tabs(["投资组合优化", "智能对话/Agent模拟"])

with tab1:
    main()  # 你的原有主流程

with tab2:
    st.header("智能对话/Agent模拟")
    if "base_config" not in st.session_state:
        st.info("请先在主流程输入投资参数，生成用户画像后再切换到本页。")
        st.stop()
    userconfig = st.session_state.base_config
    profile = convert_userconfig_to_profile(userconfig)
    st.write("当前用户画像（自动生成，行为指标不可编辑）：")
    st.json(profile)

    if st.button("开始Agent模拟", key="start_agent_sim_btn"):
        with st.spinner("正在运行完整Agent模拟流程..."):
            try:
                import asyncio
                from investment_dialogue.main import main_async
                # 直接调用main_async，传入profile
                logs, daily_report = asyncio.run(main_async(profile))
                # 分步展示每一步内容
                for entry in logs:
                    if entry["step"] == "daily_report":
                        st.success(f"【{entry['desc']}】")
                        st.write(entry["content"])
                    elif entry["step"].startswith("advisor_review") or entry["step"].startswith("intentions"):
                        st.info(f"【{entry['desc']}】")
                        st.write(entry["content"])
                    else:
                        st.write(f"【{entry['desc']}】")
                        st.write(entry["content"])
            except Exception as e:
                import traceback
                st.error(f"Agent模拟出错: {e}\n{traceback.format_exc()}")

#if __name__ == "__main__":
#    main()