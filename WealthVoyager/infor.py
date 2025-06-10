import requests
import os
import time  
import streamlit as st
import json
from datetime import datetime
from urllib3.exceptions import InsecureRequestWarning
from config import DIFY_API_URL, CONVERSATION_LOG_FILE, DIFY_API_KEY
import ast
from behavior import get_behavior_metrics_by_type


def build_base_config(extracted_data: dict) -> dict:
    """按照当前规范把表单 → base_config"""
    assets = extracted_data.get("assets", extracted_data.get("资产类别", ["A股", "债券", "REITs"]))
    if not assets or not isinstance(assets, list) or len(assets) == 0:
        assets = ["A股", "债券", "REITs"]
    allocation = extracted_data.get("current_allocation", extracted_data.get("当前资产配置", [0.4, 0.4, 0.2]))
    # 自动补全investor_type和behavior_metrics
    investment_purpose = extracted_data.get("investment_purpose", extracted_data.get("投资目的", "wealth_growth"))
    # investor_type直接用investment_purpose
    investor_type = investment_purpose
    # 行为指标查表
    behavior_metrics = get_behavior_metrics_by_type(investment_purpose)
    # 修正波动率单位，始终为0~1小数
    raw_vol = extracted_data.get("acceptable_volatility", extracted_data.get("可接受的资产波动率", 7.0))
    try:
        vol = float(raw_vol)
        if vol > 1:  # 用户输入百分数，自动转为小数
            vol = vol / 100
    except Exception:
        vol = 0.2  # 默认值
    base_config = {
        "investment_purpose": investment_purpose,
        "target_amount": extracted_data.get("target_amount", extracted_data.get("目标金额", 1000000)),
        "initial_investment": extracted_data.get("initial_investment", extracted_data.get("初始资金", 300000)),
        "investment_years": extracted_data.get("investment_years", extracted_data.get("投资年限", 10)),
        "acceptable_volatility": vol,
        "max_acceptable_loss": extracted_data.get("max_acceptable_loss", extracted_data.get("最大可接受回撤", 0.3)),
        "assets": assets,
        "current_allocation": allocation,
        "liquidity_requirement": extracted_data.get("liquidity_requirement", extracted_data.get("流动性要求", "medium")),
        "leverage_allowed": extracted_data.get("leverage_allowed", extracted_data.get("允许杠杆", False)),
        "restricted_assets": extracted_data.get("restricted_assets", extracted_data.get("厌恶资产", [])),
        "investor_type": investor_type,
        "risk_tolerance": extracted_data.get("risk_tolerance", extracted_data.get("风险偏好", "Medium")),
        "success_threshold": extracted_data.get("success_threshold", 0.6),
        "behavior_metrics": behavior_metrics
    }
    return base_config

#dify交互
def interact_with_dify(query, conversation_id=None, current_allocation=""):
    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    headers = {
        'Authorization': f'Bearer {DIFY_API_KEY}',
        'Content-Type': 'application/json',
    }

    data = {
        "inputs": {"current_allocation": current_allocation},
        "query": query,
        "response_mode": "blocking",
        "conversation_id": conversation_id or "",
        "user": "abc-123",
    }

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(DIFY_API_URL, headers=headers, json=data, verify=False, timeout=20)
            response.raise_for_status()
            response_data = response.json()
            response_text = response_data.get("answer", "No valid response")
            new_conversation_id = response_data.get("conversation_id", conversation_id)
            return response_text, new_conversation_id, response_data
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)  # 等待1秒后重试
                continue
            else:
                return f"请求失败（已重试{max_retries}次）: {str(e)}", conversation_id, {}
# 记录对话
# 修改 log_conversation 函数以支持历史记录加载
def log_conversation(input_text, response_text, conversation_id, response_data):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "conversation_id": conversation_id,
        "input": input_text,
        "response": response_text,
        "response_data": response_data
    }
    
    data = []
    if os.path.exists(CONVERSATION_LOG_FILE):
        # 新增：如果文件为空，直接用空列表
        if os.path.getsize(CONVERSATION_LOG_FILE) == 0:
            data = []
        else:
            with open(CONVERSATION_LOG_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
    
    data.append(log_entry)
    
    with open(CONVERSATION_LOG_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    

# 处理用户输入
def handle_send():
    user_input = st.session_state.user_input_key
    # 获取用户自定义资产配置，去掉首尾空格
    current_allocation_input = st.session_state.get("current_allocation", "").strip()
    if not user_input:
        return

    # 调用 Dify 时传入 current_allocation
    response_text, new_conversation_id, response_data = interact_with_dify(
        user_input, st.session_state.conversation_id, current_allocation_input
    )
    st.session_state.conversation_id = new_conversation_id
    st.session_state.dify_response = response_text  # 存储 Dify 回复

    # 记录对话
    log_conversation(user_input, response_text, new_conversation_id, response_data)

    # 清空输入框
    st.session_state.user_input_key = ""

    # 在获取响应后添加历史记录
    st.session_state.conversation_history.append({
            "input": user_input,
            "response": st.session_state.dify_response,
            "timestamp": datetime.now().isoformat()  })