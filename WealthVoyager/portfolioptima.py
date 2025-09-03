# 风险相关字段推断表
RISK_MAP = {
    "低风险承受能力（保守型投资者）": {"max_acceptable_loss": 0.10, "liquidity_requirement": "high", "leverage_allowed": False},
    "低于平均水平的风险承受能力": {"max_acceptable_loss": 0.15, "liquidity_requirement": "medium", "leverage_allowed": False},
    "中等风险承受能力": {"max_acceptable_loss": 0.20, "liquidity_requirement": "medium", "leverage_allowed": False},
    "高于平均水平的风险承受能力": {"max_acceptable_loss": 0.30, "liquidity_requirement": "medium", "leverage_allowed": True},
    "高风险承受能力（激进型投资者）": {"max_acceptable_loss": 0.50, "liquidity_requirement": "low", "leverage_allowed": True},
}

import numpy as np
from scipy.optimize import minimize
import json
import re
import os
from config import CONVERSATION_LOG_FILE, ALLOWED_ASSETS, ASSET_MAPPING, BASE_URL, API_KEY, OPENAI_API_BASE, OPENAI_API_KEY
import openai 
from openai import OpenAI
from openai.types.chat.chat_completion import Choice
import time
import requests

#收集用户资金等信息
def get_all_user_inputs(conversation_id):
    """从conversation_log.json中获取该id下所有用户输入文本，拼成一段大文本"""
    if not os.path.exists(CONVERSATION_LOG_FILE):
        return ""
    with open(CONVERSATION_LOG_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)
    user_texts = [entry["input"] for entry in logs if entry.get("conversation_id") == conversation_id]
    return "\n".join(user_texts)

def llm_profile_extract(history_text):
    """用deepseek大模型分析历史文本，提取投资目的、风险偏好、流动性需求、杠杆意愿等"""
    url = f"{OPENAI_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""
以下是用户与AI的全部对话历史，请你帮我识别用户的投资目的（只能选：退休养老、子女教育、购房置业、财富增长），风险偏好（激进/平衡/保守）。请用json格式输出所有能识别的字段。
对话历史：
{history_text}
"""
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个金融理财专家，善于从用户对话中提取投资画像。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=20)
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        print(f"[DEBUG] deepseek原始返回content: {content}")
        import json as _json
        try:
            profile = _json.loads(content)
        except Exception:
            import re
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                profile = _json.loads(match.group(0))
            else:
                profile = {}
        print(f"[DEBUG] deepseek解析后profile: {profile}")
        return profile
    except Exception as e:
        print(f"llm_profile_extract调用失败: {e}")
        return {}

def infer_risk_type_by_volatility(acceptable_volatility):
    if acceptable_volatility <= 8:
        return "低风险承受能力（保守型投资者）"
    elif acceptable_volatility <= 12:
        return "低于平均水平的风险承受能力"
    elif acceptable_volatility <= 18:
        return "中等风险承受能力"
    elif acceptable_volatility <= 25:
        return "高于平均水平的风险承受能力"
    else:
        return "高风险承受能力（激进型投资者）"

def infer_risk_fields(extracted):
    risk_type = extracted.get("风险评估类型") or extracted.get("用户类型")
    if not risk_type and "可接受的资产波动率" in extracted:
        try:
            vol = float(extracted["可接受的资产波动率"])
            risk_type = infer_risk_type_by_volatility(vol)
            extracted["风险评估类型"] = risk_type
        except Exception as e:
            print("DEBUG - 可接受的资产波动率类型转换失败:", extracted["可接受的资产波动率"], e)
    if risk_type and risk_type in RISK_MAP:
        for k, v in RISK_MAP[risk_type].items():
            extracted[k] = v
    return extracted

def fill_defaults(extracted):
    """对缺失字段填默认值"""
    defaults = {
        "投资目的": "财富增值",
        "最大可接受回撤": 0.3,
        "流动性要求": "medium",
        "允许杠杆": False,
        # ...其它字段默认值...
    }
    for k, v in defaults.items():
        if k not in extracted or extracted[k] is None:
            extracted[k] = v
    return extracted

def parse_structured_block(text):
    import re
    match = re.search(r'\{([^\{\}]*)\}', text)
    if not match:
        return {}
    block = match.group(1)
    result = {}
    for line in block.split('\n'):
        if '：' in line:
            k, v = line.split('：', 1)
            result[k.strip()] = v.strip()
    return result

def extract_last_entry(conversation_id):
    # 1) 统一选用可写路径：优先环境变量，其次 config.CONVERSATION_LOG_FILE，最后兜底 /tmp
    log_path = os.environ.get("CONVERSATION_LOG_FILE", CONVERSATION_LOG_FILE)
    if not os.path.isabs(log_path):
        # 云端仓库目录只读，若相对路径则回落到 /tmp
        log_path = "/tmp/conversation_log.json"

    print(f"DEBUG - 开始提取对话信息，conversation_id: {conversation_id}, log_path: {log_path}")

    input_data = []
    # 2) 读取日志文件（容错：不存在、为空、JSON损坏都不抛异常）
    if os.path.exists(log_path):
        try:
            if os.path.getsize(log_path) > 0:
                print(f"DEBUG - 找到对话日志文件: {log_path}")
                with open(log_path, "r", encoding="utf-8") as file:
                    input_data = json.load(file)
            else:
                print(f"DEBUG - 对话日志文件为空: {log_path}")
        except Exception as e:
            print(f"DEBUG - 读取日志失败({log_path}): {e}")

    # 3) 根据 conversation_id 过滤；若拿不到，则使用最后一条合法记录兜底
    user_entries = [e for e in input_data if e.get("conversation_id") == conversation_id] if input_data else []
    if not user_entries and input_data:
        print("DEBUG - 未匹配到该会话ID，使用最后一条记录兜底")
        user_entries = [input_data[-1]]

    answer = ""
    if user_entries:
        last_entry = user_entries[-1]
        print(f"DEBUG - 获取最后一条记录的时间戳: {last_entry.get('timestamp', '未知')}")
        response_data = last_entry.get("response_data") or {}
        answer = response_data.get("answer", "") or ""
        print(f"DEBUG - 获取到的回答长度: {len(answer)} 字符")
        print(f"DEBUG - 回答前100字符: {answer[:100]}...")

    # 4) 如果文件读不到答案，直接从 Streamlit 会话内存兜底
    if not answer:
        try:
            import streamlit as st  # 延迟导入，避免非 Streamlit 场景报错
            answer = (st.session_state.get("dify_response") or "").strip()
            if answer:
                print("DEBUG - 从 st.session_state.dify_response 获取到答案内容")
        except Exception as e:
            print(f"DEBUG - 无法从 st.session_state 读取 dify_response: {e}")

    if not answer:
        print("DEBUG - 未找到可解析的答案内容")
        return {}

    # 5) 解析逻辑（与原始版本一致，略做健壮性处理）
    data = {}
    print("DEBUG - 开始提取关键信息")
    for line in answer.split("\n"):
        if "：" not in line:
            continue
        parts = line.split("：")
        if len(parts) < 2:
            continue
        key = parts[0].strip()
        value_str = parts[1].strip()
        print(f"DEBUG - 提取到键值对: {key} = {value_str}")
        if key == "目标金额":
            try:
                data["目标金额"] = int(value_str)
                print(f"DEBUG - 成功提取目标金额: {data['目标金额']}")
            except ValueError:
                data["目标金额"] = 0
                print(f"DEBUG - 目标金额转换失败，使用默认值: 0")
        elif key == "投资年限":
            try:
                data["投资年限"] = int(value_str)
                print(f"DEBUG - 成功提取投资年限: {data['投资年限']}")
            except ValueError:
                data["投资年限"] = 0
                print(f"DEBUG - 投资年限转换失败，使用默认值: 0")
        elif key == "初始资金":
            try:
                data["初始资金"] = int(value_str)
                print(f"DEBUG - 成功提取初始资金: {data['初始资金']}")
            except ValueError:
                data["初始资金"] = 0
                print(f"DEBUG - 初始资金转换失败，使用默认值: 0")
        elif key == "可接受的资产波动率":
            try:
                data["可接受的资产波动率"] = float(value_str)
                print(f"DEBUG - 成功提取可接受的资产波动率: {data['可接受的资产波动率']}")
            except ValueError:
                data["可接受的资产波动率"] = 0.0
                print(f"DEBUG - 可接受的资产波动率转换失败，使用默认值: 0.0")
        elif key == "成功概率":
            try:
                value_float = float(value_str.replace("%", "")) / 100 if "%" in value_str else float(value_str)
                data["成功概率"] = value_float
                print(f"DEBUG - 成功提取成功概率: {data['成功概率']}")
            except ValueError:
                data["成功概率"] = 0.0
                print(f"DEBUG - 成功概率转换失败，使用默认值: 0.0")
        elif key == "投资目的":
            data["投资目的"] = value_str
        elif key == "最大可接受回撤":
            try:
                data["最大可接受回撤"] = float(value_str.replace("%", ""))/100 if "%" in value_str else float(value_str)
            except ValueError:
                data["最大可接受回撤"] = 0.3
        elif key == "流动性要求":
            data["流动性要求"] = value_str.lower()
        elif key == "允许杠杆":
            data["允许杠杆"] = value_str.lower() in ["是", "yes", "true"]
        elif key == "厌恶资产":
            data["厌恶资产"] = [x.strip() for x in value_str.split(",") if x.strip()]
        elif key == "用户类型":
            data["用户类型"] = value_str
        elif key == "风险偏好":
            data["风险偏好"] = value_str.capitalize()
        elif key == "风险评估类型":
            data["风险评估类型"] = value_str

    print("DEBUG - 开始提取资产类别")
    assets_match = re.search(r"资产类别\s*[:：]\s*\[(.*?)\]", answer, re.DOTALL) or \
                   re.search(r"资产类别\s*=\s*\[(.*?)\]", answer, re.DOTALL)
    if assets_match:
        assets_text = assets_match.group(1)
        print(f"DEBUG - 提取到资产类别文本: {assets_text}")
        assets = [asset.strip() for asset in assets_text.split("\n") if asset.strip()]
        data["资产类别"] = assets
        print(f"DEBUG - 成功提取资产类别: {assets}")
    else:
        data["资产类别"] = []
        print("DEBUG - 未找到资产类别信息")
        asset_lines = re.findall(r"→\s*([^→\n]+?)\s+(\d+[\.%]?\d*%?)", answer)
        if asset_lines:
            print(f"DEBUG - 从箭头格式提取到资产: {asset_lines}")
            data["资产类别"] = [line[0].strip() for line in asset_lines]
            print(f"DEBUG - 从箭头格式成功提取资产类别: {data['资产类别']}")

    print("DEBUG - 开始提取资产配置")
    allocation_match = re.search(r"当前资产配置\s*[:：]\s*\[(.*?)\]", answer, re.DOTALL) or \
                       re.search(r"当前资产配置\s*=\s*\[(.*?)\]", answer, re.DOTALL)
    if allocation_match:
        allocation_text = allocation_match.group(1)
        print(f"DEBUG - 提取到资产配置文本: {allocation_text}")
        allocation = [float(a.strip()) for a in allocation_text.split("\n") if a.strip()]
        data["当前资产配置"] = allocation
        print(f"DEBUG - 成功提取资产配置: {allocation}")
    else:
        data["当前资产配置"] = []
        print("DEBUG - 未找到资产配置信息")
        asset_lines = re.findall(r"→\s*([^→\n]+?)\s+(\d+[\.%]?\d*%?)", answer)
        if asset_lines:
            print(f"DEBUG - 从箭头格式提取百分比: {asset_lines}")
            allocations = []
            for line in asset_lines:
                percent_str = line[1].strip()
                try:
                    percent = float(percent_str.replace('%', '')) / 100
                except ValueError:
                    percent = 0.0
                allocations.append(percent)
            if allocations:
                data["当前资产配置"] = allocations
                print(f"DEBUG - 从箭头格式成功提取资产配置: {allocations}")

    defaults = {
        "success_threshold": 0.6,
        "投资目的": "财富增值",
        "最大可接受回撤": 0.3,
        "流动性要求": "medium",
        "允许杠杆": False,
        "厌恶资产": [],
        "用户类型": "平衡型",
        "风险偏好": "Medium"
    }
    for k, v in defaults.items():
        if k not in data:
            data[k] = v
            print(f"DEBUG - 设置默认值 {k}: {v}")

    risk_type = data.get("风险评估类型")
    if risk_type and risk_type in RISK_MAP:
        for k, v in RISK_MAP[risk_type].items():
            data[k] = v
            print(f"DEBUG - 风险类型补全 {k}: {v}")

    print("DEBUG - 最终提取的数据:")
    for k, v in data.items():
        print(f"DEBUG - {k}: {v}")

    return data

# def portfolio_optimization(mean_returns, cov_matrix, target_amount, years, initial_funds, max_volatility):
#     num_assets = len(mean_returns)

#     def objective(weights):
#         return np.dot(weights.T, np.dot(cov_matrix, weights))

#     def portfolio_return(weights):
#         return np.dot(weights, mean_returns)

#     def future_value(weights):
#         return initial_funds * (1 + portfolio_return(weights)) ** years

#     constraints = [
#         {"type": "eq", "fun": lambda w: np.sum(w) - 1},
#         {"type": "ineq", "fun": lambda w: max_volatility - np.sqrt(objective(w))},
#         {"type": "ineq", "fun": lambda w: future_value(w) - target_amount}
#     ]

#     bounds = [(0, 1) for _ in range(num_assets)]
#     initial_weights = np.array([1 / num_assets] * num_assets)

#     result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints, method="SLSQP")
    
#     if result.success:
#         optimized_weights = np.round(result.x, 2)
#         optimized_weights /= np.sum(optimized_weights)
#         expected_return = round(portfolio_return(optimized_weights), 4)
#         expected_volatility = round(np.sqrt(objective(optimized_weights)), 4)
#         final_amount = round(initial_funds * (1 + expected_return) ** years, 2)
#         return optimized_weights, expected_return, expected_volatility, final_amount
#     else:
#         return None, None, None, None
def portfolio_optimization(mean_returns, cov_matrix, config):
    """
    改进后的投资组合优化函数，使用完整的用户画像配置
    
    参数:
        mean_returns: 各资产预期收益率数组
        cov_matrix: 资产协方差矩阵
        config: 包含用户画像配置的字典，包括:
            - target_amount: 目标金额
            - investment_years: 投资年限
            - initial_investment: 初始资金
            - acceptable_volatility: 可接受波动率
            - max_acceptable_loss: 最大可接受回撤
            - leverage_allowed: 是否允许杠杆
            - liquidity_requirement: 流动性要求(low/medium/high)
            - restricted_assets: 厌恶资产列表
    """
    num_assets = len(mean_returns)
    
    # 从配置中提取参数
    target_amount = config.get("target_amount")
    years = config.get("investment_years")
    initial_funds = config.get("initial_investment")
    max_volatility = config.get("acceptable_volatility")
    max_loss = config.get("max_acceptable_loss", 0.3)
    leverage = config.get("leverage_allowed", False)
    liquidity = config.get("liquidity_requirement", "medium")
    restricted = config.get("restricted_assets", [])

    def objective(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    def portfolio_return(weights):
        return np.dot(weights, mean_returns)

    def future_value(weights):
        return initial_funds * (1 + portfolio_return(weights)) ** years

    # 基础约束条件
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # 权重总和为1
        {"type": "ineq", "fun": lambda w: max_volatility - np.sqrt(objective(w))},  # 波动率限制
        {"type": "ineq", "fun": lambda w: future_value(w) - target_amount}  # 目标金额要求
    ]

    # 添加最大回撤约束
    constraints.append({
        "type": "ineq", 
        "fun": lambda w: max_loss - np.sqrt(objective(w)) * 2.33  # 假设正态分布，99%置信度
    })

    # 设置资产权重边界
    bounds = [(0, 1) for _ in range(num_assets)]  # 默认不允许杠杆
    
    # 如果允许杠杆，调整边界
    if leverage:
        bounds = [(-1, 1) for _ in range(num_assets)]  # 允许做空和杠杆
        
    # 考虑流动性要求
    if liquidity == "high":
        # 对流动性差的资产设置上限
        illiquid_assets = ["REITs", "大宗商品", "私募股权"]
        for i, asset in enumerate(config.get("assets", [])):
            if asset in illiquid_assets:
                bounds[i] = (0, 0.2)  # 流动性差资产不超过20%
                
    # 排除厌恶资产
    for i, asset in enumerate(config.get("assets", [])):
        if asset in restricted:
            bounds[i] = (0, 0)  # 完全排除该资产

    initial_weights = np.array([1 / num_assets] * num_assets)
    
    # 优化
    result = minimize(
        objective, 
        initial_weights, 
        bounds=bounds, 
        constraints=constraints, 
        method="SLSQP"
    )
    
    if result.success:
        optimized_weights = np.round(result.x, 2)
        optimized_weights /= np.sum(optimized_weights)  # 归一化
        expected_return = round(portfolio_return(optimized_weights), 4)
        expected_volatility = round(np.sqrt(objective(optimized_weights)), 4)
        final_amount = round(initial_funds * (1 + expected_return) ** years, 2)
        
        return {
            "weights": optimized_weights,
            "expected_return": expected_return,
            "expected_volatility": expected_volatility,
            "final_amount": final_amount,
            "max_drawdown": round(np.sqrt(objective(optimized_weights)) * 2.33, 4)  # 估算最大回撤
        }
    else:
        return None 

#与大模型交互预测收益率
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)


def chat(messages) -> Choice:
    # 添加重试次数限制和指数退避策略
    max_retries = 5
    retry_count = 0
    base_wait_time = 1  # 初始等待时间（秒）
    
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model="moonshot-v1-128k",
                messages=messages,
                temperature=0.3,
                tools=[{
                    "type": "builtin_function",
                    "function": {"name": "$web_search"}
                }]
            )
            return completion.choices[0]
        except openai.RateLimitError as e:
            wait_time = base_wait_time * (2 ** retry_count)  # 指数退避
            print(f"速率限制超出。等待 {wait_time} 秒后重试... (尝试 {retry_count+1}/{max_retries})")
            time.sleep(wait_time)
            retry_count += 1
        except Exception as e:
            print(f"API 调用出错: {str(e)}")
            # 创建一个模拟的响应对象
            mock_response = type('obj', (object,), {
                'message': type('obj', (object,), {
                    'content': '{"A股": {"rate": "8.5%", "source": "模拟数据"}, "债券": {"rate": "3.2%", "source": "模拟数据"}, "REITs": {"rate": "8.4%", "source": "模拟数据"}, "港股": {"rate": "8.4%", "source": "模拟数据"}, "美股": {"rate": "10.4%", "source": "模拟数据"}, "黄金": {"rate": "4.3%", "source": "模拟数据"}, "大宗商品": {"rate": "3.2%", "source": "模拟数据"}}'
                }),
                'finish_reason': 'stop'  # 添加 finish_reason 属性
            })
            return mock_response
    
    # 如果所有重试都失败，返回备用数据
    print("达到最大重试次数，使用备用数据")
    mock_response = type('obj', (object,), {
        'message': type('obj', (object,), {
            'content': '{"A股": {"rate": "8.5%", "source": "模拟数据"}, "债券": {"rate": "3.2%", "source": "模拟数据"}, "REITs": {"rate": "8.4%", "source": "模拟数据"}, "港股": {"rate": "8.4%", "source": "模拟数据"}, "美股": {"rate": "10.4%", "source": "模拟数据"}, "黄金": {"rate": "4.3%", "source": "模拟数据"}, "大宗商品": {"rate": "3.2%", "source": "模拟数据"}}'
        }),
        'finish_reason': 'stop'  # 添加 finish_reason 属性
    })

    return mock_response
