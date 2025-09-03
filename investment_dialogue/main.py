from typing import List, Tuple
import json
import os
from .dialogue_manager import InvestmentDialogue
from camel.models import ModelFactory
from .config import MODEL_CONFIG
from camel.types import ModelPlatformType, ModelType
from .firecrawl_client import FirecrawlClient
from .config_firecrawl import Config as FirecrawlConfig
import asyncio
import re
from .investor_agent import InvestorAgent
from .advisor_agent import AdvisorAgent

def load_market_news() -> List[Tuple[str, float]]:
    """加载市场新闻及其情绪值
    
    Returns:
        List[Tuple[str, float]]: 新闻内容和情绪值的列表
    """
    # 预设的新闻和情绪值库
    return [
        ("美联储宣布，在此次会议上不会提高利率，维持联邦基金利率的目标区间在5%至5.25%。这是自去年三月以来首次暂停加息。", 0.5),
        ("最新中国制造业采购经理指数（PMI）数据不仅再次未达到预期，还出现了类似悬崖式的下跌，跌破了扩张/收缩的门槛几个百分点。", -0.8),
        ("受美联储持续加息和全球避险情绪上升的影响，美元指数大幅上涨，人民币汇率连续多日暴跌。", -0.7),
        ("全球贸易收缩加剧，航运业遭受前所未有的打击，行业领导者中国远洋海运集团有限公司也面临破产风险。", -0.9),
    ]

# 新增：异步获取实时新闻
async def get_realtime_news(n: int = 5) -> List[Tuple[str, float]]:
    """
    调用MCP服务，获取今日最热点金融市场新闻
    """
    client = FirecrawlClient(FirecrawlConfig())
    await client.initialize()
    prompt = f"""
请列出今天最热点的{n}条金融市场新闻。每条新闻只需包含\"新闻标题\"和\"一句话摘要\"，每条不超过100字。输出格式要求：每条新闻一行，内容和情绪值用|||分隔。例如：新闻标题：xxx；摘要：xxx|||情绪值
"""
    try:
        response = await client.process_query(prompt)
        print("MCP原始返回内容：", response)
        news_list = []
        for line in response.splitlines():
            if '|||' in line:
                parts = line.split('|||')
                content = parts[0].strip()[:100]
                try:
                    sentiment = float(parts[1].strip())
                except Exception:
                    sentiment = 0.0
                news_list.append((content, sentiment))
        if not news_list:
            return load_market_news()[:n]
        return news_list[:n]
    except Exception as e:
        print(f"获取实时新闻失败，使用静态新闻。原因: {e}")
        return load_market_news()[:n]
    finally:
        await client.close()

# 新增：让Agent选择新闻
def agent_select_news(news_list: List[Tuple[str, float]], investor_agent) -> int:
    """
    让投资者Agent从新闻列表中选择最感兴趣的一条，返回其索引
    """
    news_text = "\n".join([f"{i+1}. {news[0][:30]}" for i, news in enumerate(news_list)])  # 只取前30字
    prompt = (
        f"以下是今天的金融市场新闻标题：\n{news_text}\n"
        "请你结合你的投资者画像来选择你最感兴趣的一条新闻，并说明理由。"
        "只需回复你选择的新闻编号（如：2），以及简要理由。"
    )
    response = investor_agent.step(prompt)
    match = re.search(r"(\d+)", response.msg.content)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(news_list):
            return idx
    return 0

async def process_single_news_async(news: str, sentiment: float, investor_profile: dict = None) -> None:
    """异步处理单条新闻直到达成对话目标"""
    task_prompt = """
    这是一个模拟投资者对市场信息的反应过程。系统将展示投资者如何基于个人特征
    （包括风险偏好、损失厌恶、信息敏感度等）对市场信息做出反应和决策。

    对话目标：
    1. 展示投资者如何根据自己的性格特征和投资偏好解读市场信息
    2. 展示投资者的情绪变化如何影响其市场判断和投资决策
    3. 展示投资者如何权衡短期目标和风险控制要求
    4. 展示投资者在面对市场波动时的心理调整过程
    
    对话应该持续到：
    1. 投资者完成对市场信息的充分消化和分析
    2. 形成了明确的投资决策或调整意向
    3. 展现出符合其个人特征的情绪和行为反应
    """
    print("\n=== 调试信息 ===")
    print("正在创建模型实例，参数如下：")
    try:
        DEEPSEEK_API_KEY = "sk-1ddd28a0c2184057a7aff10f6d5640f4"
        model = ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type="deepseek-chat",
            url="https://api.deepseek.com/v1",
            api_key=DEEPSEEK_API_KEY,
            model_config_dict={
                "temperature": 0.7,
                "max_tokens": 2000
            }
        )
        print("模型创建成功！")
    except Exception as e:
        print(f"模型创建出错: {str(e)}")
        print(f"错误类型: {type(e)}")
        raise
    print("\n=== 创建对话管理器 ===")
    try:
        dialogue = InvestmentDialogue(task_prompt=task_prompt, model=model, investor_profile=investor_profile)
        print("对话管理器创建成功！")
    except Exception as e:
        print(f"对话管理器创建出错: {str(e)}")
        print(f"错误类型: {type(e)}")
        raise
    print("\n=== 当前市场信息 ===")
    print(news)
    while True:
        print("\n=== 对话过程 ===")
        try:
            (
                news_msg,
                advisor_interpretation_msg,
                investor_reaction_msg,
                advisor_final_advice_msg,
                investor_final_response_msg
            ) = dialogue.process_market_news(news, sentiment)
            print("新闻处理完成！")
            print("\n=== 第1轮：顾问解读新闻 ===")
            print(f"新闻原文:\n{news_msg.content}")
            print(f"\n投资顾问解读:\n{advisor_interpretation_msg.content}")
            print("\n=== 第2轮：投资者反应 ===")
            print(f"投资者反应 (基于新闻和顾问解读):\n{investor_reaction_msg.content}")
            print("\n=== 第3轮：顾问分析与建议 ===")
            print(f"投资顾问分析与建议:\n{advisor_final_advice_msg.content}")
            print("\n=== 第4轮：投资者最终回应 ===")
            print(f"投资者最终回应 (基于顾问建议):\n{investor_final_response_msg.content}")
            print("\n=== 当前状态 ===")
            state = dialogue.get_dialogue_state()
            print(json.dumps(state, ensure_ascii=False, indent=2))
            user_input = input("\n对话目标是否已达成？(y/n): ")
            if user_input.lower() == 'y':
                break
        except Exception as e:
            print(f"\n处理新闻时发生错误：{e}")
            import traceback
            traceback.print_exc()
            print("请重新选择要处理的新闻")
            break

def get_default_profile():
    return {
        "investment_purpose": "retirement",
        "target_amount": 1000000,
        "initial_investment": 100000,
        "investment_years": 15,
        "volatility_tolerance": 0.15,
        "max_acceptable_loss": 0.30,
        "asset_allocation": {
            "A_shares": 0.50,
            "US_stocks": 0.30,
            "bonds": 0.10,
            "gold": 0.10
        },
        "liquidity_requirement": "low",
        "leverage_allowed": False,
        "restricted_assets": [],
        "investor_type": "retirement",  # retirement, child_education, house_purchase, wealth_growth
        "risk_tolerance": "Medium",
        "behavior_metrics": {
            "loss_aversion": 0.8,
            "news_policy_sensitivity": 0.5,
            "investment_experience": 0.8,
            "real_time_emotion": 0.5,
            "herding_tendency": 0.4,
            "regret_aversion": 0.8,
            "overconfidence": 0.3,
            "illusion_of_control": 0.2,
            "decision_delay": 0.6
        }
    }

async def get_market_summary_via_deep_research(raw_content: str, model) -> str:
    """
    用大模型对firecrawl_deep_research抓取到的原始内容做摘要总结
    """
    prompt = (
        f"以下是今日全球金融市场的深度聚合原始内容：\n{raw_content}\n"
        "请你用中文对上述内容进行总结，输出一份300字以内的市场综述，涵盖主要事件、影响和市场情绪。"
    )
    print("\n=== 传给LLM做市场综述摘要的prompt如下（前500字） ===")
    print(prompt[:500])
    print(f"\n=== 传给LLM的内容总长度: {len(prompt)} ===")
    messages = [{"role": "user", "content": prompt}]
    response = model.run(messages) if hasattr(model, 'run') else model.chat(messages)
    print("\n=== LLM返回的市场综述摘要如下（前500字） ===")
    print(str(response)[:500])
    return str(response).strip()

async def analyze_sentiment_for_user(summary: str, user_profile: dict, model) -> float:
    """
    用大模型根据市场综述和用户profile分析情绪值，返回[-1,1]
    """
    prompt = (
        f"以下是今日市场综述：\n{summary}\n"
        f"投资者画像：\n{json.dumps(user_profile, ensure_ascii=False, indent=2)}\n"
        "请你判断这则市场综述对该投资者的情绪影响（-1极度悲观，0中性，1极度乐观），只回复一个数字。"
    )
    print("\n=== 传给LLM的prompt如下（前500字） ===")
    print(prompt[:500])
    print(f"\n=== 传给LLM的内容总长度: {len(prompt)} ===")
    # 用大模型推理
    messages = [{"role": "user", "content": prompt}]
    response = model.run(messages) if hasattr(model, 'run') else model.chat(messages)
    # 智能提取content字段
    content = None
    if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
        content = response.choices[0].message.content
    elif isinstance(response, dict) and 'choices' in response and response['choices'] and 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
        content = response['choices'][0]['message']['content']
    else:
        content = str(response)
    print("\n=== LLM返回的情绪分析原始内容如下（前500字） ===")
    print(str(content)[:500])
    import re
    match = re.search(r"-?\d+\.?\d*", str(content))
    if match:
        return float(match.group(0))
    return 0.0

async def main_async(profile=None):
    from .investor_agent import InvestorAgent
    from .advisor_agent import AdvisorAgent
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import json

    logs = []  # 新增：收集每一步输出

    DEEPSEEK_API_KEY = "sk-1ddd28a0c2184057a7aff10f6d5640f4"
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEEPSEEK,
        model_type="deepseek-chat",
        url="https://api.deepseek.com/v1",
        api_key=DEEPSEEK_API_KEY,
        model_config_dict={
            "temperature": 0.7,
            "max_tokens": 2000
        }
    )
    if profile is None:
        profile = get_default_profile()
    investor = InvestorAgent(model=model, user_profile=profile)
    advisor = AdvisorAgent(model=model)
    profile_str = json.dumps(profile, ensure_ascii=False, indent=2)

    # 1. 用firecrawl_client深度调研抓取原始新闻和市场综述
    client = FirecrawlClient(FirecrawlConfig())
    await client.initialize()
    result = await client.debug_deep_research(
        query="""请完成以下两步任务，并以严格的JSON对象格式输出：

1. 列出今天全球金融市场的主要新闻，每条新闻包含标题、正文、来源网站和原文链接，但输出格式为一段连续的文本，每条新闻用"【新闻标题】"开头，用"来源：xxx"结尾，多条新闻之间用换行分隔。

2. 基于上述新闻内容，对今天的全球金融市场热点新闻进行深度调研和总结，输出一份简明扼要的市场综述（300字以内），涵盖主要事件、影响和市场情绪。

最终输出格式为：
{
  "raw_content": "【新闻标题1】新闻正文1...来源：xxx\n【新闻标题2】新闻正文2...来源：xxx",
  "summary": "市场综述内容"
}""",
        maxDepth=1,
        timeLimit=90,
        maxUrls=10
    )
    await client.close()

    # 2. 提取原始content并解析JSON
    raw_content = None
    summary = None
    if hasattr(result, 'content'):
        raw_content = result.content
    elif isinstance(result, dict) and 'content' in result:
        raw_content = result['content']
    elif isinstance(result, dict) and 'results' in result and result['results']:
        raw_content = result['results'][0].get('content', '')
    else:
        raw_content = str(result)
    
    # 尝试解析JSON格式的返回内容
    try:
        # 如果返回的是 TextContent 对象，提取其 text 字段
        if hasattr(raw_content, '__iter__') and len(raw_content) > 0:
            if hasattr(raw_content[0], 'text'):
                content_text = raw_content[0].text
            else:
                content_text = str(raw_content)
        else:
            content_text = str(raw_content)
        
        # 尝试解析为 JSON
        parsed_data = json.loads(content_text)
        raw_content = parsed_data.get("raw_content", "")
        summary = parsed_data.get("summary", "")
        
        # 如果解析失败，使用原始内容作为 raw_content
        if not raw_content:
            raw_content = content_text
            summary = "解析失败，使用原始内容"
            
    except json.JSONDecodeError:
        # 如果解析失败，使用原始内容
        raw_content = content_text
        summary = "解析失败，使用原始内容"
    
    # 限制长度
    raw_content = raw_content[:3000]
    summary = summary[:1500]
    
    # 保存到 logs，确保格式与 page_market_news 兼容
    logs.append({"step": "raw_content", "desc": "MCP原始市场内容", "content": raw_content})
    logs.append({"step": "summary", "desc": "市场综述摘要", "content": summary})

    # 3. 用大模型分析情绪值（直接使用 firecrawl 返回的 summary）
    sentiment = await analyze_sentiment_for_user(summary, profile, model)
    logs.append({"step": "sentiment", "desc": "个性化情绪值", "content": sentiment})

    # 4. 用summary作为新闻，sentiment作为情绪值，进入分步BDI+博弈流程
    news = summary
    logs.append({"step": "news", "desc": "今日市场综述", "content": news})

    # 顾问解读新闻
    advisor_prompt = (
        f"你是一名资深投资顾问，以下是客户的投资者画像：\n{profile_str}\n\n"
        f"请你用专业视角解读以下市场新闻，分析其对该投资者的潜在影响，指出风险与机会，并给出简要建议：\n{news}\n"
        f"请用中文输出。"
    )
    advisor_interpretation = advisor.step(advisor_prompt)
    logs.append({"step": "advisor_interpretation", "desc": "顾问解读", "content": advisor_interpretation.msg.content})

    # 投资者分步BDI
    # Step 1: Beliefs
    prompt_beliefs = (
        f"你是一名投资者，以下是你的个人画像：\n{profile_str}\n\n"
        f"以下是投资顾问对新闻的解读：\n{advisor_interpretation.msg.content}\n\n"
        f"请你只根据上述信息，更新你对市场的信念（Beliefs），包括但不限于：市场观点、风险认知、政策影响、市场趋势等。\n"
        f"不要输出目标（Desires）或意图（Intentions），只需详细描述你的信念。"
    )
    beliefs_response = investor.step(prompt_beliefs)
    logs.append({"step": "beliefs", "desc": "投资者信念（Beliefs）", "content": beliefs_response.msg.content})

    # Step 2: Desires
    prompt_desires = (
        f"你是一名投资者，以下是你的个人画像：\n{profile_str}\n\n"
        f"以下是投资顾问对新闻的解读：\n{advisor_interpretation.msg.content}\n\n"
        f"这是你最新的市场信念（Beliefs）：\n{beliefs_response.msg.content}\n\n"
        f"请你只根据上述信念和顾问解读，重新评估和描述你的投资目标和愿望（Desires），包括短期目标、风险偏好、持仓目标等。\n"
        f"不要输出信念（Beliefs）或意图（Intentions），只需详细描述你的愿望。"
    )
    desires_response = investor.step(prompt_desires)
    logs.append({"step": "desires", "desc": "投资者愿望（Desires）", "content": desires_response.msg.content})

    # Step 3: Intentions
    prompt_intentions = (
        f"你是一名投资者，以下是你的个人画像：\n{profile_str}\n\n"
        f"以下是投资顾问对新闻的解读：\n{advisor_interpretation.msg.content}\n\n"
        f"这是你最新的市场信念（Beliefs）：\n{beliefs_response.msg.content}\n\n"
        f"这是你最新的投资愿望（Desires）：\n{desires_response.msg.content}\n\n"
        f"请你只根据上述信念、愿望和顾问解读，形成你当前的投资意图（Intentions），包括组合调整、信息需求、风险管理等。\n"
        f"不要输出信念（Beliefs）或愿望（Desires），只需详细描述你的意图。"
    )
    intentions_response = investor.step(prompt_intentions)
    logs.append({"step": "intentions", "desc": "投资者意图（Intentions）", "content": intentions_response.msg.content})

    # 顾问评价与多轮博弈
    max_rounds = 10  # 最大轮数保护
    round_count = 0
    last_advisor_review = None
    last_intentions_response = None
    while True:
        round_count += 1
        advisor_review_prompt = (
            f"你是一名资深投资顾问，以下是客户的投资者画像：\n{profile_str}\n\n"
            f"以下是你对新闻的解读：\n{advisor_interpretation.msg.content}\n\n"
            f"以下是投资者的最终投资意图（Intentions）：\n{intentions_response.msg.content}\n\n"
            f"请你评价投资者的决策是否合理。如果你有异议或建议，请详细说明理由并给出具体建议；如果你完全同意，请直接回复'同意，无需修改'。"
        )
        advisor_review = advisor.step(advisor_review_prompt)
        last_advisor_review = advisor_review.msg.content
        logs.append({"step": f"advisor_review_{round_count}", "desc": f"顾问评价-第{round_count}轮", "content": advisor_review.msg.content})
        if "同意" in advisor_review.msg.content and "无需修改" in advisor_review.msg.content:
            logs.append({"step": "end", "desc": "模拟结束：双方达成共识", "content": "模拟结束：双方达成共识"})
            break
        elif round_count >= max_rounds:
            logs.append({"step": "end", "desc": f"模拟自动终止：已达到最大轮数{max_rounds}", "content": "模拟自动终止：已达到最大轮数"})
            break
        else:
            investor_reconsider_prompt = (
                f"你是一名投资者，以下是你的个人画像：\n{profile_str}\n\n"
                f"以下是投资顾问对你的投资意图的评价和建议：\n{advisor_review.msg.content}\n\n"
                f"请你根据顾问的建议，决定是否要修改你的投资意图（Intentions）。如果你同意建议，请给出修改后的新意图；如果你坚持原意，请说明理由。"
            )
            intentions_response = investor.step(investor_reconsider_prompt)
            last_intentions_response = intentions_response.msg.content
            logs.append({"step": f"intentions_{round_count}", "desc": f"投资者修正意图-第{round_count}轮", "content": intentions_response.msg.content})
    # === 模拟结束后生成每日投资简报 ===
    # 收集所需信息
    market_summary_for_report = summary
    advisor_interpretation_for_report = advisor_interpretation.msg.content
    intentions_for_report = last_intentions_response if last_intentions_response else intentions_response.msg.content
    advisor_review_for_report = last_advisor_review if last_advisor_review else ""
    # 生成简报
    daily_report = advisor.summarize_dialogue(
        market_summary=market_summary_for_report,
        advisor_interpretation=advisor_interpretation_for_report,
        intentions=intentions_for_report,
        advisor_review=advisor_review_for_report,
        profile=profile
    )
    logs.append({"step": "daily_report", "desc": "每日投资简报", "content": daily_report})
    return logs, daily_report

if __name__ == "__main__":
    asyncio.run(main_async()) 