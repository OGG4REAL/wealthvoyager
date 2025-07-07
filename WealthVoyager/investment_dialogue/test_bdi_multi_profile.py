# -*- coding: utf-8 -*-
import sys
import copy
import json
from WealthVoyager.investment_dialogue.main import get_default_profile
from WealthVoyager.investment_dialogue.investor_agent import InvestorAgent
from WealthVoyager.investment_dialogue.advisor_agent import AdvisorAgent
from WealthVoyager.investment_dialogue.dialogue_manager import InvestmentDialogue
from camel.models import ModelFactory
from camel.types import ModelPlatformType

# ========== Tee类定义，实现双写 ==========
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, open('all_output.txt', 'w', encoding='utf-8'))

# 1. 定义四类 behavior_metrics（按表格）
behavior_metrics_dict = {
    "retirement": {
        "loss_aversion": 0.8,
        "news_policy_sensitivity": 0.5,
        "investment_experience": 0.8,
        "real_time_emotion": 0.5,
        "herding_tendency": 0.4,
        "regret_aversion": 0.8,
        "overconfidence": 0.3,
        "illusion_of_control": 0.2,
        "decision_delay": 0.6
    },
    "child_education": {
        "loss_aversion": 0.7,
        "news_policy_sensitivity": 0.6,
        "investment_experience": 0.5,
        "real_time_emotion": 0.5,
        "herding_tendency": 0.5,
        "regret_aversion": 0.7,
        "overconfidence": 0.4,
        "illusion_of_control": 0.4,
        "decision_delay": 0.5
    },
    "house_purchase": {
        "loss_aversion": 0.6,
        "news_policy_sensitivity": 0.7,
        "investment_experience": 0.4,
        "real_time_emotion": 0.5,
        "herding_tendency": 0.6,
        "regret_aversion": 0.6,
        "overconfidence": 0.3,
        "illusion_of_control": 0.3,
        "decision_delay": 0.7
    },
    "wealth_growth": {
        "loss_aversion": 0.3,
        "news_policy_sensitivity": 0.8,
        "investment_experience": 0.6,
        "real_time_emotion": 0.5,
        "herding_tendency": 0.7,
        "regret_aversion": 0.3,
        "overconfidence": 0.8,
        "illusion_of_control": 0.7,
        "decision_delay": 0.3
    }
}

# 2. 构造四个 profile
profile_map = {}
for k in behavior_metrics_dict:
    profile = get_default_profile()
    profile["behavior_metrics"] = behavior_metrics_dict[k]
    profile["investor_type"] = k
    if k == "retirement":
        profile["investment_purpose"] = "retirement"
    elif k == "child_education":
        profile["investment_purpose"] = "child_education"
    elif k == "house_purchase":
        profile["investment_purpose"] = "house_purchase"
    elif k == "wealth_growth":
        profile["investment_purpose"] = "wealth_growth"
    profile_map[k] = copy.deepcopy(profile)

# 3. 定义统一的新闻内容（可根据需要修改）
news = (
    "突发：中美关税战升级，全球市场剧烈震荡（2025年6月12日）\n\n"
    "美国政府突然宣布对价值2000亿美元的中国进口商品加征25%关税，中国随即反制，对600亿美元美国产品加征关税。"
    "受此影响，全球主要股市大幅下挫，道琼斯指数盘中暴跌800点，A股三大指数全线下跌超3%，科技、消费、出口板块领跌。"
    "人民币兑美元汇率跌破7.5关口，创三年新低，资本外流压力加剧。"
    "避险资产大幅走强，现货黄金价格突破每盎司2500美元，十年期美债收益率跌至1.2%。"
    "市场恐慌情绪蔓延，VIX恐慌指数飙升至35，投资者纷纷抛售风险资产转向现金和黄金。"
    "多家跨国企业发布盈利预警，全球供应链中断风险加剧。"
    "分析人士警告，关税战升级将严重冲击全球经济复苏前景，企业投资和消费信心大幅下滑，短期内市场波动或将持续加剧。"
)
sentiment = -0.7  # 反映突发利空新闻的消极情绪

# 4. 初始化模型（请替换为你的API KEY）
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

# 5. 循环四类 profile，跑流程
for k, profile in profile_map.items():
    print(f"\n{'='*30}\n【投资者类型】：{k}\n{'='*30}")
    dialogue = InvestmentDialogue(
        task_prompt="模拟投资者对市场新闻的反应与顾问博弈。",
        investor_profile=profile,
        model=model
    )
    (
        news_msg,
        advisor_interpretation_msg,
        investor_reaction_msg,
        advisor_final_advice_msg,
        investor_final_response_msg
    ) = dialogue.process_market_news(news, sentiment)
    print("\n【新闻原文】\n", news_msg.content)
    print("\n【顾问解读】\n", advisor_interpretation_msg.content)
    print("\n【投资者反应】\n", investor_reaction_msg.content)
    print("\n【顾问建议】\n", advisor_final_advice_msg.content)
    print("\n【投资者最终回应】\n", investor_final_response_msg.content)
    print("\n当前对话状态：\n", json.dumps(dialogue.get_dialogue_state(), ensure_ascii=False, indent=2)) 

    # === 新增：输出每日投资简报，并单独写入 daily_reports.txt ===
    advisor = dialogue.advisor
    market_summary_for_report = news
    advisor_interpretation_for_report = advisor_interpretation_msg.content
    intentions_for_report = investor_final_response_msg.content
    advisor_review_for_report = advisor_final_advice_msg.content
    profile_for_report = profile
    daily_report = advisor.summarize_dialogue(
        market_summary=market_summary_for_report,
        advisor_interpretation=advisor_interpretation_for_report,
        intentions=intentions_for_report,
        advisor_review=advisor_review_for_report,
        profile=profile_for_report
    )
    print("\n【今日投资简报】\n", daily_report)
    # === 简报单独写入 daily_reports.txt ===
    with open('daily_reports.txt', 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*30}\n【投资者类型】：{k}\n{'='*30}\n")
        f.write(daily_report)
        f.write('\n') 