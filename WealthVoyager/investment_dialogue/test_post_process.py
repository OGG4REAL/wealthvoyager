import copy
import json
from WealthVoyager.investment_dialogue.main import get_default_profile
from WealthVoyager.investment_dialogue.investor_agent import InvestorAgent
from WealthVoyager.investment_dialogue.advisor_agent import AdvisorAgent
from WealthVoyager.investment_dialogue.dialogue_manager import InvestmentDialogue
from camel.models import ModelFactory
from camel.types import ModelPlatformType

def post_process_dialogue(state, original_behavior_metrics):
    """后处理：判断是否需要调仓或profile更新，并模拟修改，检测行为指标变化"""
    print("\n=== 后处理分析 ===")
    bdi = state["investor"]["bdi_state"]
    profile = state["investor"]["profile"]
    need_rebalance = False
    need_profile_update = False

    # 检查调仓意图
    pa = bdi["intentions"].get("portfolio_adjustment")
    if pa and isinstance(pa, str) and pa.strip():
        print("检测到投资者有调仓意图：", pa)
        need_rebalance = True
        # 示例：模拟调仓（这里只是打印，实际可按pa内容调整profile["asset_allocation"]）
        print("模拟调仓：asset_allocation将被调整（实际逻辑可自定义）")

    # 检查风险偏好等profile字段变化
    rt = bdi["desires"].get("risk_tolerance")
    if rt and rt != profile.get("risk_tolerance"):
        print(f"检测到风险偏好变化：原={profile.get('risk_tolerance')} 新={rt}")
        need_profile_update = True
        # 示例：自动更新profile
        profile["risk_tolerance"] = rt
        print("已自动更新profile中的risk_tolerance")

    # 检查behavior_metrics变化
    bm_new = profile.get("behavior_metrics", {})
    bm_old = original_behavior_metrics
    bm_changed = []
    for k in bm_new:
        if k in bm_old and bm_new[k] != bm_old[k]:
            bm_changed.append((k, bm_old[k], bm_new[k]))
    if bm_changed:
        print("检测到behavior_metrics有变化：")
        for k, old, new in bm_changed:
            print(f"  {k}: {old} → {new}")
        need_profile_update = True
    else:
        print("behavior_metrics无变化。")

    if not need_rebalance and not need_profile_update:
        print("未检测到需要调仓或profile更新的信号。")
    print("当前profile：", json.dumps(profile, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    # 1. 构造单一profile
    profile = get_default_profile()
    # 保存原始behavior_metrics
    original_behavior_metrics = copy.deepcopy(profile["behavior_metrics"])
    # 2. 初始化模型
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
    # 3. 新闻和情绪
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
    sentiment = -0.7
    # 4. 跑对话流程
    dialogue = InvestmentDialogue(
        task_prompt="模拟投资者对市场新闻的反应与顾问博弈。",
        investor_profile=copy.deepcopy(profile),
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
    # 5. 后处理分析
    state = dialogue.get_dialogue_state()
    post_process_dialogue(state, original_behavior_metrics)
