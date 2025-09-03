from typing import Dict, List, Optional, Tuple
from camel.societies import RolePlaying
from camel.messages import BaseMessage
from camel.agents import ChatAgent

from .investor_agent import InvestorAgent
from .advisor_agent import AdvisorAgent

class InvestmentDialogue(RolePlaying):
    """投资对话管理器,继承自RolePlaying"""
    
    def __init__(
        self,
        task_prompt: str,
        investor_profile: Optional[Dict] = None,
        model = None,
    ) -> None:
        # 初始化投资者和顾问代理
        self.investor = InvestorAgent(model=model, user_profile=investor_profile)
        self.advisor = AdvisorAgent(model=model)
        
        super().__init__(
            assistant_role_name="Investment Advisor",
            user_role_name="Investor",
            task_prompt=task_prompt,
            with_task_specify=True,
            with_task_planner=True,
            assistant_agent_kwargs=dict(model=model),
            user_agent_kwargs=dict(model=model),
            task_specify_agent_kwargs=dict(model=model),
            task_planner_agent_kwargs=dict(model=model)
        )
        
    def process_market_news(self, news: str, sentiment: float) -> Tuple[BaseMessage, BaseMessage, BaseMessage, BaseMessage, BaseMessage]:
        """处理市场新闻，按照 Advisor -> Investor -> Advisor -> Investor 的顺序

        Args:
            news (str): 新闻内容
            sentiment (float): 新闻情绪值(-1到1)

        Returns:
            Tuple[BaseMessage, BaseMessage, BaseMessage, BaseMessage, BaseMessage]:
                新闻消息, 顾问解读, 投资者反应, 顾问建议, 投资者最终回应
        """
        print("\n--- 开始处理新闻 --- ")
        # 1. 创建新闻消息
        news_message = BaseMessage.make_user_message(
            role_name="MarketNews",  # 英文
            content=news
        )
        print(f"新闻消息创建: {news_message.content[:50]}...")

        # 2. 顾问解读新闻
        print("顾问正在解读新闻...")
        # 获取投资者画像的字符串表示（新版结构，英文key）
        profile = self.investor.profile
        bm = profile["behavior_metrics"]
        bm_str = "\n".join([
            f"- {k}: {v}" for k, v in bm.items()
        ])
        investor_profile_str = f"""Current Investor Profile:
Investment Purpose: {profile['investment_purpose']}
Target Amount: {profile['target_amount']}
Initial Investment: {profile['initial_investment']}
Investment Years: {profile['investment_years']}
Volatility Tolerance: {profile['volatility_tolerance']}
Max Acceptable Loss: {profile['max_acceptable_loss']}
Asset Allocation: {profile['asset_allocation']}
Liquidity Requirement: {profile['liquidity_requirement']}
Leverage Allowed: {profile['leverage_allowed']}
Restricted Assets: {profile['restricted_assets']}
Investor Type: {profile['investor_type']}
Risk Tolerance: {profile['risk_tolerance']}
\nBehavior Metrics:\n{bm_str}\n"""

        # 构建包含投资者画像的提示
        advisor_interpretation_prompt = BaseMessage.make_user_message(
            role_name="System",  # 英文
            content=(
                f"{investor_profile_str}\n"
                "请用中文回答。\n"
                "Please analyze the following market news for its potential impact on this investor's investment strategy:\n\n"
                f"News:\n{news}"
            )
        )
        advisor_interpretation = self.advisor.step(advisor_interpretation_prompt)
        print(f"Advisor interpretation completed: {advisor_interpretation.msg.content[:50]}...")

        # 3. 投资者根据新闻和顾问解读做出反应
        print("Investor is reacting to the news and advisor's interpretation...")
        # 在投资者反应前更新其情绪
        self.investor.update_emotion(sentiment)
        print(f"Investor's emotion updated, current value: {self.investor.profile['behavior_metrics']['real_time_emotion']:.3f}")

        # 准备给投资者的提示，包含新闻和顾问的解读
        investor_prompt_content = f"请用中文回答。\nNews:\n{news}\n\nAdvisor's interpretation:\n{advisor_interpretation.msg.content}\n\nPlease analyze the current situation based on your investor profile and BDI state, and explain your initial thoughts and feelings."
        investor_prompt = BaseMessage.make_user_message(
            role_name="System",  # 英文
            content=investor_prompt_content
        )
        investor_reaction = self.investor.step(investor_prompt)
        print(f"Investor reaction completed: {investor_reaction.msg.content[:50]}...")

        # 4. 顾问分析投资者状态（基于其反应）并提供建议
        print("Advisor is analyzing investor state and providing advice...")
        # 注意：这里的分析目前仅基于投资者的画像和BDI状态，可能需要未来增强以考虑具体反应内容
        advice_analysis = self.advisor.analyze_client_state(
            self.investor.profile,
            self.investor.bdi_state
        )

        # 准备给顾问的提示，包含投资者的反应和状态分析
        advisor_advice_prompt_content = f"请用中文回答。\nThis is the investor's reaction:\n{investor_reaction.msg.content}\n\nThis is my analysis of the client's current state:\n1. Emotion state: {advice_analysis['emotion_state']}\n2. Risk state: {advice_analysis['risk_state']}\nAreas of focus: {', '.join(advice_analysis['focus_points'])}\n\nPlease provide specific, actionable investment advice based on all the above information and my professional judgment."
        advisor_advice_prompt = BaseMessage.make_user_message(
             role_name="System", 
             content=advisor_advice_prompt_content
        )
        advisor_final_advice = self.advisor.step(advisor_advice_prompt)
        print(f"Advisor advice generated: {advisor_final_advice.msg.content[:50]}...")

        # 5. 投资者对最终建议做出回应
        print("Investor is responding to the final advice...")
        # 直接将顾问的建议消息传递给投资者
        investor_final_response = self.investor.step(advisor_final_advice.msg)
        print(f"Investor final response completed: {investor_final_response.msg.content[:50]}...")
        print("--- News processing ended ---")

        # 返回所有中间消息以便展示
        return (
            news_message,                      # 原始新闻
            advisor_interpretation.msg,        # 顾问对新闻的解读
            investor_reaction.msg,             # 投资者对新闻+解读的反应
            advisor_final_advice.msg,          # 顾问基于投资者反应的最终建议
            investor_final_response.msg        # 投资者对最终建议的回应
        )
        
    def get_dialogue_state(self) -> Dict:
        """获取当前对话状态
        
        Returns:
            Dict: 包含当前对话状态的字典
        """
        # 获取最后一条顾问建议，如果存在的话
        last_advice_content = None
        context_tuple = self.advisor.memory.get_context()
        if context_tuple and context_tuple[0]:
            last_msg = context_tuple[0][-1]
            if isinstance(last_msg, dict):
                last_advice_content = last_msg.get('content', '')
            else:
                last_advice_content = getattr(last_msg, 'content', '')
            
        return {
            "investor": {
                "profile": self.investor.profile,
                "bdi_state": self.investor.bdi_state
            },
            "advisor": {
                "last_advice": last_advice_content
            }
        } 