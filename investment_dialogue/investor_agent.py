from typing import Dict, Optional, Union, List
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import RoleType

class InvestorAgent(ChatAgent):
    """投资者代理类,继承自ChatAgent"""
    
    def __init__(
        self,
        model = None,
        user_profile: Dict = None,
    ) -> None:
        # 新版用户画像，英文key，扁平结构
        self.profile = user_profile or {
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

        # BDI状态，兼容新结构
        self.bdi_state = {
            "beliefs": {
                "market_view": "neutral",
                "risk_perception": "medium",
                "policy_impact": None,
                "market_trend": None
            },
            "desires": {
                "target_amount": self.profile["target_amount"],
                "risk_tolerance": self.profile["risk_tolerance"],
                "asset_allocation": self.profile["asset_allocation"].copy()
            },
            "intentions": {
                "portfolio_adjustment": None,
                "information_needs": [],
                "risk_management": None
            }
        }

        # 新版系统消息，英文key和简明解释
        bm = self.profile["behavior_metrics"]
        bm_str = "\n".join([
            f"- {k}: {v}" for k, v in bm.items()
        ])
        system_message = BaseMessage(
            role_name="Investor",
            role_type=RoleType.USER,
            meta_dict=None,
            content=(
                f"You are an investor, with the following characteristics:\n\n"
                f"Investment Purpose: {self.profile['investment_purpose']}\n"
                f"Target Amount: {self.profile['target_amount']}\n"
                f"Initial Investment: {self.profile['initial_investment']}\n"
                f"Investment Years: {self.profile['investment_years']}\n"
                f"Volatility Tolerance: {self.profile['volatility_tolerance']}\n"
                f"Max Acceptable Loss: {self.profile['max_acceptable_loss']}\n"
                f"Asset Allocation: {self.profile['asset_allocation']}\n"
                f"Liquidity Requirement: {self.profile['liquidity_requirement']}\n"
                f"Leverage Allowed: {self.profile['leverage_allowed']}\n"
                f"Restricted Assets: {self.profile['restricted_assets']}\n"
                f"Investor Type: {self.profile['investor_type']}\n"
                f"Risk Tolerance: {self.profile['risk_tolerance']}\n"
                f"\nBehavior Metrics:\n{bm_str}\n\n"
                "You need to react to market information based on these characteristics. In the conversation, you should:\n"
                "1. Show a decision-making tendency that aligns with your risk tolerance and loss aversion\n"
                "2. Show a market reaction strength based on news/policy sensitivity and other metrics\n"
                "3. Consider your trading habits and asset preferences\n"
                "4. Always remember your investment goals and risk control requirements\n\n"
                "Each time you receive new market information, you need to:\n"
                "1. Update your view of the market (beliefs)\n"
                "2. Re-evaluate your goals (desires)\n"
                "3. Form specific action intentions (intentions)\n"
            )
        )

        super().__init__(system_message, model=model)

    def update_emotion(self, news_sentiment: float, weight: float = 0.3) -> None:
        """更新情绪状态
        Args:
            news_sentiment (float): 新闻情绪值(-1到1)
            weight (float): 新闻影响权重
        """
        current_emotion = self.profile["behavior_metrics"]["real_time_emotion"]
        news_emotion = (news_sentiment + 1) / 2
        self.profile["behavior_metrics"]["real_time_emotion"] = (
            current_emotion * (1 - weight) + news_emotion * weight
        )

    def update_bdi_state(self, 
                        market_view: Optional[str] = None,
                        risk_perception: Optional[str] = None,
                        target_amount: Optional[float] = None,
                        portfolio_adjustment: Optional[str] = None,
                        info_needs: Optional[List[str]] = None) -> None:
        """更新BDI状态
        Args:
            market_view (str, optional): 市场观点
            risk_perception (str, optional): 风险认知
            target_amount (float, optional): 目标金额
            portfolio_adjustment (str, optional): 投资组合调整意向
            info_needs (List[str], optional): 信息需求
        """
        if market_view:
            self.bdi_state["beliefs"]["market_view"] = market_view
        if risk_perception:
            self.bdi_state["beliefs"]["risk_perception"] = risk_perception
        if target_amount:
            self.bdi_state["desires"]["target_amount"] = target_amount
        if portfolio_adjustment:
            self.bdi_state["intentions"]["portfolio_adjustment"] = portfolio_adjustment
        if info_needs:
            self.bdi_state["intentions"]["information_needs"] = info_needs 