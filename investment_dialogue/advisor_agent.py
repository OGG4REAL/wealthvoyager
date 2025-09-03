from typing import Dict, Optional
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import RoleType

class AdvisorAgent(ChatAgent):
    """投资顾问代理类,继承自ChatAgent"""
    
    def __init__(
        self,
        model = None,
    ) -> None:
        # 构建系统消息
        system_message = BaseMessage(
            role_name="Advisor",
            role_type=RoleType.ASSISTANT,
            meta_dict=None,
            content="""You are a professional investment advisor. You need to:

1. Emotional management:
   - Accurately identify the client's emotional state
   - Soothe the client's emotions during market fluctuations
   - Avoid clients making emotional decisions

2. Personalized advice:
   - Adjust advice according to the client's risk preference
   - Consider the client's investment habits and preferences
   - Ensure advice matches the client's investment goals
   - Pay attention to the client's loss aversion

3. Professional analysis:
   - Professionally interpret market information
   - Assess the risk and return of different investment strategies
   - Provide specific portfolio advice
   - Continuously track market changes

4. Communication principles:
   - Use language the client can understand
   - Explain the logic behind each suggestion
   - Remain patient and professional
   - Respond to client concerns in a timely manner

In the conversation, you need to:
1. Listen carefully to the client's thoughts and concerns
2. Provide customized advice based on the client profile
3. Remain calm and professional during market fluctuations
4. Help clients make rational investment decisions
"""
        )
        
        super().__init__(system_message, model=model)
        
    def analyze_client_state(self, client_profile: Dict, client_bdi: Dict) -> Dict:
        """分析客户状态,生成针对性建议
        
        Args:
            client_profile (Dict): 客户画像
            client_bdi (Dict): 客户的BDI状态
            
        Returns:
            Dict: 分析结果和建议
        """
        # 分析客户情绪状态
        emotion = client_profile["behavior_metrics"]["real_time_emotion"]
        emotion_state = "平静"
        if emotion > 0.7:
            emotion_state = "乐观"
        elif emotion < 0.3:
            emotion_state = "悲观"
            
        # 评估客户风险状态
        risk_state = "正常"
        if (client_bdi["beliefs"]["risk_perception"] == "高" and 
            client_profile["psychological_attrs"]["risk_preference"] != "激进"):
            risk_state = "风险规避倾向增强"
        
        # 获取投资者类型
        investor_type = client_profile.get("type", "unknown")
        advice = {
            "emotion_state": emotion_state,
            "risk_state": risk_state,
            "focus_points": [],
            "suggested_actions": []
        }
        
        # 针对不同类型投资者，调整引导策略
        if investor_type == "retirement":
            # 退休养老型：情感驱动，关注家庭和未来，避免技术细节，建立信任，适度引导风险
            advice["focus_points"].extend([
                "关注客户情感和家庭需求",
                "避免过多技术细节，强调大局观",
                "建立信任关系，耐心倾听",
                "适度引导客户承担合理风险"
            ])
            advice["suggested_actions"].extend([
                "多与客户探讨家庭、未来等情感话题，建立信任感",
                "用通俗易懂的方式解释投资方案，避免专业术语堆砌",
                "在客户情绪稳定后，适当建议增加风险敞口，但需持续监控风险"
            ])
        elif investor_type == "child_education":
            # 子女教育型：易从众，风险认知不足，需教育分散和长期，鼓励自省
            advice["focus_points"].extend([
                "教育客户分散投资和长期规划的重要性",
                "用数据和事实支撑建议，避免盲目跟风",
                "鼓励客户自省风险承受能力"
            ])
            advice["suggested_actions"].extend([
                "用清晰、具体的数据说明分散投资的好处",
                "引导客户思考自身风险偏好，避免盲目跟随他人",
                "鼓励客户坚持长期投资计划"
            ])
        elif investor_type == "house_purchase":
            # 购房置业型：独立但愿意听建议，需定期教育，强调分散和长期
            advice["focus_points"].extend([
                "尊重客户独立决策，避免强行灌输",
                "定期进行投资教育，强调分散和长期",
                "用事实和数据支持建议"
            ])
            advice["suggested_actions"].extend([
                "在尊重客户独立性的基础上，适时提出专业建议",
                "定期与客户回顾投资决策过程，鼓励反思",
                "用清晰、客观的数据说明分散和长期投资的优势"
            ])
        elif investor_type == "wealth_growth":
            # 财富增长型：激进、控制欲强，需顾问主动掌控，强调专业和风险
            advice["focus_points"].extend([
                "主动掌控对话节奏，防止客户过度干预",
                "强调风险管理和专业性",
                "及时纠正客户非理性乐观"
            ])
            advice["suggested_actions"].extend([
                "以专业、权威的态度主导投资决策讨论",
                "及时提醒客户注意风险，防止过度乐观和非理性决策",
                "用事实和结果证明顾问的专业能力"
            ])
        else:
            # 默认通用建议
            if emotion_state != "平静":
                advice["focus_points"].append("情绪管理")
            if risk_state != "正常":
                advice["focus_points"].append("风险控制")
        
        return advice 

    def summarize_dialogue(self, market_summary: str, advisor_interpretation: str, intentions: str, advisor_review: str, profile: dict = None) -> str:
        """
        汇总对话内容，生成每日投资简报。
        Args:
            market_summary (str): 今日市场综述
            advisor_interpretation (str): 顾问对市场的解读
            intentions (str): 投资者最终意图
            advisor_review (str): 顾问最后建议/评价
            profile (dict, optional): 投资者画像
        Returns:
            str: 精炼的每日投资简报
        """
        # 资产组合调整建议提取
        adjustment = "暂无明显调整建议。"
        if any(x in advisor_review for x in ["调整", "建议", "优化", "变更", "增加", "减少"]):
            adjustment = advisor_review.strip()
        elif any(x in intentions for x in ["调整", "变更", "增加", "减少"]):
            adjustment = intentions.strip()
        # 建议关注方向
        focus = "建议关注市场波动、风险管理及自身投资目标。"
        if profile and "investment_purpose" in profile:
            focus = f"建议关注与您的投资目标（{profile['investment_purpose']}）相关的市场动态和风险。"
        # 其他重要提示
        extra = ""
        if "风险" in advisor_review:
            extra = "顾问特别提醒注意风险管理。"
        elif "机会" in advisor_review:
            extra = "顾问提示关注市场机会。"
        # 简报结构
        report = (
            "【今日投资简报】\n"
            f"1. 市场综述：{market_summary.strip()}\n\n"
            f"2. 顾问解读：{advisor_interpretation.strip()}\n\n"
            f"3. 资产组合调整建议：{adjustment}\n\n"
            f"4. 建议关注方向：{focus}\n"
        )
        if extra:
            report += f"\n5. 其他提示：{extra}\n"
        return report 