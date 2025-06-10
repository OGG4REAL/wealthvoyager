import json
import math
import random
from typing import List, Optional

from langchain.schema import SystemMessage, HumanMessage, AIMessage, FunctionMessage
from langchain.chat_models import ChatOpenAI

FUNCTIONS = [
    {
        "name": "adjust_investment_strategy",
        "description": (
            "根据蒙特卡洛模拟的结果，基于限制，只修改被允许修改的属性，"
            "提高方案的成功率，使其超过阈值 {success_threshold}。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "target_amount": {
                            "type": "number",
                            "description": "目标金额"
                        },
                        "investment_years": {
                            "type": "number",
                            "description": "投资年限，单位：年"
                        },
                        "initial_investment": {
                            "type": "number",
                            "description": "初始投资金额"
                        },
                        "acceptable_volatility": {
                            "type": "number",
                            "description": "可接受的波动率上限"
                        },
                        "assets": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "可选择的投资资产（可选）"
                        }
                    },
                    "required": [
                        "target_amount",
                        "investment_years",
                        "initial_investment",
                        "acceptable_volatility"
                    ]
                },
                "success_threshold": {
                    "type": "number",
                    "description": "投资策略的最低成功率阈值",
                    "default": 0.6
                }
            },
            "required": ["config", "success_threshold"]
        }
    }
]


class MonteCarloSimulator:
    """
    蒙特卡洛模拟工具类，支持多资产收益率、协方差矩阵，
    并根据风险值找出随机组合。
    """

    # 默认资产预期收益率
    DEFAULT_RETURNS = {
        "A股": 0.0848,
        "债券": 0.0322,
        "REITs": 0.0843,
        "港股": 0.0839,
        "美股": 0.1038,
        "黄金": 0.0433,
        "大宗商品": 0.0318
    }

    # 默认资产协方差矩阵
    DEFAULT_COVARIANCE = {
        "A股": {
            "A股": 0.044479,
            "债券": 0.031437,
            "美股": 0.003474,
            "黄金": 0.002532,
            "大宗商品": -0.000609,
            "REITs": 0.004414,
            "港股": 0.020851
        },
        "债券": {
            "A股": 0.031437,
            "债券": 0.264196,
            "美股": -0.007815,
            "黄金": -0.003858,
            "大宗商品": -0.01188,
            "REITs": 0.004965,
            "港股": -0.024757
        },
        "美股": {
            "A股": 0.003474,
            "债券": -0.007815,
            "美股": 0.016053,
            "黄金": 0.004564,
            "大宗商品": 0.002928,
            "REITs": 0.000408,
            "港股": 0.004175
        },
        "黄金": {
            "A股": 0.002532,
            "债券": -0.003858,
            "美股": 0.004564,
            "黄金": 0.02253,
            "大宗商品": 0.009974,
            "REITs": 0.001208,
            "港股": 0.007991
        },
        "大宗商品": {
            "A股": -0.000609,
            "债券": -0.01188,
            "美股": 0.002928,
            "黄金": 0.009974,
            "大宗商品": 0.083463,
            "REITs": 0.001395,
            "港股": 0.013183
        },
        "REITs": {
            "A股": 0.004414,
            "债券": 0.004965,
            "美股": 0.000408,
            "黄金": 0.001208,
            "大宗商品": 0.001395,
            "REITs": 0.025921,
            "港股": -0.000408
        },
        "港股": {
            "A股": 0.020851,
            "债券": -0.024757,
            "美股": 0.004175,
            "黄金": 0.007991,
            "大宗商品": 0.013183,
            "REITs": -0.000408,
            "港股": 0.064262
        }
    }

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        mean_returns: Optional[List[float]] = None,
        cov_matrix: Optional[List[List[float]]] = None,
        risk_value: float = 0.15,
        current_allocation: Optional[List[float]] = None,
        num_simulations: int = 50000
    ):
        """
        - assets: 投资资产列表
        - mean_returns: 对应资产的年化收益率（若不提供，则从默认中提取）
        - cov_matrix: 对应资产的协方差矩阵（若不提供，则从默认中提取）
        - risk_value: 允许的最大组合波动率
        - current_allocation: 若指定则用之(须满足权重和=1等)，否则自动找一种随机权重
        - num_simulations: 迭代次数
        """
        self.assets = assets or ["A股", "债券", "REITs"]
        self.assets = self._filter_unknown_assets(self.assets)
        if not self.assets:
            self.assets = ["A股", "债券", "REITs"]

        self.n_assets = len(self.assets)
        self.risk_value = risk_value
        self.num_simulations = num_simulations

        # 处理收益率
        if mean_returns is None:
            self.mean_returns = [self.DEFAULT_RETURNS[a] for a in self.assets]
        else:
            self.mean_returns = mean_returns

        # 处理协方差矩阵
        if cov_matrix is None:
            self.cov_matrix = []
            for a1 in self.assets:
                row = []
                for a2 in self.assets:
                    row.append(self.DEFAULT_COVARIANCE[a1][a2])
                self.cov_matrix.append(row)
        else:
            self.cov_matrix = cov_matrix

        self.current_allocation = current_allocation

    @classmethod
    def _filter_unknown_assets(cls, assets: List[str]) -> List[str]:
        return [a for a in assets if a in cls.DEFAULT_RETURNS]

    @staticmethod
    def _random_weights(size: int) -> List[float]:
        vals = [random.random() for _ in range(size)]
        s = sum(vals)
        return [v / s for v in vals]

    @staticmethod
    def _dot_product(vec1: List[float], vec2: List[float]) -> float:
        return sum(a * b for a, b in zip(vec1, vec2))

    @classmethod
    def _matrix_vector_multiply(cls, matrix: List[List[float]], vector: List[float]) -> List[float]:
        return [cls._dot_product(row, vector) for row in matrix]

    @classmethod
    def _portfolio_volatility(cls, weights: List[float], cov_matrix: List[List[float]]) -> float:
        interm = cls._matrix_vector_multiply(cov_matrix, weights)
        return math.sqrt(cls._dot_product(weights, interm))

    @staticmethod
    def _multivariate_normal(mean: List[float], cov: List[List[float]], size: int) -> List[List[float]]:
        import random
        # 确保 size 是整数
        size = int(size)
        samples = []
        for _ in range(size):
            sample = [
                random.gauss(mean[i], math.sqrt(cov[i][i])) for i in range(len(mean))
            ]
            samples.append(sample)
        return samples

    @classmethod
    def _adjust_weights_to_risk(cls, weights: List[float], cov_matrix: List[List[float]], risk_value: float) -> List[float]:
        """
        若组合波动率超出风控阈值，则尝试搜索更合适的权重
        """
        current_vol = cls._portfolio_volatility(weights, cov_matrix)
        if current_vol <= risk_value:
            return weights

        max_attempts = 1000
        best_w = weights
        best_vol = current_vol
        for _ in range(max_attempts):
            new_w = cls._random_weights(len(weights))
            new_vol = cls._portfolio_volatility(new_w, cov_matrix)
            # 如果找到更合适的，就更新
            if new_vol <= risk_value and new_vol < best_vol:
                best_w = new_w
                best_vol = new_vol
        return best_w

    def run_simulation(self, initial_investment: float, years: int, target_value: float) -> dict:
        # 如果没指定当前权重，就随机找一个满足风险阈值的组合
        if not self.current_allocation:
            found_w = None
            for _ in range(10000):
                w = self._random_weights(self.n_assets)
                vol = self._portfolio_volatility(w, self.cov_matrix)
                if vol <= self.risk_value:
                    found_w = w
                    break
            if found_w is None:
                found_w = [1.0 / self.n_assets] * self.n_assets
                found_w = self._adjust_weights_to_risk(found_w, self.cov_matrix, self.risk_value)
            weights = found_w
        else:
            # 验证 current_allocation
            if len(self.current_allocation) != self.n_assets:
                raise ValueError("current_allocation 与资产数量不匹配")
            if not math.isclose(sum(self.current_allocation), 1.0, rel_tol=1e-6):
                raise ValueError("current_allocation 权重和必须是1")
            if any(w < 0 for w in self.current_allocation):
                raise ValueError("存在负权重")
            weights = self._adjust_weights_to_risk(self.current_allocation, self.cov_matrix, self.risk_value)

        final_values = []
        for _ in range(self.num_simulations):
            yearly_returns = self._multivariate_normal(self.mean_returns, self.cov_matrix, years)
            portfolio_growth = 1.0
            for ret_vec in yearly_returns:
                portfolio_growth *= (1.0 + self._dot_product(ret_vec, weights))
            final_values.append(initial_investment * portfolio_growth)

        mean_value = sum(final_values) / len(final_values)
        std_value = math.sqrt(
            sum((x - mean_value) ** 2 for x in final_values) / len(final_values)
        )
        target_probability = sum(1 for x in final_values if x >= target_value) / len(final_values) * 100
        if mean_value != 0:
            vol_percent_mean = std_value / mean_value * 100
        else:
            vol_percent_mean = 0  # 或者设置为其他默认值

        return {
            "weights": [round(w, 4) for w in weights],
            "mean_final_value": round(mean_value, 2),
            "portfolio_volatility": round(std_value, 2),
            "volatility_percent_mean": round(vol_percent_mean, 2),
            "target_achievement_probability": round(target_probability, 2),
            "assets": self.assets
        }


class InvestmentAdvisor:
    """
    多轮迭代投资顾问：
      - 当成功率不达标时，允许大模型调用 "adjust_investment_strategy" 函数输出新配置。
      - 我们在本地不做任何自动修改，把“怎么调”完全交给大模型。
    """

    def __init__(self, user_config: dict, openai_api_key: str, openai_api_base: str):
        self.user_config = user_config
        self.success_threshold = user_config.get("success_threshold", 0.6)

        self.llm = ChatOpenAI(
            model_name="deepseek-chat",
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            temperature=0.0,
            model_kwargs={
                "functions": FUNCTIONS,
                "function_call": "auto"
            }
        )

    def local_monte_carlo(self, config: dict) -> float:
        simulator = MonteCarloSimulator(
            assets=config.get("assets"),
            risk_value=config["acceptable_volatility"]
        )
        result = simulator.run_simulation(
            initial_investment=config["initial_investment"],
            years=config["investment_years"],
            target_value=config["target_amount"]
        )
        return result["target_achievement_probability"] / 100.0

    def process_messages(self, messages: list, current_config: dict) -> AIMessage:
        """
        这里处理大模型返回的消息，若它调用了函数，就在本地执行，然后把执行结果再回给大模型。
        """
        try:
            response = self.llm.invoke(messages)

            if (
                isinstance(response, AIMessage)
                and "function_call" in response.additional_kwargs
            ):
                fn_name = response.additional_kwargs["function_call"]["name"]
                args_str = response.additional_kwargs["function_call"]["arguments"]

                try:
                    args = json.loads(args_str)
                except Exception:
                    return AIMessage(
                        content=json.dumps({
                            "config": current_config,
                            "reasoning": "函数参数解析失败，保持原配置"
                        }),
                        role="assistant"
                    )

                if fn_name == "adjust_investment_strategy":
                    # 大模型尝试调整 config
                    new_config = self.local_adjust_investment_strategy(
                        args["config"],
                        args["success_threshold"]
                    )
                    function_msg = FunctionMessage(
                        name=fn_name,
                        content=json.dumps(new_config, ensure_ascii=False)
                    )
                    new_messages = messages + [
                        AIMessage(content=" ", additional_kwargs={"function_call": response.additional_kwargs["function_call"]}),
                        function_msg
                    ]
                    followup_response = self.llm.invoke(new_messages)

                    if not followup_response.content:
                        return AIMessage(
                            content=json.dumps({
                                "config": new_config.get("config", current_config),
                                "reasoning": "保持当前配置(大模型无额外回复)"
                            }),
                            role="assistant"
                        )
                    return followup_response

                else:
                    return AIMessage(
                        content=json.dumps({
                            "config": current_config,
                            "reasoning": f"未知函数名：{fn_name}，保持原配置"
                        }),
                        role="assistant"
                    )

            else:
                if not response.content:
                    return AIMessage(
                        content=json.dumps({
                            "config": current_config,
                            "reasoning": "大模型未给出任何内容，保持原配置"
                        }),
                        role="assistant"
                    )
                return response

        except Exception as e:
            return AIMessage(
                content=json.dumps({
                    "config": current_config,
                    "reasoning": f"处理出错：{str(e)}，保持原配置"
                }),
                role="assistant"
            )

    def local_adjust_investment_strategy(self, config: dict, success_threshold: float) -> dict:
        """
        彻底交给大模型来调参，这里只做蒙特卡洛测试 + 把结果告诉大模型；
        不再在本地做任何修改。
        """
        success_rate = self.local_monte_carlo(config)
        reasoning = (
            f"本地检测成功率: {success_rate:.2%}, "
            f"阈值: {success_threshold:.2%}. "
        )
        if success_rate >= success_threshold:
            reasoning += "已达标，无需调整。"
        else:
            reasoning += "未达标，需要大模型重新尝试新的参数。"

        # 不做本地改动，直接把大模型传入的 config 原样返回
        return {
            "config": config,
            "reasoning": reasoning
        }

    def request_new_config_from_llm(
        self,
        current_config: dict,
        current_success_rate: float,
        allowed_params: list
    ) -> dict:
        """
        1) 给大模型一些信息（当前配置、成功率、可修改字段等）;
        2) 大模型可能会再次调用函数 monte_carlo_simulation，也可能直接输出文本;
        3) 我们要从大模型的最终回复 (AIMessage) 中解析出新的 config JSON。
        4) 对不在 allowed_params 里的字段恢复原值。
        """

        gap_to_target = self.success_threshold - current_success_rate
        adjustment_intensity = "大幅" if abs(gap_to_target) > 0.2 else "小幅"

        # 准备系统提示
        system_prompt = (
            f"你是一位资深理财顾问，需要通过调整投资参数来达到目标成功率。\n"
            f"当前成功率与目标相差 {gap_to_target:.2%}，需要{adjustment_intensity}调整。\n\n"
            "严格要求：\n"
            f"1. 你只能修改 {', '.join(allowed_params)} 字段\n"
            "2. 其他字段必须保持完全不变\n"
            "3. 必须使用以下完整的 JSON 格式返回，不要添加任何其他文本：\n\n"
            "{\n"
            '    "config": {\n'
            f'        "target_amount": {current_config["target_amount"]},\n'
            f'        "investment_years": {current_config["investment_years"]},\n'
            f'        "initial_investment": {current_config["initial_investment"]},\n'
            f'        "acceptable_volatility": {current_config["acceptable_volatility"]}\n'
            "    },\n"
            '    "reasoning": "在这里说明调整原因"\n'
            "}\n\n"
            "调整原则：\n"
            "1. 如果成功率过低，应该：\n"
            "   - 降低目标金额 target_amount\n"
            "   - 增加投资年限 investment_years\n"
            "   - 增加初始投资 initial_investment\n\n"
            "调整幅度规则：\n"
            f"1. 当前允许修改的字段 {', '.join(allowed_params)} 的调整幅度：\n"
            "   - 离目标远（>20%）时：调整 20-35%\n"
            "   - 接近目标时：调整 5-10%\n"
            "2. 其他字段：严禁修改\n\n"
            "参数限制：\n"
            f"- 如果修改 target_amount：不得低于 initial_investment 的 1.2 倍 ({current_config['initial_investment'] * 1.2})\n"
            f"- 如果修改 investment_years：必须在 1-10 年范围内\n"
            f"- 如果修改 initial_investment：每次调整不超过当前值的 35%\n\n"
            f"当前不可修改的字段及其值：\n{json.dumps({k: v for k, v in current_config.items() if k not in allowed_params}, indent=2)}\n\n"
            "你可以使用 monte_carlo_simulation 函数测试新配置的效果。"
        )

        # 准备用户提示
        user_prompt = f"""
    当前配置: {json.dumps(current_config, ensure_ascii=False)}
    当前成功率: {current_success_rate:.2%}
    目标成功率: {self.success_threshold:.2%}
    差距: {gap_to_target:.2%}

    请遵循系统提示中的调整原则，可以多次调用 monte_carlo_simulation 来测试不同配置。
    返回JSON格式，包含两个字段：
    1. config: 新的配置
    2. reasoning: 详细说明你的调整逻辑和每个修改的原因

    config结构示例:
    {{
    "target_amount": ...,
    "investment_years": ...,
    "initial_investment": ...,
    "acceptable_volatility": ...
    }}
    （对不允许修改的字段，务必保持原值不变）
    """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response_message = self.process_messages(messages, current_config)
        content_str = response_message.content.strip()

        # 解析大模型返回的 JSON
        data = None
        try:
            data = json.loads(content_str)
        except:
            # 如果有 ```json``` 包裹，先提取
            json_str = ""
            if "```json" in content_str:
                try:
                    json_str = content_str.split("```json")[1].split("```")[0].strip()
                except:
                    pass
            elif "```" in content_str:
                try:
                    json_str = content_str.split("```")[1].split("```")[0].strip()
                except:
                    pass
            else:
                start = content_str.find("{")
                end = content_str.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = content_str[start:end]

            if json_str:
                try:
                    data = json.loads(json_str)
                except Exception as e:
                    print(f"⚠️ 再次解析失败, 保持原配置. 错误: {e}")
                    return current_config
            else:
                print("⚠️ 无法找到有效 JSON，保持原配置")
                return current_config

        if not data:
            print("⚠️ 大模型未返回可解析 JSON，保持原配置")
            return current_config

        new_config = data.get("config", {})
        if not isinstance(new_config, dict):
            print("⚠️ 返回的 config 不是对象，保持原配置")
            return current_config

        reasoning = data.get("reasoning", "无详细说明")

        print("\n=== 大模型给出的改动建议 ===")
        print(f"允许修改字段: {allowed_params}")
        print("返回的 new_config:", new_config)
        print("理由:", reasoning)

        # 强制只改 allowed_params
        sanitized = dict(current_config)
        for k, old_v in current_config.items():
            if k in allowed_params and k in new_config:
                sanitized[k] = new_config[k]
            else:
                sanitized[k] = old_v

        changed = [k for k in sanitized if sanitized[k] != current_config[k]]
        if changed:
            print(f"本次实际修改字段: {changed}")
        else:
            print("⚠️ 大模型没有对允许修改的参数进行调整。")

        return sanitized

    def run_optimization(self, allowed_params: list, max_rounds: int = 5):
        """
        多轮迭代优化流程:
        1) 先本地测成功率
        2) 如果达到要求就结束，否则请求大模型改策略
        3) 大模型可多次调用 adjust_investment_strategy 并输出新参数
        4) 循环直到达标或超出最大轮数
        """
        current_config = dict(self.user_config)
        best_config = None
        best_rate = 0
        
        for round_i in range(1, max_rounds + 1):
            success_rate = self.local_monte_carlo(current_config)
            print(f"\n==== 第 {round_i} 轮模拟 ====")
            print("当前配置:", current_config)
            print(f"当前成功率: {success_rate:.2%}")

            # 更新最佳配置
            if success_rate > best_rate:
                best_config = dict(current_config)
                best_rate = success_rate

            if success_rate >= self.success_threshold:
                print(f"🎉 已达到目标成功率({self.success_threshold:.2%})，停止迭代。")
                break
            else:
                print("未达标，让大模型提供改进策略...")
                new_config = self.request_new_config_from_llm(
                    current_config,
                    success_rate,
                    allowed_params
                )
                current_config = new_config

        # 确保返回的是最佳配置
        final_config = best_config if best_config else current_config
        final_rate = self.local_monte_carlo(final_config)
        
        # 多次验证最终成功率
        verification_rates = []
        for _ in range(5):
            rate = self.local_monte_carlo(final_config)
            verification_rates.append(rate)
        final_rate = sum(verification_rates) / len(verification_rates)

        print("\n======== 优化结束 ========")
        print("最终配置:", final_config)
        print(f"最终成功率: {final_rate:.2%}\n")
        return final_config, final_rate


if __name__ == "__main__":
    OPENAI_API_KEY = "sk-HCkRhThj35aWxnYND50c64BbC026434f95E6538bE92f1cC5"
    OPENAI_API_BASE = "https://api.shubiaobiao.cn/v1"

    base_config = {
        "target_amount": 250000,
        "investment_years": 3,
        "initial_investment": 150000,
        "acceptable_volatility": 0.2,
        "success_threshold": 0.6
    }

    advisor = InvestmentAdvisor(base_config, OPENAI_API_KEY, OPENAI_API_BASE)

    # 运行三种优化方案
    print("\n======== 投资方案优化选项 ========")
    
    print("\n方案 A：调整目标金额")
    config_a, rate_a = advisor.run_optimization(["target_amount"], max_rounds=3)
    
    print("\n方案 B：调整投资年限")
    config_b, rate_b = advisor.run_optimization(["investment_years"], max_rounds=3)
    
    print("\n方案 C：同时调整投资年限和初始投资")
    config_c, rate_c = advisor.run_optimization(["investment_years", "initial_investment"], max_rounds=3)

    # 整合输出结果
    print("\n======== 投资方案对比 ========")
    results = [
        ("A. 调整目标金额方案", config_a, rate_a),
        ("B. 调整投资年限方案", config_b, rate_b),
        ("C. 调整年限和初始投资方案", config_c, rate_c)
    ]
    
    # 先显示所有方案
    print("\n📊 所有可选方案：")
    for name, config, rate in results:
        print(f"\n{name}:")
        print(f"   - 目标金额: {config['target_amount']:,.0f} 元")
        print(f"   - 投资年限: {config['investment_years']} 年")
        print(f"   - 初始投资: {config['initial_investment']:,.0f} 元")
        print(f"   - 成功概率: {rate:.1%}")
        print(f"   - 达标状态: {'✅ 已达标' if rate >= 0.6 else '⚠️ 未达标'}")

    # 筛选出达标方案
    valid_results = [r for r in results if r[2] >= 0.6]
    
    if valid_results:
        print("\n💡 推荐选择：", end="")
        print("/".join([r[0].split(".")[0] for r in valid_results]))
    else:
        print("\n⚠️ 提示：所有方案均未达到60%的目标成功率，建议调整投资目标或参数范围")

    print("\n请选择投资方案（A/B/C）...")



