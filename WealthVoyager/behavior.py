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