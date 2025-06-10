# WealthVoyager 项目说明文档

## 项目简介

WealthVoyager 是一个基于 Streamlit 的智能财富管理系统，集成了投资组合优化、用户画像抽取、智能体对话模拟、外部数据抓取等多项功能。项目分为主流程（WealthVoyager 目录）和智能对话子模块（investment_dialogue 目录），支持端到端的投资顾问场景。

---

## 目录结构

```
WealthVoyager/
  ├─ difynew.py                # Streamlit主入口，主流程与界面
  ├─ baseagent02.py            # 智能体决策与多轮优化
  ├─ portfolioptima.py         # 投资组合优化算法
  ├─ infor.py                  # 用户画像生成、Dify交互、日志
  ├─ firecrawl_client_as.py    # Firecrawl/MCP数据抓取
  ├─ config.py                 # 全局参数配置
  ├─ config_firecrawl.py       # Firecrawl/MCP专用配置
  ├─ behavior.py               # 行为指标查表
  └─ ...
investment_dialogue/
  ├─ main.py                   # 智能体对话主流程（异步）
  ├─ dialogue_manager.py       # 对话管理与BDI博弈
  ├─ investor_agent.py         # 投资者智能体
  ├─ advisor_agent.py          # 顾问智能体
  ├─ firecrawl_client.py       # Firecrawl/MCP数据抓取
  ├─ config.py                 # 智能体模型配置
  └─ ...
```

---

## 主流程说明（WealthVoyager）

1. **用户交互与数据采集**  
   - 通过 Streamlit 前端收集用户投资目标、年限、初始资金、资产配置等信息。
   - 用户输入通过 infor.py 的 handle_send() 发送到 Dify 大模型，抽取结构化画像，记录到 conversation_log.json。

2. **用户画像标准化**  
   - infor.py 的 build_base_config() 将 Dify/大模型返回的中英文字段、行为指标、资产配置等统一为 base_config 字典，作为后续优化和对话的标准输入。

3. **投资组合优化**  
   - 通过 baseagent02.py 的 InvestmentAdvisor 类，结合 portfolioptima.py 的 portfolio_optimization()，实现多轮蒙特卡洛模拟+大模型参数调整，输出多种可行投资方案。
   - 支持约束：波动率、最大回撤、流动性、厌恶资产、杠杆等。

4. **外部数据抓取**  
   - firecrawl_client_as.py 集成 Firecrawl/MCP 服务，自动抓取最新市场数据、资产收益率等，辅助优化。

5. **智能体对话/Agent模拟**  
   - 在 Streamlit 的"智能对话/Agent模拟"Tab，调用 investment_dialogue.main.main_async(profile)，将 base_config 转为 profile，驱动投资者-顾问多轮对话与博弈，输出每日投资简报。

---

## 智能体对话子模块（investment_dialogue）

- **main.py**  
  提供 main_async(profile) 异步接口，输入为标准化用户画像 profile，内部流程包括：
  1. Firecrawl/MCP 抓取市场综述
  2. 大模型摘要与情绪分析
  3. 顾问解读、投资者 BDI（信念-愿望-意图）推理
  4. 顾问-投资者多轮博弈，直至达成共识或轮数上限
  5. 输出详细日志和每日投资简报

- **dialogue_manager.py**  
  封装 RolePlaying 对话管理，支持市场新闻驱动的多轮智能体交互，BDI 状态管理，情绪动态更新。

- **investor_agent.py / advisor_agent.py**  
  分别实现投资者和顾问的智能体行为、BDI 状态、情绪建模、个性化决策逻辑。

- **firecrawl_client.py**  
  与主项目 firecrawl_client_as.py 类似，负责 Firecrawl/MCP 数据抓取。

---

## 模块衔接与数据流

- **主项目调用子模块**  
  WealthVoyager 在"智能对话/Agent模拟"Tab 通过如下方式集成 investment_dialogue：

  ```python
  from investment_dialogue.main import main_async
  logs, daily_report = asyncio.run(main_async(profile))
  ```

  其中 profile 来源于 base_config，字段需与 investment_dialogue 约定一致（见 investor_agent.py）。

- **数据流**  
  用户输入 → Dify/大模型抽取画像 → base_config → 投资组合优化 → profile → main_async(profile) → 智能体多轮对话与简报输出

---

## 关键难点与注意事项

1. **用户画像标准化**  
   - Dify/大模型返回的字段中英文混杂，需用 infor.py/build_base_config() 统一为英文 profile，行为指标通过 behavior.py 查表补全。

2. **投资组合优化约束**  
   - portfolioptima.py 支持多种约束（波动率、回撤、流动性、厌恶资产、杠杆），需确保 base_config 字段齐全且含义明确。

3. **智能体对话的 BDI 状态与情绪建模**  
   - investment_dialogue 采用 BDI（信念-愿望-意图）模型，投资者情绪动态受市场新闻驱动，顾问-投资者多轮博弈，需理解 dialogue_manager.py 的状态流转。

4. **外部服务配置与依赖**  
   - Dify、Firecrawl、Deepseek、Moonshot 等服务的 API Key、Base URL、模型参数需在 config.py/config_firecrawl.py 统一配置，避免冲突。

5. **接口一致性与异常处理**  
   - 主项目与子模块接口需严格对齐，profile 字段、数据类型、异常处理需标准化，便于扩展和维护。

---

## 开发与协作建议

- 新成员建议先从 difynew.py 主流程和 baseagent02.py 智能体优化逻辑入手，再阅读 investment_dialogue/main.py 的对话流程。
- 所有外部依赖和 API Key 建议通过环境变量或统一配置管理，避免硬编码。
- 对于用户画像、资产配置、行为指标等数据结构，建议团队内部制定标准文档，保持主项目与子模块同步。
- 日志与调试信息丰富，便于定位问题，建议保留关键 debug 输出。

---

## 参考接口示例

```python
# 主流程调用智能体对话
from investment_dialogue.main import main_async
logs, daily_report = asyncio.run(main_async(profile))
print(daily_report)
```

---

如需进一步了解各模块细节，请查阅对应 .py 文件源码或联系项目维护者。欢迎团队成员补充完善本说明文档！

---

（本 README 基于完整代码核查，确保描述与实际实现一致。如有变更请及时同步更新。） 