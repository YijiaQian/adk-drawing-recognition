"""
Google ADK 标准 Agent 入口文件
用于 adk run 和 adk web 命令
"""

from .drawing_agent_adk import DrawingRecognitionAgent

# 实例化 Agent
# 注意：adk run/web 需要一个名为 root_agent 的变量
agent_instance = DrawingRecognitionAgent()
root_agent = agent_instance.agent
