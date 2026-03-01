# Airline Customer Support Agent

这个仓库是一个航空客服智能体项目。
项目支持 MCP 工具调用。
项目包含 Python 和 TypeScript 两套实现。

## 目录说明

- `python/agent/` 智能体主程序，含 CLI 和 Web UI
- `python/mcp_airline/` 航空领域 MCP 服务
- `data/` 航空数据和策略数据

## 环境准备

1. 准备 Python 3.11 或更高版本
2. 安装依赖
3. 配置模型 API Key

示例命令:

```bash
pip install -r requirements.txt
```

## 快速开始

1. 启动 MCP 航空服务
2. 启动智能体 CLI 或 Web UI
3. 按需开启 RAG 模式

示例命令:

```bash
python ingest_policy.py
agent-cli --rag http://localhost:3000/mcp
```

## 说明

- `.env` 等敏感配置文件不应上传
- 本仓库用于课程项目与实验
