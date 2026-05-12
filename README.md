# MemoBot - 智能记忆助手

MemoBot 是一个基于AI的智能记忆助手应用，能够帮助用户记住重要信息、进行知识管理，并通过自然语言交互提供个性化服务。

## 功能特性

- **智能对话**: 使用先进的AI模型进行自然语言交互
- **记忆管理**: 自动记录和管理用户的重要信息和事实
- **RAG系统**: 基于检索增强生成技术的知识库问答
- **文件处理**: 支持多种格式的文件上传和分析
- **多模型支持**: 可配置不同的AI模型后端
- **向量数据库**: 使用Chroma向量数据库进行高效信息检索

## 演示

查看MemoBot的实际运行效果：

![MemoBot界面截图](./docs/memobot-screenshot.png)

## 技术架构

MemoBot采用现代化的技术栈构建：

- **后端**: Python + FastAPI
- **前端**: React + Next.js
- **AI模型**: 支持OpenAI兼容接口的模型（如Qwen系列）
- **向量存储**: ChromaDB + 文本嵌入
- **状态管理**: LangGraph工作流引擎
- **缓存**: SQLite检查点存储

## 核心功能

### 记忆系统
- 自动提取和存储对话中的重要信息
- 支持事实验证和置信度评估
- 提供个性化的上下文感知对话

### RAG检索
- 支持文档上传和解析
- 基于BM25和向量相似性的混合检索
- 智能文本分割和索引

### 工具集成
- 图像搜索功能
- 可扩展的工具生态系统
- 支持自定义工具开发

## 配置说明

项目使用 `config.yaml` 进行配置，主要配置项包括：

- **模型配置**: AI模型选择和API密钥设置
- **向量数据库**: ChromaDB配置和嵌入模型设置
- **记忆系统**: 记忆存储路径和参数
- **检查点**: 状态持久化配置
- **摘要功能**: 对话历史自动摘要

## 快速开始

1. 克隆项目到本地
2. 安装依赖
3. 配置API密钥和模型参数
4. 启动后端服务
5. 启动前端界面

## API接口

MemoBot提供了完整的RESTful API接口，包括：
- Chat API: 对话交互
- Memory API: 记忆数据管理
- Uploads API: 文件上传处理
- Models API: 模型管理
- Skills API: 技能系统

详细API文档请参见 [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)

## 本地部署指南

### 环境要求
- Python 3.12 或更高版本
- Node.js (如果需要开发前端)

### 后端部署

1. **克隆项目**
   ```bash
   git clone <your-repo-url>
   cd MemoBot
   ```

2. **创建虚拟环境并安装依赖**
   ```bash
   python -m venv venv
   # Windows
   venv\\Scripts\\activate
   # Linux/Mac
   source venv/bin/activate
   
   pip install -e .
   ```

3. **配置环境变量**
   复制 `.env` 文件并根据需要修改配置：
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，添加必要的API密钥等
   ```

4. **配置模型参数**
   修改 `config.yaml` 中的模型配置，例如阿里云DashScope API密钥：
   ```yaml
   models:
     - name: qwen3.5-plus
       display_name: qwen
       use: langchain_openai:ChatOpenAI
       model: qwen3.5-plus
       api_key: "your-api-key-here"  # 替换为你的API密钥
       base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
   ```

5. **启动后端服务**
   ```bash
   # 使用uvicorn启动FastAPI应用
   uvicorn app.gateway.app:app --host 0.0.0.0 --port 8000 --reload
   ```

### 前端部署

1. **进入前端目录**
   ```bash
   cd myfrontend
   ```

2. **安装依赖**
   ```bash
   # 如果使用pnpm
   pnpm install
   ```

3. **启动开发服务器**
   ```bash
   pnpm dev
   ```

4. **访问应用**
   - 后端API: http://localhost:8000
   - 前端页面: http://localhost:3000

### 容器化部署（可选）

MemoBot也支持使用Docker进行部署，具体配置可在后续扩展。

## 贡献

欢迎提交Issue和Pull Request来改进MemoBot！