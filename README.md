


          
我需要先了解项目结构，找到接口相关的代码文件，然后生成md格式的接口文档。


        
            
toolName: view_folder
            
status: success
          
D:\project\used_for_code\PyCharm\MemoBot
          
我需要继续探索项目结构，查找API相关的源代码文件。


        
            
toolName: file_search
            
status: success
          
          
我需要查看项目中的API路由文件来了解接口定义。让我先读取主要的路由文件。


        
            
toolName: view_files
            
status: success
          
            
filePath: D:\project\used_for_code\PyCharm\MemoBot\app\gateway\routers\chat.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: D:\project\used_for_code\PyCharm\MemoBot\app\gateway\routers\memory.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: D:\project\used_for_code\PyCharm\MemoBot\app\gateway\routers\uploads.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: D:\project\used_for_code\PyCharm\MemoBot\app\gateway\routers\skills.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: D:\project\used_for_code\PyCharm\MemoBot\app\gateway\routers\mcp.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: D:\project\used_for_code\PyCharm\MemoBot\app\gateway\routers\models.py
          
# MemoBot API 接口文档

## 概述

MemoBot 是一个智能记忆助手应用，提供以下主要功能模块的 API：

- **Chat**: 与 Agent 进行对话交互
- **Memory**: 管理全局记忆数据
- **Uploads**: 文件上传与处理
- **Skills**: 技能管理与安装
- **MCP**: Model Context Protocol 服务器配置
- **Models**: AI 模型管理

---

## 1. Chat API

### 1.1 调用 Agent

**POST** `/api/chat`

调用 Agent 处理用户输入并获取响应。

**请求体**:

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| uid | string | 是 | 用户 ID（如 user_123） |
| thread_id | string | 是 | 会话 ID（如 id_12345） |
| message | string | 是 | 用户输入消息 |
| model_name | string | 否 | 模型名称，默认为 "qwen-plus" |

**请求示例**:
```json
{
    "uid": "user_123",
    "thread_id": "id_12345",
    "message": "Hello, how are you?",
    "model_name": "qwen-plus"
}
```

**成功响应** (200):

| 字段 | 类型 | 描述 |
|------|------|------|
| success | boolean | 请求是否成功 |
| message | string | Agent 的响应消息 |
| thread_id | string | 会话 ID |

**响应示例**:
```json
{
    "success": true,
    "message": "I'm doing well! How can I assist you today?",
    "thread_id": "id_12345"
}
```

---

### 1.2 清除 Agent 缓存

**DELETE** `/api/chat/cache`

清除特定用户/模型的 Agent 缓存或全部缓存。

**查询参数**:

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| uid | string | 否 | 用户 ID |
| model_name | string | 否 | 模型名称 |

**成功响应** (200):
```json
{
    "success": true,
    "message": "Agent cache cleared"
}
```

---

### 1.3 获取缓存统计

**GET** `/api/chat/cache/stats`

获取 Agent 缓存的统计信息。

**成功响应** (200):
```json
{
    "cached_agents_count": 5
}
```

---

## 2. Memory API

### 2.1 获取记忆数据

**GET** `/api/memory?uid={uid}`

获取当前全局记忆数据，包括用户上下文、历史记录和事实。

**查询参数**:

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| uid | string | 是 | 用户 ID |

**成功响应** (200):

| 字段 | 类型 | 描述 |
|------|------|------|
| version | string | 记忆数据版本 |
| lastUpdated | string | 最后更新时间戳 |
| user | UserContext | 用户上下文信息 |
| history | HistoryContext | 历史上下文信息 |
| facts | list[Fact] | 事实列表 |

**响应示例**:
```json
{
    "version": "1.0",
    "lastUpdated": "2024-01-15T10:30:00Z",
    "user": {
        "workContext": {"summary": "Working on DeerFlow project", "updatedAt": "..."},
        "personalContext": {"summary": "Prefers concise responses", "updatedAt": "..."},
        "topOfMind": {"summary": "Building memory API", "updatedAt": "..."}
    },
    "history": {
        "recentMonths": {"summary": "Recent development activities", "updatedAt": "..."},
        "earlierContext": {"summary": "", "updatedAt": ""},
        "longTermBackground": {"summary": "", "updatedAt": ""}
    },
    "facts": [
        {
            "id": "fact_abc123",
            "content": "User prefers TypeScript over JavaScript",
            "category": "preference",
            "confidence": 0.9,
            "createdAt": "2024-01-15T10:30:00Z",
            "source": "thread_xyz"
        }
    ]
}
```

---

### 2.2 重新加载记忆数据

**POST** `/api/memory/reload?uid={uid}`

从存储文件重新加载记忆数据，刷新内存缓存。

**查询参数**:

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| uid | string | 是 | 用户 ID |

**成功响应** (200): 同 `GET /api/memory`

---

### 2.3 获取记忆配置

**GET** `/api/memory/config`

获取记忆系统的当前配置。

**成功响应** (200):

| 字段 | 类型 | 描述 |
|------|------|------|
| enabled | boolean | 是否启用记忆功能 |
| storage_path | string | 记忆存储文件路径 |
| debounce_seconds | int | 记忆更新防抖时间（秒） |
| max_facts | int | 最大存储事实数量 |
| fact_confidence_threshold | float | 事实置信度阈值（0-1） |
| injection_enabled | boolean | 是否启用记忆注入 |
| max_injection_tokens | int | 记忆注入最大 token 数 |

---

### 2.4 获取记忆状态

**GET** `/api/memory/status?uid={uid}`

获取记忆系统的完整状态，包括配置和数据。

**查询参数**:

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| uid | string | 是 | 用户 ID |

**成功响应** (200):

| 字段 | 类型 | 描述 |
|------|------|------|
| config | MemoryConfigResponse | 记忆配置 |
| data | MemoryResponse | 记忆数据 |

---

## 3. Uploads API

### 3.1 上传并处理文件

**POST** `/api/uploads/upload?uid={uid}`

上传文件并处理内容（转换为 Markdown 后存储到 RAG）。

**查询参数**:

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| uid | string | 是 | 用户 ID |

**请求体**: `multipart/form-data`

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| files | File | 是 | 要上传的文件列表 |

**支持的文件类型**:
- PDF (.pdf)
- PowerPoint (.ppt, .pptx)
- Excel (.xls, .xlsx)
- Word (.doc, .docx)

**成功响应** (200):

| 字段 | 类型 | 描述 |
|------|------|------|
| success | boolean | 是否成功 |
| files | list[dict] | 处理的文件信息 |
| message | string | 处理结果消息 |

**响应示例**:
```json
[
    {
        "success": true,
        "files": [{"filename": "document.pdf"}],
        "message": "Processed successfully"
    }
]
```

---

### 3.2 列出已上传文件

**GET** `/api/uploads/list?uid={uid}`

列出会话上传目录中的所有文件。

**查询参数**:

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| uid | string | 是 | 用户 ID |

**成功响应** (200):
```json
{
    "files": [],
    "count": 0
}
```

---

### 3.3 删除已上传文件

**DELETE** `/api/uploads/{filename}?uid={uid}`

从会话上传目录中删除指定文件。

**路径参数**:

| 参数 | 类型 | 描述 |
|------|------|------|
| filename | string | 要删除的文件名 |

**查询参数**:

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| uid | string | 是 | 用户 ID |

---

## 4. Skills API

### 4.1 列出所有技能

**GET** `/api/skills`

获取所有可用技能列表（包括公共和自定义技能）。

**成功响应** (200):

| 字段 | 类型 | 描述 |
|------|------|------|
| skills | list[SkillResponse] | 技能列表 |

**响应示例**:
```json
{
    "skills": [
        {
            "name": "PDF Processing",
            "description": "Extract and analyze PDF content",
            "license": "MIT",
            "category": "public",
            "enabled": true
        }
    ]
}
```

---

### 4.2 获取技能详情

**GET** `/api/skills/{skill_name}`

获取特定技能的详细信息。

**路径参数**:

| 参数 | 类型 | 描述 |
|------|------|------|
| skill_name | string | 技能名称 |

**成功响应** (200):

| 字段 | 类型 | 描述 |
|------|------|------|
| name | string | 技能名称 |
| description | string | 技能描述 |
| license | string | 许可证信息 |
| category | string | 技能类别（public/custom） |
| enabled | boolean | 是否启用 |

---

### 4.3 更新技能状态

**PUT** `/api/skills/{skill_name}`

更新技能的启用状态。

**路径参数**:

| 参数 | 类型 | 描述 |
|------|------|------|
| skill_name | string | 技能名称 |

**请求体**:

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| enabled | boolean | 是 | 是否启用技能 |

**请求示例**:
```json
{
    "enabled": false
}
```

**成功响应** (200): 更新后的技能信息。

---

### 4.4 安装技能

**POST** `/api/skills/install`

从 .skill 文件（ZIP 归档）安装技能。

**请求体**:

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| thread_id | string | 是 | 会话 ID |
| path | string | 是 | .skill 文件的虚拟路径 |

**请求示例**:
```json
{
    "thread_id": "abc123-def456",
    "path": "/mnt/user-data/outputs/my-skill.skill"
}
```

**成功响应** (200):

| 字段 | 类型 | 描述 |
|------|------|------|
| success | boolean | 是否安装成功 |
| skill_name | string | 安装的技能名称 |
| message | string | 安装结果消息 |

---

## 5. MCP API

### 5.1 获取 MCP 配置

**GET** `/api/mcp/config`

获取当前 Model Context Protocol 服务器配置。

**成功响应** (200):

| 字段 | 类型 | 描述 |
|------|------|------|
| mcp_servers | dict | MCP 服务器配置映射 |

**响应示例**:
```json
{
    "mcp_servers": {
        "github": {
            "enabled": true,
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_TOKEN": "ghp_xxx"},
            "description": "GitHub MCP server for repository operations"
        }
    }
}
```

---

### 5.2 更新 MCP 配置

**PUT** `/api/mcp/config`

更新 MCP 服务器配置并保存到文件。

**请求体**:

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| mcp_servers | dict | 是 | MCP 服务器配置映射 |

**请求示例**:
```json
{
    "mcp_servers": {
        "github": {
            "enabled": true,
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_TOKEN": "$GITHUB_TOKEN"},
            "description": "GitHub MCP server for repository operations"
        }
    }
}
```

**成功响应** (200): 更新后的 MCP 配置。

---

## 6. Models API

### 6.1 列出所有模型

**GET** `/api/models`

获取所有可用 AI 模型列表。

**成功响应** (200):

| 字段 | 类型 | 描述 |
|------|------|------|
| models | list[ModelResponse] | 模型列表 |

**响应示例**:
```json
{
    "models": [
        {
            "name": "gpt-4",
            "display_name": "GPT-4",
            "description": "OpenAI GPT-4 model",
            "supports_thinking": false
        },
        {
            "name": "claude-3-opus",
            "display_name": "Claude 3 Opus",
            "description": "Anthropic Claude 3 Opus model",
            "supports_thinking": true
        }
    ]
}
```

---

### 6.2 获取模型详情

**GET** `/api/models/{model_name}`

获取特定模型的详细信息。

**路径参数**:

| 参数 | 类型 | 描述 |
|------|------|------|
| model_name | string | 模型名称 |

**成功响应** (200):

| 字段 | 类型 | 描述 |
|------|------|------|
| name | string | 模型唯一标识 |
| model | string | 实际提供者模型标识 |
| display_name | string | 人类可读名称 |
| description | string | 模型描述 |
| supports_thinking | boolean | 是否支持思考模式 |
| supports_reasoning_effort | boolean | 是否支持推理努力 |

---

## 错误响应格式

所有 API 端点的错误响应格式统一：

```json
{
    "detail": "Error message describing the issue"
}
```

**常见 HTTP 状态码**:

| 状态码 | 描述 |
|--------|------|
| 400 | 请求参数无效 |
| 403 | 访问被拒绝 |
| 404 | 资源未找到 |
| 409 | 资源冲突（如技能已存在） |
| 500 | 服务器内部错误 |

---

## 数据模型

### ChatRequest

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| uid | string | 是 | 用户 ID |
| thread_id | string | 是 | 会话 ID |
| message | string | 是 | 用户消息 |
| model_name | string | 否 | 模型名称 |

### ChatResponse

| 字段 | 类型 | 描述 |
|------|------|------|
| success | boolean | 请求是否成功 |
| message | string | 响应消息 |
| thread_id | string | 会话 ID |

### MemoryResponse

| 字段 | 类型 | 描述 |
|------|------|------|
| version | string | 版本号 |
| lastUpdated | string | 最后更新时间 |
| user | UserContext | 用户上下文 |
| history | HistoryContext | 历史上下文 |
| facts | list[Fact] | 事实列表 |

### SkillResponse

| 字段 | 类型 | 描述 |
|------|------|------|
| name | string | 技能名称 |
| description | string | 技能描述 |
| license | string | 许可证 |
| category | string | 类别 |
| enabled | boolean | 是否启用 |

### ModelResponse

| 字段 | 类型 | 描述 |
|------|------|------|
| name | string | 模型标识 |
| model | string | 提供者模型标识 |
| display_name | string | 显示名称 |
| description | string | 描述 |
| supports_thinking | boolean | 支持思考模式 |
| supports_reasoning_effort | boolean | 支持推理努力 |
        