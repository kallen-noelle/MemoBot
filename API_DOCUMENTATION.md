# MemoBot API 接口文档

> 基础URL: `http://localhost:8000`
> 所有API前缀: `/api`

---

## 目录

1. [Chat API](#1-chat-api)
2. [Memory API](#2-memory-api)
3. [Models API](#3-models-api)
4. [Uploads API](#4-uploads-api)
5. [Skills API](#5-skills-api)
6. [MCP API](#6-mcp-api)

---

## 1. Chat API

### POST `/api/chat` - 调用 Agent 对话

**功能**: 与 AI Agent 进行对话交互

**请求头**:
```
Content-Type: application/json
```

**请求体**:
| 字段 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| uid | string | 是 | 用户ID | `user_123` |
| thread_id | string | 是 | 线程ID | `thread_abc` |
| message | string | 是 | 用户消息 | `你好，帮我解释什么是机器学习` |
| model_name | string | 否 | 模型名称，默认 `qwen-plus` | `qwen-plus` |

**请求示例**:
```json
{
    "uid": "user_123",
    "thread_id": "thread_abc",
    "message": "你好",
    "model_name": "qwen-plus"
}
```

**响应示例**:
```json
{
    "success": true,
    "message": "Agent 的回复内容",
    "thread_id": "thread_abc"
}
```

---

### DELETE `/api/chat/cache` - 清除 Agent 缓存

**功能**: 清除已缓存的 Agent 实例

**Query 参数**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| uid | string | 否 | 用户ID |
| model_name | string | 否 | 模型名称 |

**响应示例**:
```json
{
    "success": true,
    "message": "Agent cache cleared"
}
```

---

## 2. Memory API

### GET `/api/memory` - 获取记忆数据

**功能**: 获取用户的全局记忆数据（用户上下文、历史、事实）

**Query 参数**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| uid | string | 是 | 用户ID |

**响应示例**:
```json
{
    "version": "1.0",
    "lastUpdated": "2024-01-15T10:30:00Z",
    "user": {
        "workContext": {
            "summary": "正在开发 DeerFlow 项目",
            "updatedAt": "2024-01-15T10:30:00Z"
        },
        "personalContext": {
            "summary": "偏好简洁的回复",
            "updatedAt": "2024-01-14T08:00:00Z"
        },
        "topOfMind": {
            "summary": "构建记忆 API",
            "updatedAt": "2024-01-15T09:00:00Z"
        }
    },
    "history": {
        "recentMonths": {
            "summary": "近期开发活动",
            "updatedAt": "2024-01-15T10:30:00Z"
        },
        "earlierContext": {
            "summary": "",
            "updatedAt": ""
        },
        "longTermBackground": {
            "summary": "",
            "updatedAt": ""
        }
    },
    "facts": [
        {
            "id": "fact_abc123",
            "content": "用户偏好 TypeScript",
            "category": "preference",
            "confidence": 0.9,
            "createdAt": "2024-01-15T10:30:00Z",
            "source": "thread_xyz"
        }
    ]
}
```

---

### POST `/api/memory/reload` - 重载记忆数据

**功能**: 从存储文件重新加载记忆数据

**Query 参数**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| uid | string | 是 | 用户ID |

**响应**: 同 GET `/api/memory`

---

### GET `/api/memory/config` - 获取记忆配置

**功能**: 获取记忆系统的配置信息

**响应示例**:
```json
{
    "enabled": true,
    "storage_path": ".deer-flow/memory.json",
    "debounce_seconds": 30,
    "max_facts": 100,
    "fact_confidence_threshold": 0.7,
    "injection_enabled": true,
    "max_injection_tokens": 2000
}
```

---

### GET `/api/memory/status` - 获取记忆状态

**功能**: 同时获取记忆配置和当前数据

**Query 参数**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| uid | string | 是 | 用户ID |

**响应示例**:
```json
{
    "config": {
        "enabled": true,
        "storage_path": ".deer-flow/memory.json",
        "debounce_seconds": 30,
        "max_facts": 100,
        "fact_confidence_threshold": 0.7,
        "injection_enabled": true,
        "max_injection_tokens": 2000
    },
    "data": {
        "version": "1.0",
        "lastUpdated": "2024-01-15T10:30:00Z",
        "user": {...},
        "history": {...},
        "facts": [...]
    }
}
```

---

## 3. Models API

### GET `/api/models` - 列出所有模型

**功能**: 获取所有可用的 AI 模型列表

**响应示例**:
```json
{
    "models": [
        {
            "name": "qwen-plus",
            "model": "qwen3.5-plus",
            "display_name": "qwen",
            "description": null,
            "supports_thinking": true,
            "supports_reasoning_effort": false
        }
    ]
}
```

---

### GET `/api/models/{model_name}` - 获取模型详情

**功能**: 获取特定模型的详细信息

**路径参数**:
| 字段 | 类型 | 说明 |
|------|------|------|
| model_name | string | 模型名称 |

**响应示例**:
```json
{
    "name": "qwen-plus",
    "model": "qwen3.5-plus",
    "display_name": "qwen",
    "description": null,
    "supports_thinking": true,
    "supports_reasoning_effort": false
}
```

---

## 4. Uploads API

### POST `/api/uploads/upload` - 上传并处理文件

**功能**: 上传文件并提取内容存入向量数据库和图数据库

**Query 参数**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| uid | string | 是 | 用户ID |

**Body**: `multipart/form-data`

| 字段 | 类型 | 说明 |
|------|------|------|
| files | File | 要上传的文件（支持多文件） |

**支持的格式**:
- 文档: `.pdf`, `.doc`, `.docx`
- 表格: `.xls`, `.xlsx`
- 演示: `.ppt`, `.pptx`
- 文本: `.txt`, `.md`

**响应示例**:
```json
[
    {
        "success": true,
        "files": [
            {
                "filename": "document.pdf"
            }
        ],
        "message": "Processed successfully"
    }
]
```

**Apifox 配置示例**:
```
Method: POST
URL: http://localhost:8000/api/uploads/upload?uid=user_123
Body: multipart/form-data
files: [选择文件]
```

---

### GET `/api/uploads/list` - 列出已上传文件

**功能**: 列出用户上传的所有文件

**Query 参数**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| uid | string | 是 | 用户ID |

**响应示例**:
```json
{
    "files": [],
    "count": 0
}
```

---

### DELETE `/api/uploads/{filename}` - 删除文件

**功能**: 删除指定的已上传文件

**路径参数**:
| 字段 | 类型 | 说明 |
|------|------|------|
| filename | string | 要删除的文件名 |

---

## 5. Skills API

### GET `/api/skills` - 列出所有 Skills

**功能**: 获取所有可用的技能列表

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
        },
        {
            "name": "Frontend Design",
            "description": "Generate frontend designs",
            "license": null,
            "category": "custom",
            "enabled": false
        }
    ]
}
```

---

### GET `/api/skills/{skill_name}` - 获取 Skill 详情

**功能**: 获取特定技能的详细信息

**路径参数**:
| 字段 | 类型 | 说明 |
|------|------|------|
| skill_name | string | 技能名称 |

---

### PUT `/api/skills/{skill_name}` - 更新 Skill

**功能**: 启用或禁用某个技能

**路径参数**:
| 字段 | 类型 | 说明 |
|------|------|------|
| skill_name | string | 技能名称 |

**请求体**:
```json
{
    "enabled": true
}
```

---

### POST `/api/skills/install` - 安装 Skill

**功能**: 从 `.skill` 文件安装新技能

**请求体**:
```json
{
    "thread_id": "thread_abc",
    "path": "mnt/user-data/outputs/my-skill.skill"
}
```

**响应示例**:
```json
{
    "success": true,
    "skill_name": "my-skill",
    "message": "Skill installed successfully"
}
```

---

## 6. MCP API

### GET `/api/mcp/config` - 获取 MCP 配置

**功能**: 获取 Model Context Protocol 服务器配置

**响应示例**:
```json
{
    "mcp_servers": {
        "github": {
            "enabled": true,
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {
                "GITHUB_TOKEN": "ghp_xxx"
            },
            "description": "GitHub MCP server"
        }
    }
}
```

---

### PUT `/api/mcp/config` - 更新 MCP 配置

**功能**: 更新 MCP 服务器配置并保存到文件

**请求体**:
```json
{
    "mcp_servers": {
        "github": {
            "enabled": true,
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {},
            "description": "GitHub MCP server"
        }
    }
}
```

---

## Apifox 使用指南

### 环境配置
```
环境: Development
Base URL: http://localhost:8000
```

### 常用请求示例

#### 1. Chat 对话
```
POST /api/chat

{
    "uid": "user_123",
    "thread_id": "thread_abc",
    "message": "你好",
    "model_name": "qwen-plus"
}
```

#### 2. 文件上传
```
POST /api/uploads/upload?uid=user_123

Content-Type: multipart/form-data

files: [选择文件]
```

#### 3. 获取记忆
```
GET /api/memory?uid=user_123
```

#### 4. 列出模型
```
GET /api/models
```

---

## 错误响应

所有 API 错误响应格式:

```json
{
    "detail": "错误描述信息"
}
```

常见 HTTP 状态码:
- `200` - 请求成功
- `400` - 请求参数错误
- `404` - 资源不存在
- `500` - 服务器内部错误
