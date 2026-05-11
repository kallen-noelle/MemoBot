const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = {
  chat: {
    send: `${API_BASE_URL}/api/chat`,
    clearCache: `${API_BASE_URL}/api/chat/cache`,
  },
  uploads: {
    upload: `${API_BASE_URL}/api/uploads/upload`,
    list: `${API_BASE_URL}/api/uploads/list`,
  },
  memory: {
    get: `${API_BASE_URL}/api/memory`,
    reload: `${API_BASE_URL}/api/memory/reload`,
    config: `${API_BASE_URL}/api/memory/config`,
    status: `${API_BASE_URL}/api/memory/status`,
  },
  models: {
    list: `${API_BASE_URL}/api/models`,
    get: (modelName: string) => `${API_BASE_URL}/api/models/${modelName}`,
  },
  skills: {
    list: `${API_BASE_URL}/api/skills`,
    get: (skillName: string) => `${API_BASE_URL}/api/skills/${skillName}`,
    update: (skillName: string) => `${API_BASE_URL}/api/skills/${skillName}`,
    install: `${API_BASE_URL}/api/skills/install`,
  },
  mcp: {
    config: `${API_BASE_URL}/api/mcp/config`,
  },
};

export interface ChatRequest {
  uid: string;
  thread_id: string;
  message: string;
  model_name?: string;
}

export interface ChatResponse {
  success: boolean;
  message: string;
  thread_id: string;
}

export interface UploadResponse {
  success: boolean;
  files: Array<{ filename: string }>;
  message: string;
}

export interface MemoryResponse {
  version: string;
  lastUpdated: string;
  user: {
    workContext: { summary: string; updatedAt: string };
    personalContext: { summary: string; updatedAt: string };
    topOfMind: { summary: string; updatedAt: string };
  };
  history: {
    recentMonths: { summary: string; updatedAt: string };
    earlierContext: { summary: string; updatedAt: string };
    longTermBackground: { summary: string; updatedAt: string };
  };
  facts: Array<{
    id: string;
    content: string;
    category: string;
    confidence: number;
    createdAt: string;
    source: string;
  }>;
}

export interface ModelInfo {
  name: string;
  model: string;
  display_name: string | null;
  description: string | null;
  supports_thinking: boolean;
  supports_reasoning_effort: boolean;
}

export interface ModelsListResponse {
  models: ModelInfo[];
}
