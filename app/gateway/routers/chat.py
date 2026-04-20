"""API for calling agent with user input."""

import logging
import threading
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.agent.lead_agent.agent import make_lead_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

# Agent 单例缓存
_agent_cache: Dict[str, object] = {}
_agent_cache_lock = threading.Lock()


class ChatRequest(BaseModel):
    """Request model for chat completion."""
    
    uid: str = Field(..., description="User ID (e.g., user_123)")
    thread_id: str = Field(..., description="Thread ID (e.g., id_12345)")
    message: str = Field(..., description="User input message")
    model_name: str = Field("qwen-plus", description="Model name (e.g., qwen-plus)")


class ChatResponse(BaseModel):
    """Response model for chat completion."""
    
    success: bool = Field(..., description="Whether the request succeeded")
    message: str = Field(..., description="Agent's response message")
    thread_id: str = Field(..., description="Thread ID")


def _get_agent_key(uid: str, model_name: str) -> str:
    """Generate a unique key for caching agents."""
    return f"{uid}:{model_name}"


def _get_or_create_agent(uid: str, model_name: str) -> object:
    """Get cached agent or create a new one with singleton pattern."""
    key = _get_agent_key(uid, model_name)
    
    with _agent_cache_lock:
        if key in _agent_cache:
            logger.debug(f"Using cached agent for key: {key}")
            return _agent_cache[key]
        
        # Create new agent
        logger.info(f"Creating new agent for key: {key}")
        agent_config = {
            "configurable": {
                "thinking_enabled": True,
                "model_name": model_name,
                "is_bootstrap": False,
                "uid": uid
            },
            "metadata": {}
        }
        
        agent = make_lead_agent(agent_config)
        _agent_cache[key] = agent
        return agent


@router.post("", response_model=ChatResponse, summary="Call Agent")
async def chat(request: ChatRequest) -> ChatResponse:
    """Call the agent with user input and get a response.
    
    Args:
        request: Chat request containing uid, thread_id, message, and model_name.
        
    Returns:
        Agent's response message.
        
    Raises:
        HTTPException: 500 if agent call fails.
    """
    try:
        # Get or create agent (using singleton pattern)
        agent = _get_or_create_agent(
            uid=request.uid,
            model_name=request.model_name
        )
        
        # Prepare messages
        messages = [
            {"role": "user", "content": request.message}
        ]
        
        # Build config with all required parameters
        invoke_config = {
            "configurable": {
                "thread_id": request.thread_id,
                "uid": request.uid,
                "model_name": request.model_name
            }
        }
        
        # Call agent
        response = agent.invoke(
            {"messages": messages},
            config=invoke_config,
            context={"thread_id": request.thread_id, "uid": request.uid, "model_name": request.model_name}
        )
        
        # Extract response message
        if isinstance(response, dict) and "messages" in response:
            msgs = response["messages"]
            if msgs:
                last_msg = msgs[-1]
                content = getattr(last_msg, 'content', str(last_msg))
                return ChatResponse(
                    success=True,
                    message=content,
                    thread_id=request.thread_id
                )
        
        return ChatResponse(
            success=True,
            message="No response from agent",
            thread_id=request.thread_id
        )
        
    except Exception as e:
        logger.error(f"Failed to call agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to call agent: {str(e)}")


@router.delete("/cache", summary="Clear Agent Cache")
async def clear_agent_cache(uid: Optional[str] = None, model_name: Optional[str] = None) -> dict:
    """Clear the agent cache for a specific user/model or all agents.
    
    Args:
        uid: Optional user ID to clear cache for.
        model_name: Optional model name to clear cache for.
        
    Returns:
        Success message.
    """
    with _agent_cache_lock:
        if uid and model_name:
            key = _get_agent_key(uid, model_name)
            if key in _agent_cache:
                del _agent_cache[key]
                logger.info(f"Cleared agent for uid={uid}, model={model_name}")
        elif uid:
            keys_to_remove = [k for k in _agent_cache if k.startswith(f"{uid}:")]
            for key in keys_to_remove:
                del _agent_cache[key]
            logger.info(f"Cleared {len(keys_to_remove)} agent(s) for uid: {uid}")
        else:
            _agent_cache.clear()
            logger.info("Cleared all agents from cache")
    
    return {"success": True, "message": "Agent cache cleared"}


@router.get("/cache/stats", summary="Get Cache Stats")
async def get_cache_stats() -> dict:
    """Get statistics about the agent cache.
    
    Returns:
        Cache statistics including number of cached agents.
    """
    with _agent_cache_lock:
        return {"cached_agents_count": len(_agent_cache)}
