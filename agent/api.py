"""
FastAPI server for Agentic RAG with Knowledge Graph.

Endpoints:
- Single-Agent System (Baseline): /chat, /chat/stream
- Multi-Agent System (MAS): /chat/mas, /chat/mas/stream
- Comparison: /compare
- Evaluation: /evaluation/*
- Metrics: /metrics/*
- Scientific Metrics: /metrics/scientific/*
- Health: /health
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import asynccontextmanager
import json
from uuid import uuid4
import time

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from pydantic_ai import Agent

from .db_utils import (
    initialize_database,
    close_database,
    db_pool,
    create_session,
    get_session,
    save_message,
    get_session_messages
)
from .graph_utils import initialize_graph, close_graph
from .providers import get_model
from .tools import (
    vector_search_tool,
    graph_search_tool,
    hybrid_search_tool,
    get_document_tool,
    list_documents_tool,
    get_tool_performance_summary,
    check_tools_health
)
from .models import AgentDependencies

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# OPTIONAL MODULE IMPORTS WITH GRACEFUL DEGRADATION
# ============================================================================

MONITORING_AVAILABLE = False
EVALUATION_AVAILABLE = False
MAS_AVAILABLE = False
SCIENTIFIC_METRICS_AVAILABLE = False

# Create dummy classes to avoid NameErrors
class DummyTracker:
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def mark_retrieval_complete(self): pass
    def set_response(self, *args): pass
    def add_tool(self, *args): pass

try:
    from monitoring.tracker import (
        initialize_tracker,
        shutdown_tracker,
        get_tracker,
        QueryTrackerContext,
        MetricsReporter
    )
    MONITORING_AVAILABLE = True
    logger.info("✓ Monitoring module loaded")
except ImportError as e:
    logger.warning(f"⚠ Monitoring module not available: {e}")
    QueryTrackerContext = DummyTracker
    async def initialize_tracker(*args, **kwargs): pass
    async def shutdown_tracker(): pass
    def get_tracker(): return None

try:
    from evaluation.evaluator import (
        EvaluationRunner,
        EvaluationConfig,
        run_quick_evaluation,
        compare_evaluation_runs,
        get_best_configuration
    )
    from evaluation.visualizer import (
        EvaluationVisualizer,
        quick_visualization,
        compare_runs_visually
    )
    EVALUATION_AVAILABLE = True
    logger.info("✓ Evaluation module loaded")
except ImportError as e:
    logger.warning(f"⚠ Evaluation module not available: {e}")
    class EvaluationConfig:
        def __init__(self, *args, **kwargs): pass
        def to_dict(self): return {}
    async def run_quick_evaluation(*args, **kwargs): return None

try:
    from agents.graph import MASGraphExecutor
    from agents.coordinator import CoordinatorAgent, create_coordinator
    MAS_AVAILABLE = True
    logger.info("✓ Multi-Agent System loaded")
except ImportError as e:
    logger.warning(f"⚠ Multi-Agent System not available: {e}")
    class MASGraphExecutor:
        def __init__(self, *args, **kwargs): pass
        async def execute(self, *args, **kwargs): 
            return {"answer": "", "metadata": {}}
        def get_graph_structure(self): 
            return "MAS not available"
    
    class CoordinatorAgent:
        def __init__(self, *args, **kwargs): pass
        async def process_query(self, *args, **kwargs): 
            return {"answer": "", "metadata": {}}
        def get_coordinator_statistics(self): 
            return {}
    
    def create_coordinator(*args, **kwargs): 
        return CoordinatorAgent()

# ✅ NEW: Import scientific metrics collector
try:
    from evaluation.metrics_collector import (
        ScientificMetricsCollector,
        initialize_metrics_collector,
        get_metrics_collector
    )
    SCIENTIFIC_METRICS_AVAILABLE = True
    logger.info("✓ Scientific Metrics Collector loaded")
except ImportError as e:
    logger.warning(f"⚠ Scientific Metrics Collector not available: {e}")
    class ScientificMetricsCollector:
        def __init__(self, *args, **kwargs): pass
        async def start_evaluation_run(self, *args, **kwargs): return str(uuid4())
        async def complete_evaluation_run(self, *args, **kwargs): pass
        async def record_query_metrics(self, *args, **kwargs): return str(uuid4())
        async def record_retrieval_metrics(self, *args, **kwargs): return str(uuid4())
        async def record_validation_metrics(self, *args, **kwargs): pass
        async def get_evaluation_summary(self, *args, **kwargs): return {}
    
    async def initialize_metrics_collector(*args, **kwargs): pass
    def get_metrics_collector(): return None

# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

# Global instances
mas_executor: Any = None
coordinator_agent: Any = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global mas_executor, coordinator_agent, MAS_AVAILABLE
    
    # Startup
    logger.info("="*60)
    logger.info("Starting Agentic RAG API...")
    logger.info("="*60)
    
    try:
        # Initialize database
        await initialize_database()
        logger.info("✓ Database initialized")
        
        # Initialize graph
        await initialize_graph()
        logger.info("✓ Graph database initialized")
        
        # ✅ NEW: Initialize scientific metrics collector
        if SCIENTIFIC_METRICS_AVAILABLE:
            try:
                await initialize_metrics_collector(db_pool)
                logger.info("✓ Scientific Metrics Collector initialized")
            except Exception as e:
                logger.warning(f"⚠ Failed to initialize metrics collector: {e}")
        
        # Initialize monitoring (optional)
        if MONITORING_AVAILABLE:
            try:
                await initialize_tracker(
                    buffer_size=1000,
                    window_minutes=5,
                    persist_interval_seconds=60,
                    enable_persistence=True
                )
                logger.info("✓ Metrics tracker initialized")
            except Exception as e:
                logger.warning(f"⚠ Failed to initialize monitoring: {e}")
        
        # Initialize Multi-Agent System (optional)
        if MAS_AVAILABLE:
            try:
                mas_executor = MASGraphExecutor(enable_validation=True)
                coordinator_agent = create_coordinator(use_validation=True)
                logger.info("✓ Multi-Agent System initialized")
            except Exception as e:
                logger.warning(f"⚠ Failed to initialize MAS: {e}")
                MAS_AVAILABLE = False
        
        logger.info("="*60)
        logger.info("✓ API startup complete")
        logger.info(f"  - Single-Agent: ✓")
        logger.info(f"  - Multi-Agent: {'✓' if MAS_AVAILABLE else '✗'}")
        logger.info(f"  - Monitoring: {'✓' if MONITORING_AVAILABLE else '✗'}")
        logger.info(f"  - Evaluation: {'✓' if EVALUATION_AVAILABLE else '✗'}")
        logger.info(f"  - Scientific Metrics: {'✓' if SCIENTIFIC_METRICS_AVAILABLE else '✗'}")
        logger.info("="*60)
        
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down Agentic RAG API...")
        
        if MONITORING_AVAILABLE:
            try:
                await shutdown_tracker()
                logger.info("✓ Metrics tracker shutdown")
            except:
                pass
        
        try:
            await close_graph()
            logger.info("✓ Graph database closed")
        except:
            pass
        
        try:
            await close_database()
            logger.info("✓ Database closed")
        except:
            pass
        
        logger.info("✓ API shutdown complete")

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Agentic RAG with Knowledge Graph",
    description="Single-Agent and Multi-Agent Systems with RAG and KG",
    version="3.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID")
    user_id: Optional[str] = Field(None, description="User ID")
    search_type: str = Field(default="hybrid", description="Search type: vector, graph, or hybrid")
    enable_validation: bool = Field(default=True, description="Enable answer validation (MAS only)")
    enable_scientific_metrics: bool = Field(default=False, description="Enable scientific metrics collection")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    message: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Session ID")
    system_type: str = Field(..., description="System used: 'single-agent' or 'multi-agent'")
    confidence: Optional[float] = Field(None, description="Answer confidence (MAS only)")
    sources: List[str] = Field(default=[], description="Source documents")
    validation: Optional[Dict[str, Any]] = Field(None, description="Validation report (MAS only)")
    metadata: Dict[str, Any] = Field(..., description="Execution metadata")
    scientific_metrics_id: Optional[str] = Field(None, description="Scientific metrics ID (if enabled)")

class ComparisonRequest(BaseModel):
    """Request for comparing single-agent vs MAS."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID")
    search_type: str = Field(default="hybrid", description="Search type")
    enable_scientific_metrics: bool = Field(default=False, description="Enable scientific metrics collection")

class ComparisonResponse(BaseModel):
    """Response with results from both systems."""
    query: str
    single_agent: ChatResponse
    multi_agent: ChatResponse
    comparison: Dict[str, Any]

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    components: Dict[str, bool]
    systems_available: Dict[str, bool]
    version: str

# ✅ NEW: Scientific metrics models
class EvaluationRunRequest(BaseModel):
    """Request to start an evaluation run."""
    name: str = Field(..., description="Name of evaluation run")
    description: str = Field(default="", description="Description of experiment")
    configuration: Dict[str, Any] = Field(default={}, description="Configuration parameters")

class EvaluationRunResponse(BaseModel):
    """Response for evaluation run."""
    run_id: str
    name: str
    status: str
    started_at: str
    configuration: Dict[str, Any]

# ============================================================================
# CORE ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Agentic RAG API",
        "version": "3.1.0",
        "systems": {
            "single_agent": True,
            "multi_agent": MAS_AVAILABLE
        },
        "endpoints": {
            "single_agent": {
                "chat": "/chat",
                "chat_stream": "/chat/stream"
            },
            "multi_agent": {
                "chat": "/chat/mas",
                "chat_stream": "/chat/mas/stream"
            } if MAS_AVAILABLE else None,
            "comparison": "/compare" if MAS_AVAILABLE else None,
            "scientific_metrics": {
                "start_run": "/metrics/scientific/runs/start",
                "complete_run": "/metrics/scientific/runs/{run_id}/complete",
                "get_run": "/metrics/scientific/runs/{run_id}",
                "export_csv": "/metrics/scientific/runs/{run_id}/export"
            } if SCIENTIFIC_METRICS_AVAILABLE else None,
            "health": "/health",
            "metrics": "/metrics",
            "evaluation": "/evaluation"
        },
        "features": {
            "monitoring": MONITORING_AVAILABLE,
            "evaluation": EVALUATION_AVAILABLE,
            "multi_agent": MAS_AVAILABLE,
            "scientific_metrics": SCIENTIFIC_METRICS_AVAILABLE
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check."""
    tools_health = await check_tools_health()
    
    all_healthy = all(tools_health.values())
    status = "healthy" if all_healthy else "degraded"
    
    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        components=tools_health,
        systems_available={
            "monitoring": MONITORING_AVAILABLE,
            "evaluation": EVALUATION_AVAILABLE,
            "multi_agent": MAS_AVAILABLE,
            "scientific_metrics": SCIENTIFIC_METRICS_AVAILABLE
        },
        version="3.1.0"
    )

# ============================================================================
# SINGLE-AGENT ENDPOINTS (BASELINE)
# ============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat_single_agent(request: ChatRequest):
    """
    Single-agent chat endpoint (baseline) with scientific metrics.
    """
    query_id = str(uuid4())
    start_time = time.time()
    metrics_id = None
    
    # Get scientific metrics collector
    collector = get_metrics_collector() if request.enable_scientific_metrics else None
    
    try:
        # Get or create session
        session_id = request.session_id or str(uuid4())
        session = await get_session(session_id)
        
        if not session:
            session = await create_session(
                session_id=session_id,
                user_id=request.user_id,
                metadata=request.metadata or {}
            )
        
        # Save user message
        await save_message(
            session_id=session_id,
            role="user",
            content=request.message,
            metadata=request.metadata or {}
        )
        
        # Track with monitoring if available
        ctx = None
        if MONITORING_AVAILABLE:
            try:
                ctx = QueryTrackerContext(
                    query_id=query_id,
                    query_text=request.message,
                    model=get_model(),
                    session_id=session_id,
                    user_id=request.user_id,
                    search_type=request.search_type
                )
                ctx.__enter__()
            except:
                ctx = None
        
        # Create single agent
        deps = AgentDependencies(
            session_id=session_id,
            search_type=request.search_type
        )
        
        # --- START FIX ---
        # 1. Get the raw model name
        model_name_str = get_model() # This will be "openai/gpt-oss-120b"
        
        # 2. Add the provider prefix (like we did in base_agent.py)
        formatted_model_name = model_name_str
        if ":" not in formatted_model_name:
            if formatted_model_name.startswith("gpt"):
                formatted_model_name = f"openai:{formatted_model_name}"
            elif "gpt-oss" in formatted_model_name: # The logic from the previous fix
                formatted_model_name = f"groq:{formatted_model_name}"
            elif "compound" in formatted_model_name:
                formatted_model_name = f"groq:{formatted_model_name}"

        # 3. Prepare the client config for Groq
        client_config = {}
        if formatted_model_name.startswith("groq:"):
            client_config = {
                "base_url": os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                "api_key": os.getenv("LLM_API_KEY") 
            }
        # --- END FIX ---

        agent = Agent(
            formatted_model_name, # Use the formatted name
            deps_type=AgentDependencies,
            system_prompt="""You are a helpful AI assistant with access to a knowledge base.
            You can search using vector similarity, knowledge graphs, and hybrid approaches.
            Always cite your sources and be precise in your answers.""",
            **client_config # Pass the config dictionary
        )
        
        # Register tools based on search type
        if request.search_type == "vector":
            agent.tool(vector_search_tool)
        elif request.search_type == "graph":
            agent.tool(graph_search_tool)
        else:  # hybrid
            agent.tool(hybrid_search_tool)
            agent.tool(graph_search_tool)
        
        agent.tool(get_document_tool)
        agent.tool(list_documents_tool)
        
        # Run agent
        retrieval_start = time.time()
        result = await agent.run(request.message, deps=deps)
        response_text = result.data
        retrieval_end = time.time()
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        retrieval_latency_ms = (retrieval_end - retrieval_start) * 1000
        
        # Save assistant message
        await save_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            metadata={"search_type": request.search_type}
        )
        
        # ✅ Record scientific metrics
        if collector:
            prompt_tokens = int(len(request.message.split()) * 1.3)
            completion_tokens = int(len(response_text.split()) * 1.3)
            
            metrics_id = await collector.record_query_metrics(
                query_text=request.message,
                response_text=response_text,
                system_type="single-agent",
                total_latency_ms=latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=latency_ms - retrieval_latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                estimated_cost_usd=0.0,  # Calculate if needed
                tools_used=[request.search_type],
                success=True,
                session_id=session_id,
                metadata={"search_type": request.search_type}
            )
        
        # Complete tracking
        if ctx:
            try:
                ctx.mark_retrieval_complete()
                prompt_tokens = len(request.message.split()) * 1.3
                completion_tokens = len(response_text.split()) * 1.3
                ctx.set_response(response_text, int(prompt_tokens), int(completion_tokens))
                ctx.__exit__(None, None, None)
            except:
                pass
        
        return ChatResponse(
            message=response_text,
            session_id=session_id,
            system_type="single-agent",
            confidence=None,
            sources=[],
            validation=None,
            metadata={
                "query_id": query_id,
                "latency_ms": latency_ms,
                "retrieval_latency_ms": retrieval_latency_ms,
                "search_type": request.search_type
            },
            scientific_metrics_id=metrics_id
        )
    
    except Exception as e:
        logger.error(f"Single-agent chat failed: {e}", exc_info=True)
        
        # Record error metrics
        if collector:
            await collector.record_query_metrics(
                query_text=request.message,
                response_text="",
                system_type="single-agent",
                total_latency_ms=(time.time() - start_time) * 1000,
                retrieval_latency_ms=None,
                generation_latency_ms=None,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                estimated_cost_usd=0.0,
                tools_used=[],
                success=False,
                error_message=str(e),
                session_id=request.session_id,
                metadata={}
            )
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream_single_agent(request: ChatRequest):
    """Streaming single-agent chat endpoint."""
    query_id = str(uuid4())
    
    async def event_generator():
        try:
            session_id = request.session_id or str(uuid4())
            session = await get_session(session_id)
            
            if not session:
                session = await create_session(
                    session_id=session_id,
                    user_id=request.user_id,
                    metadata=request.metadata or {}
                )
            
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'system_type': 'single-agent'})}\n\n"
            
            await save_message(
                session_id=session_id,
                role="user",
                content=request.message,
                metadata=request.metadata or {}
            )
            
            # Create agent
            deps = AgentDependencies(
                session_id=session_id,
                search_type=request.search_type
            )
            
            # --- START FIX ---
            # 1. Get the raw model name
            model_name_str = get_model() # This will be "openai/gpt-oss-120b"
            
            # 2. Add the provider prefix (like we did in base_agent.py)
            formatted_model_name = model_name_str
            if ":" not in formatted_model_name:
                if formatted_model_name.startswith("gpt"):
                    formatted_model_name = f"openai:{formatted_model_name}"
                elif "gpt-oss" in formatted_model_name: # The logic from the previous fix
                    formatted_model_name = f"groq:{formatted_model_name}"
                elif "compound" in formatted_model_name:
                    formatted_model_name = f"groq:{formatted_model_name}"

            # 3. Prepare the client config for Groq
            client_config = {}
            if formatted_model_name.startswith("groq:"):
                client_config = {
                    "base_url": os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                    "api_key": os.getenv("LLM_API_KEY") 
                }
            # --- END FIX ---

            agent = Agent(
                formatted_model_name, # Use the formatted name
                deps_type=AgentDependencies,
                system_prompt="You are a helpful AI assistant.",
                **client_config # Pass the config dictionary
            )
            
            if request.search_type == "hybrid":
                agent.tool(hybrid_search_tool)
                agent.tool(graph_search_tool)
            elif request.search_type == "vector":
                agent.tool(vector_search_tool)
            else:
                agent.tool(graph_search_tool)
            
            agent.tool(get_document_tool)
            agent.tool(list_documents_tool)
            
            # Stream response
            full_response = ""
            async with agent.run_stream(request.message, deps=deps) as stream:
                async for chunk in stream.stream_text():
                    full_response += chunk
                    yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
            
            await save_message(
                session_id=session_id,
                role="assistant",
                content=full_response,
                metadata={"search_type": request.search_type}
            )
            
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
        
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# ============================================================================
# MULTI-AGENT SYSTEM ENDPOINTS
# ============================================================================

if MAS_AVAILABLE:
    if MAS_AVAILABLE:
        @app.post("/chat/mas", response_model=ChatResponse)
        async def chat_multi_agent(request: ChatRequest):
            """Multi-Agent System chat endpoint with scientific metrics and detailed validation."""
            query_id = str(uuid4())
            start_time = time.time()
            metrics_id = None
            
            # Get scientific metrics collector
            collector = get_metrics_collector() if request.enable_scientific_metrics else None
            
            try:
                session_id = request.session_id or str(uuid4())
                session = await get_session(session_id)
                
                if not session:
                    session = await create_session(
                        session_id=session_id,
                        user_id=request.user_id,
                        metadata=request.metadata or {}
                    )
                
                await save_message(
                    session_id=session_id,
                    role="user",
                    content=request.message,
                    metadata=request.metadata or {}
                )
                
                # Execute MAS
                response = await mas_executor.execute(
                    query=request.message,
                    session_id=session_id,
                    user_id=request.user_id,
                    search_type=request.search_type,
                    config=request.metadata or {}
                )
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                # ✅ EXTRACT VALIDATION WITH DETAILED CLAIMS
                validation_report = response.get("validation")
                if validation_report:
                    # Build comprehensive validation response with ALL fields
                    validation_output = {
                        "is_valid": validation_report.get("is_valid", False),
                        "hallucination_score": validation_report.get("hallucination_score", 0.0),
                        "issues": validation_report.get("issues", []),
                        "verified_claims": validation_report.get("verified_claims", 0),
                        "total_claims": validation_report.get("total_claims", 0),
                        "papers_referenced": validation_report.get("papers_referenced", 0),
                        
                        # ✅ NEW: Expose detailed claims
                        "detailed_claims": validation_report.get("detailed_claims", []),
                        
                        # ✅ NEW: Expose hallucination examples
                        "hallucination_examples": validation_report.get("hallucination_examples", []),
                        
                        # ✅ NEW: Expose verification summary
                        "verification_summary": validation_report.get("verification_summary", {}),
                        
                        # Citation verification details
                        "citation_verification": validation_report.get("citation_verification", {})
                    }
                else:
                    validation_output = None
                
                # ✅ Record scientific metrics
                if collector:
                    prompt_tokens = int(len(request.message.split()) * 1.3)
                    completion_tokens = int(len(response["answer"].split()) * 1.3)
                    
                    metrics_id = await collector.record_query_metrics(
                        query_text=request.message,
                        response_text=response["answer"],
                        system_type="multi-agent",
                        total_latency_ms=latency_ms,
                        retrieval_latency_ms=response.get("metadata", {}).get("retrieval_latency_ms"),
                        generation_latency_ms=response.get("metadata", {}).get("generation_latency_ms"),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        estimated_cost_usd=0.0,
                        tools_used=response.get("metadata", {}).get("tools_used", [request.search_type]),
                        success=True,
                        session_id=session_id,
                        metadata=response.get("metadata", {})
                    )
                    
                    # Record retrieval metrics
                    await collector.record_retrieval_metrics(
                        query_metric_id=metrics_id,
                        search_type=request.search_type,
                        k=request.metadata.get("k", 5) if request.metadata else 5,
                        retrieved_chunk_ids=[],
                        relevant_chunk_ids=None,
                        precision_at_k=None,
                        recall_at_k=None,
                        f1_at_k=None,
                        mrr=None,
                        avg_similarity=0.0,
                        total_context_length=len(response["answer"]),
                        metadata={}
                    )
                    
                    # Record validation metrics
                    if validation_report:
                        await collector.record_validation_metrics(
                            query_metric_id=metrics_id,
                            validation_report=validation_report
                        )
                
                await save_message(
                    session_id=session_id,
                    role="assistant",
                    content=response["answer"],
                    metadata={
                        "system_type": "multi-agent",
                        "confidence": response.get("confidence"),
                        "validation": validation_output
                    }
                )
                
                response["metadata"]["latency_ms"] = latency_ms
                
                return ChatResponse(
                    message=response["answer"],
                    session_id=session_id,
                    system_type="multi-agent",
                    confidence=response.get("confidence"),
                    sources=response.get("sources", []),
                    validation=validation_output,  # ← Use validation_output with detailed claims
                    metadata=response["metadata"],
                    scientific_metrics_id=metrics_id
                )
            
            except Exception as e:
                logger.error(f"Multi-agent chat failed: {e}", exc_info=True)
                
                # Record error metrics
                if collector:
                    await collector.record_query_metrics(
                        query_text=request.message,
                        response_text="",
                        system_type="multi-agent",
                        total_latency_ms=(time.time() - start_time) * 1000,
                        retrieval_latency_ms=None,
                        generation_latency_ms=None,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        estimated_cost_usd=0.0,
                        tools_used=[],
                        success=False,
                        error_message=str(e),
                        session_id=request.session_id,
                        metadata={}
                    )
                
                raise HTTPException(status_code=500, detail=str(e))
        
    @app.post("/compare", response_model=ComparisonResponse)
    async def compare_systems(request: ComparisonRequest):
        """Compare Single-Agent vs Multi-Agent systems."""
        query_id = str(uuid4())
        
        # Get collector if metrics enabled
        collector = get_metrics_collector() if request.enable_scientific_metrics else None
        
        # Run both systems in parallel
        single_request = ChatRequest(
            message=request.message,
            session_id=request.session_id,
            search_type=request.search_type,
            enable_scientific_metrics=request.enable_scientific_metrics
        )
        
        mas_request = ChatRequest(
            message=request.message,
            session_id=request.session_id,
            search_type=request.search_type,
            enable_scientific_metrics=request.enable_scientific_metrics
        )
        
        # Execute both
        single_result, mas_result = await asyncio.gather(
            chat_single_agent(single_request),
            chat_multi_agent(mas_request)
        )
        
        # Compare results
        comparison = {
            "latency_difference_ms": mas_result.metadata["latency_ms"] - single_result.metadata["latency_ms"],
            "latency_ratio": mas_result.metadata["latency_ms"] / single_result.metadata["latency_ms"] if single_result.metadata["latency_ms"] > 0 else 0,
            "mas_has_validation": mas_result.validation is not None,
            "mas_confidence": mas_result.confidence,
            "response_length_single": len(single_result.message),
            "response_length_mas": len(mas_result.message)
        }
        
        return ComparisonResponse(
            query=request.message,
            single_agent=single_result,
            multi_agent=mas_result,
            comparison=comparison
        )

# ============================================================================
# SCIENTIFIC METRICS ENDPOINTS
# ============================================================================

if SCIENTIFIC_METRICS_AVAILABLE:
    @app.post("/metrics/scientific/runs/start", response_model=EvaluationRunResponse)
    async def start_scientific_run(request: EvaluationRunRequest):
        """Start a scientific evaluation run."""
        collector = get_metrics_collector()
        if not collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")
        
        run_id = await collector.start_evaluation_run(
            name=request.name,
            description=request.description,
            configuration=request.configuration
        )
        
        return EvaluationRunResponse(
            run_id=run_id,
            name=request.name,
            status="running",
            started_at=datetime.now().isoformat(),
            configuration=request.configuration
        )
    
    @app.post("/metrics/scientific/runs/{run_id}/complete")
    async def complete_scientific_run(run_id: str):
        """Complete a scientific evaluation run."""
        collector = get_metrics_collector()
        if not collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")
        
        await collector.complete_evaluation_run(run_id)
        summary = await collector.get_evaluation_summary(run_id)
        
        return {
            "run_id": run_id,
            "status": "completed",
            "summary": summary
        }
    
    @app.get("/metrics/scientific/runs/{run_id}")
    async def get_scientific_run(run_id: str):
        """Get detailed metrics for a scientific evaluation run."""
        collector = get_metrics_collector()
        if not collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")
        
        summary = await collector.get_evaluation_summary(run_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        
        return summary
    
    @app.get("/metrics/scientific/runs/{run_id}/export")
    async def export_scientific_run_csv(run_id: str):
        """Export evaluation run metrics as CSV for academic papers."""
        collector = get_metrics_collector()
        if not collector:
            raise HTTPException(status_code=503, detail="Metrics collector not available")
        
        # Get all metrics for the run
        import csv
        import io
        
        async with db_pool.connection() as conn:
            result = await conn.execute("""
                SELECT 
                    qm.id,
                    qm.query_text,
                    qm.system_type,
                    qm.total_latency_ms,
                    qm.retrieval_latency_ms,
                    qm.generation_latency_ms,
                    qm.prompt_tokens,
                    qm.completion_tokens,
                    qm.total_tokens,
                    qm.success,
                    qm.metadata->>'validation' as validation_report,
                    rm.search_type,
                    rm.k,
                    rm.precision_at_k,
                    rm.recall_at_k,
                    rm.f1_at_k,
                    rm.mrr,
                    rm.avg_similarity_score
                FROM query_metrics qm
                LEFT JOIN retrieval_metrics rm ON qm.id = rm.query_metric_id
                WHERE qm.evaluation_run_id = $1
                ORDER BY qm.created_at
            """, run_id)
            
            rows = await result.fetchall()
        
        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'query_id', 'query_text', 'system_type', 'total_latency_ms',
            'retrieval_latency_ms', 'generation_latency_ms', 'prompt_tokens',
            'completion_tokens', 'total_tokens', 'success', 'validation_report',
            'search_type', 'k', 'precision_at_k', 'recall_at_k', 'f1_at_k',
            'mrr', 'avg_similarity_score'
        ])
        
        # Data
        for row in rows:
            writer.writerow(row)
        
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=evaluation_run_{run_id}.csv"}
        )

# ============================================================================
# METRICS ENDPOINTS
# ============================================================================

@app.get("/metrics/tools")
async def get_tool_metrics():
    """Get tool execution metrics."""
    try:
        performance = get_tool_performance_summary()
        return {
            "tool_performance": performance,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get tool metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SESSION ENDPOINTS
# ============================================================================

@app.get("/sessions/{session_id}")
async def get_session_endpoint(session_id: str):
    """Get session details."""
    session = await get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session

@app.get("/sessions/{session_id}/messages")
async def get_session_messages_endpoint(session_id: str, limit: int = 50):
    """Get messages for a session."""
    messages = await get_session_messages(session_id, limit=limit)
    
    return {
        "session_id": session_id,
        "messages": messages,
        "count": len(messages)
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8058"))
    
    uvicorn.run(
        "agent.api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
