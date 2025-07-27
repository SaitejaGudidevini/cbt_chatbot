"""
CBT Context Engineering API - Only Existing Code
From: api/context_engineered_cbt_api.py + run_context_engineered_cbt.py
"""

import os
import sys
import logging
import uvicorn
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Query, Depends, Header, Request
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import httpx
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import json
from pathlib import Path

# Import our modules (existing code only)
from utils import config
from cbt_logic import ConversationManager, ConversationPhase
from context_logic import EnhancedConversationManager, EnhancedResponseGenerator
from utils import (
    StartConversationResponse, SendMessageRequest, SendMessageResponse,
    ConversationStateResponse, ResetConversationRequest, ResetConversationResponse,
    ExportConversationResponse, HealthResponse, ErrorResponse,
    # Add new imports for authentication
    UserSignupRequest, UserLoginRequest, UserAuthResponse, UserProfileResponse,
    # Import BaseModel and Field for request models
    BaseModel, Field
)

# Setup logging using existing config
logging.basicConfig(
    level=getattr(logging, config.get_api_config('log_level')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Authentication security
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dependency to get current user from JWT token
    This is a simplified version - in production you'd validate the JWT properly
    """
    if not config.pocketbase_manager:
        raise HTTPException(
            status_code=503,
            detail="Authentication not available. Database required."
        )
    
    token = credentials.credentials
    
    # For now, we'll do a simple token validation
    # In production, you'd decode and verify the JWT token properly
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    
    # TODO: Add proper JWT validation here
    # For now, we'll extract user_id from token (simplified)
    try:
        # This is a placeholder - you'd normally decode the JWT
        # For testing, we can pass user_id directly in the token
        return {"user_id": token, "authenticated": True}
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

class ContextEngineeredCBTAPI:
    """
    Enhanced CBT API that preserves original business logic while adding:
    - Context engineering with memory management
    - n8n workflow integration (optional)
    - Enhanced conversation tracking
    - AWS deployment ready configuration
    - User authentication with PocketBase
    """
    
    def __init__(self,
                 use_ollama: bool = True,
                 ollama_model: str = None,
                 ollama_base_url: str = None,
                 hf_endpoint_url: str = None,
                 hf_api_token: str = None,
                 n8n_base_url: str = None,
                 enable_n8n: bool = False,  # New parameter to disable n8n
                 knowledge_graph_path: str = None,
                 max_token_budget: int = None,
                 # ML Integration paths (now optional - uses config)
                 classifier_model_path: str = None,
                 sequence_regressor_path: str = None,
                 evaluator_model_path: str = None):
        
        logger.info("🚀 Initializing Context Engineered CBT API (AWS Ready)")
        
        # Initialize FastAPI with enhanced documentation
        self.app = FastAPI(
            title="Context Engineered CBT API with Authentication",
            description="Enhanced CBT conversation system with user authentication, context engineering, memory management, and optional n8n workflow integration",
            version="2.1.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Use configuration-based settings
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model or config.get_api_config('ollama_model')
        self.ollama_base_url = ollama_base_url or config.get_api_config('ollama_base_url')
        
        # n8n configuration - now optional
        self.enable_n8n = enable_n8n
        self.n8n_base_url = n8n_base_url or config.get_api_config('n8n_base_url') if enable_n8n else None
        
        # ML model paths (use config-based paths)
        self.classifier_model_path = classifier_model_path or str(config.get_model_path('classifier'))
        self.sequence_regressor_path = sequence_regressor_path or str(config.get_model_path('sequence_regressor'))
        self.evaluator_model_path = evaluator_model_path or str(config.get_model_path('evaluator'))
        
        # Context engineering settings
        self.knowledge_graph_path = knowledge_graph_path or config.get_api_config('knowledge_graph_path')
        self.max_token_budget = max_token_budget or config.get_api_config('max_token_budget')
        
        # Initialize components
        self._initialize_components()
        self._setup_routes()
        
        logger.info("✅ Context Engineered CBT API initialized successfully")
    
    def _initialize_components(self):
        """Initialize all API components"""
        try:
            # Initialize enhanced conversation manager
            self.enhanced_conversation_manager = EnhancedConversationManager(
                classifier_model_path=self.classifier_model_path,
                use_ml_classifier=True,
                knowledge_graph_path=self.knowledge_graph_path
            )
            
            # Initialize enhanced response generator  
            self.enhanced_response_generator = EnhancedResponseGenerator(
                use_ollama=self.use_ollama,
                ollama_model=self.ollama_model,
                ollama_base_url=self.ollama_base_url
            )
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get('/', tags=["Interface"])
        async def get_api_info():
            """Get API information"""
            return {
                "service": "Context Engineered CBT API with Authentication",
                "version": "2.1.0",
                "status": "operational",
                "environment": config.get_environment_info()['environment'],
                "authentication_required": config.pocketbase_manager is not None,
                "features": {
                    "original_cbt_logic": True,
                    "context_engineering": True,
                    "knowledge_graph": True,
                    "n8n_integration": self.enable_n8n,
                    "pocketbase_auth": config.pocketbase_manager is not None
                },
                "endpoints": {
                    "public": ["/", "/health", "/auth/signup", "/auth/login", "/database/health"],
                    "protected": ["/conversation/*", "/auth/profile/*"]
                }
            }
        
        @self.app.get('/health', response_model=HealthResponse, tags=["Health"])
        async def health_check():
            """Health check endpoint"""
            try:
                health_data = {
                    "status": "healthy",
                    "service": "Context Engineered CBT API"
                }
                
                # Add PocketBase health check if available
                if config.pocketbase_manager:
                    pb_health = await config.pocketbase_manager.health_check()
                    health_data["database"] = pb_health
                
                return HealthResponse(**health_data)
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail="Service unhealthy")
        
        # ========================================
        # PUBLIC AUTHENTICATION ENDPOINTS
        # ========================================
        
        @self.app.post('/auth/signup', response_model=UserAuthResponse, tags=["Authentication"])
        async def signup_user(request: UserSignupRequest):
            """Sign up a new user with email and password"""
            if not config.pocketbase_manager:
                raise HTTPException(
                    status_code=503, 
                    detail="Database not available. PocketBase is required for user authentication."
                )
            
            try:
                result = await config.pocketbase_manager.signup_user(request.model_dump())
                
                if result['success']:
                    logger.info(f"✅ User signup successful: {result['email']}")
                    return UserAuthResponse(**result)
                else:
                    logger.warning(f"⚠️ User signup failed: {result['message']}")
                    raise HTTPException(status_code=400, detail=result['message'])
                    
            except Exception as e:
                logger.error(f"❌ Signup error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error during signup")
        
        @self.app.post('/auth/login', response_model=UserAuthResponse, tags=["Authentication"])
        async def login_user(request: UserLoginRequest):
            """Login user with email and password"""
            if not config.pocketbase_manager:
                raise HTTPException(
                    status_code=503, 
                    detail="Database not available. PocketBase is required for user authentication."
                )
            
            try:
                result = await config.pocketbase_manager.login_user(request.email, request.password)
                
                if result['success']:
                    logger.info(f"✅ User login successful: {result['email']}")
                    return UserAuthResponse(**result)
                else:
                    logger.warning(f"⚠️ User login failed: {result['message']}")
                    raise HTTPException(status_code=401, detail=result['message'])
                    
            except Exception as e:
                logger.error(f"❌ Login error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error during login")
        
        # ========================================
        # PROTECTED AUTHENTICATION ENDPOINTS
        # ========================================
        
        @self.app.get('/auth/profile/{user_id}', response_model=UserProfileResponse, tags=["Authentication"])
        async def get_user_profile(user_id: str, current_user: dict = Depends(get_current_user)):
            """Get user profile by ID (Protected)"""
            if not config.pocketbase_manager:
                raise HTTPException(
                    status_code=503, 
                    detail="Database not available. PocketBase is required for user authentication."
                )
            
            try:
                user_data = await config.pocketbase_manager.get_user_by_id(user_id)
                
                if user_data:
                    return UserProfileResponse(**user_data)
                else:
                    raise HTTPException(status_code=404, detail="User not found")
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ Get profile error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post('/database/setup-collections', tags=["Admin"])
        async def setup_collections():
            """Manually create PocketBase collections (admin only)"""
            if not config.pocketbase_manager:
                raise HTTPException(
                    status_code=503,
                    detail="PocketBase not configured"
                )
            
            success = await config.pocketbase_manager.ensure_conversations_collection()
            if success:
                return {"message": "Collections created successfully"}
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create collections"
                )
        
        
        @self.app.get('/pocketbase/status', tags=["Admin"])
        async def pocketbase_status():
            """Check if PocketBase is running"""
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get("http://localhost:8090/api/health")
                    return {
                        "pocketbase_running": True,
                        "status_code": response.status_code,
                        "message": "PocketBase is accessible"
                    }
            except Exception as e:
                return {
                    "pocketbase_running": False,
                    "error": str(e),
                    "message": "Cannot connect to PocketBase on port 8090"
                }
        
        @self.app.get('/database/health', tags=["Database"])
        async def database_health():
            """Check database connection health"""
            if not config.pocketbase_manager:
                return {
                    "status": "unavailable",
                    "message": "PocketBase manager not initialized"
                }
            
            try:
                health_data = await config.pocketbase_manager.health_check()
                return health_data
            except Exception as e:
                logger.error(f"❌ Database health check failed: {e}")
                return {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # ========================================
        # FRONTEND PAGES
        # ========================================
        
        @self.app.get('/frontend/auth', tags=["Frontend"])
        async def auth_page():
            """Serve authentication page"""
            auth_file = Path(__file__).parent / "static" / "auth.html"
            if auth_file.exists():
                return HTMLResponse(auth_file.read_text())
            else:
                raise HTTPException(status_code=404, detail="Auth page not found")

        @self.app.get('/frontend/dashboard', tags=["Frontend"])
        async def dashboard_page():
            """Serve CBT API dashboard"""
            dashboard_file = Path(__file__).parent / "static" / "dashboard.html"
            if dashboard_file.exists():
                return HTMLResponse(dashboard_file.read_text())
            else:
                raise HTTPException(status_code=404, detail="Dashboard page not found")

        # ========================================
        # CLASSIFICATION ENDPOINTS
        # ========================================
        
        class ClassifyRequest(BaseModel):
            text: str = Field(..., description="Text to classify")
            threshold: float = Field(0.7, description="Confidence threshold for CBT trigger detection")
        
        @self.app.post('/classify/text', tags=["Classification"])
        async def classify_text(request: ClassifyRequest):
            """Classify if a text is CBT-triggering using the binary classifier API"""
            try:
                # Use the conversation manager's classifier
                conversation_manager = self.enhanced_conversation_manager.conversation_managers.get(
                    list(self.enhanced_conversation_manager.conversation_managers.keys())[0]
                ) if self.enhanced_conversation_manager.conversation_managers else None
                
                if not conversation_manager or not conversation_manager.use_ml_classifier:
                    # Create a temporary conversation manager just for classification
                    from cbt_logic import ConversationManager
                    temp_manager = ConversationManager(
                        use_ml_classifier=True,
                        classifier_model_path=self.classifier_model_path,
                        use_classifier_api=False
                    )
                    
                    if temp_manager.cbt_classifier:
                        result = temp_manager.cbt_classifier.predict(request.text, request.threshold)
                    else:
                        raise HTTPException(status_code=503, detail="Binary classifier not available")
                else:
                    # Use existing classifier
                    if conversation_manager.cbt_classifier:
                        result = conversation_manager.cbt_classifier.predict(request.text, request.threshold)
                    else:
                        raise HTTPException(status_code=503, detail="Binary classifier not available")
                
                return {
                    "text": request.text[:100] + "..." if len(request.text) > 100 else request.text,
                    "is_cbt_trigger": result['is_cbt_trigger'],
                    "confidence": result['confidence'],
                    "threshold": result['threshold']
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Classification error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ========================================
        # POCKETBASE PROXY
        # ========================================
        
        @self.app.api_route("/pb/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"], include_in_schema=False)
        async def pocketbase_proxy(request: Request, path: str):
            """Proxy requests to PocketBase admin and API"""
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    url = f"http://localhost:8090/{path}"
                    
                    # Get request body if present
                    body = await request.body() if request.method in ["POST", "PUT", "PATCH"] else None
                    
                    # Forward the request
                    response = await client.request(
                        method=request.method,
                        url=url,
                        content=body,
                        headers={
                            key: value for key, value in request.headers.items()
                            if key.lower() not in ["host", "content-length"]
                        }
                    )
                    
                    # Return the response
                    return Response(
                        content=response.content,
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )
            except httpx.ConnectError:
                return HTMLResponse(
                    content="""
                    <h2>PocketBase Connection Error</h2>
                    <p>Could not connect to PocketBase. This might be because:</p>
                    <ul>
                        <li>PocketBase is still starting up (wait a few seconds)</li>
                        <li>PocketBase failed to start (check logs)</li>
                        <li>Running in development without PocketBase</li>
                    </ul>
                    <p>Check the application logs for more details.</p>
                    """,
                    status_code=503
                )
        
        # ========================================
        # PROTECTED CBT CONVERSATION ENDPOINTS
        # ========================================
        
        @self.app.post('/conversation/start', response_model=StartConversationResponse, tags=["Conversation"])
        async def start_conversation(
            current_user: dict = Depends(get_current_user),
            user_id: str = None, 
            channel_info: Dict = None
        ):
            """Start a new conversation (Protected - requires authentication)"""
            logger.info(f"Starting new conversation for authenticated user: {current_user['user_id']}")
            
            try:
                # Use authenticated user's ID
                authenticated_user_id = current_user['user_id']
                
                # Start enhanced conversation
                conversation_data = await self.enhanced_conversation_manager.start_conversation(
                    user_id=authenticated_user_id, 
                    channel_info=channel_info
                )
                
                conversation_id = conversation_data["conversation_id"]
                
                # Generate initial response
                initial_response = "Hello! I'm here to support you through our conversation today. How are you feeling?"
                
                # Save conversation to database if possible
                if config.pocketbase_manager:
                    await config.pocketbase_manager.save_conversation(
                        authenticated_user_id, 
                        {
                            "conversation_id": conversation_id,
                            "phase": "chit_chat",
                            "initial_response": initial_response
                        }
                    )
                
                logger.info(f"✅ Started conversation {conversation_id} for user {authenticated_user_id}")
                
                return StartConversationResponse(
                    conversation_id=conversation_id,
                    response=initial_response,
                    phase="chit_chat"
                )
                
            except Exception as e:
                logger.error(f"Error starting conversation: {e}")
                raise HTTPException(status_code=500, detail=f"Error starting conversation: {str(e)}")
        
        @self.app.post('/conversation/message', response_model=SendMessageResponse, tags=["Conversation"])
        async def send_message(
            request: SendMessageRequest,
            current_user: dict = Depends(get_current_user)
        ):
            """Send a message in conversation (Protected - requires authentication)"""
            conversation_id = request.conversation_id
            user_message = request.message.strip()
            
            logger.info(f"Processing message for conversation {conversation_id} from user {current_user['user_id']}")
            
            try:
                # Process message through enhanced conversation manager
                response_data = await self.enhanced_conversation_manager.process_message_with_context(
                    user_input=user_message,
                    conversation_id=conversation_id,
                    llm_client=self.enhanced_response_generator
                )
                
                # Save updated conversation to database if possible
                if config.pocketbase_manager:
                    await config.pocketbase_manager.save_conversation(
                        current_user['user_id'],
                        {
                            "conversation_id": conversation_id,
                            "phase": response_data["phase"],
                            "last_message": user_message,
                            "last_response": response_data["response"],
                            "progress_scores": response_data.get("current_progress_scores")
                        }
                    )
                
                return SendMessageResponse(
                    conversation_id=conversation_id,
                    response=response_data["response"],
                    phase=response_data["phase"],
                    current_progress_scores=response_data.get("current_progress_scores")
                )
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")
        
        @self.app.get('/conversation/state', response_model=ConversationStateResponse, tags=["Conversation"])
        async def get_conversation_state(
            conversation_id: str = Query(...),
            current_user: dict = Depends(get_current_user)
        ):
            """Get conversation state (Protected - requires authentication)"""
            try:
                state_data = await self.enhanced_conversation_manager.get_enhanced_conversation_state(
                    conversation_id
                )
                
                return ConversationStateResponse(**state_data)
                
            except Exception as e:
                logger.error(f"Error getting conversation state: {e}")
                raise HTTPException(status_code=404, detail="Conversation not found")
        
        @self.app.post('/conversation/reset', response_model=ResetConversationResponse, tags=["Conversation"])
        async def reset_conversation(
            request: ResetConversationRequest,
            current_user: dict = Depends(get_current_user)
        ):
            """Reset conversation (Protected - requires authentication)"""
            try:
                reset_data = await self.enhanced_conversation_manager.reset_conversation(
                    request.conversation_id
                )
                
                return ResetConversationResponse(
                    conversation_id=reset_data["conversation_id"],
                    message="Conversation reset successfully"
                )
                
            except Exception as e:
                logger.error(f"Error resetting conversation: {e}")
                raise HTTPException(status_code=500, detail=f"Error resetting conversation: {str(e)}")

def create_app(enable_n8n: bool = False):
    """Create the FastAPI application (existing function)"""
    logger.info("🚀 Creating Context Engineered CBT API application")
    
    cbt_api = ContextEngineeredCBTAPI(enable_n8n=enable_n8n)
    
    # Setup static files if they exist
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        cbt_api.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    return cbt_api.app

def main():
    """Main entry point for AWS-ready CBT API (from run_context_engineered_cbt.py)"""
    
    logger.info("🚀 Starting CBT API with AWS-ready configuration")
    logger.info(f"Environment: {config.get_environment_info()['environment']}")
    logger.info(f"Project Root: {config.project_root}")
    
    
    # Import and create app
    app = create_app()
    
    # Get configuration
    host = config.get_api_config('api_host')
    port = config.get_api_config('api_port')
    
    # Run the server
    logger.info(f"🌐 Starting server on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=config.get_api_config('log_level').lower()
    )

if __name__ == "__main__":
    main()