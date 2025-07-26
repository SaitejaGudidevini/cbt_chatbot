"""
Utils - Configuration and Models
From: config/deployment_config.py + models.py
"""

import os
from pathlib import Path
from typing import Dict, Optional, List, Any, Union
import logging
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
import asyncio
import httpx
import json

logger = logging.getLogger(__name__)

# ========================================
# POCKETBASE IMPORTS AND TYPES
# ========================================

try:
    from pocketbase import PocketBase
    from pocketbase.models import Record
    from pocketbase.client import ClientResponseError
    POCKETBASE_AVAILABLE = True
    logger.info("✅ PocketBase client available")
except ImportError:
    logger.warning("⚠️ PocketBase client not available. Install with: pip install pocketbase")
    POCKETBASE_AVAILABLE = False
    PocketBase = None
    Record = None
    ClientResponseError = Exception

# ========================================
# DEPLOYMENT CONFIGURATION (from config/deployment_config.py)
# ========================================

class DeploymentConfig:
    """Environment-aware configuration for CBT API deployment"""
    
    def __init__(self):
        # Detect environment
        self.is_aws = self._is_aws_environment()
        self.is_docker = self._is_docker_environment()
        self.is_local = not (self.is_aws or self.is_docker)
        
        # Set project root based on environment
        self.project_root = self._determine_project_root()
        
        # Initialize paths
        self.model_paths = self._initialize_model_paths()
        self.api_config = self._initialize_api_config()
        
        # Initialize PocketBase manager
        self.pocketbase_manager = None
        if POCKETBASE_AVAILABLE:
            self.pocketbase_manager = PocketBaseManager(self.api_config)
        
        logger.info(f"🚀 CBT API Configuration:")
        logger.info(f"   Environment: {'AWS' if self.is_aws else 'Docker' if self.is_docker else 'Local'}")
        logger.info(f"   Project Root: {self.project_root}")
        logger.info(f"   Model Paths: {self.model_paths}")
        logger.info(f"   PocketBase: {'Enabled' if self.pocketbase_manager else 'Disabled'}")
    
    def _is_aws_environment(self) -> bool:
        """Detect if running in AWS environment"""
        aws_indicators = [
            'AWS_EXECUTION_ENV',
            'AWS_LAMBDA_FUNCTION_NAME',
            'AWS_REGION',
            'ECS_CONTAINER_METADATA_URI',
        ]
        return any(os.environ.get(indicator) for indicator in aws_indicators)
    
    def _is_docker_environment(self) -> bool:
        """Detect if running in Docker container"""
        return os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'
    
    def _determine_project_root(self) -> Path:
        """Determine project root based on environment"""
        # 1. Check environment variable first
        if env_root := os.environ.get('CBTAPI_ROOT'):
            return Path(env_root)
        
        # 2. AWS/Docker: Use /app
        if self.is_aws or self.is_docker:
            return Path('/app')
        
        # 3. Local development: Find project root
        current = Path(__file__).parent.parent  # Go up from config/ to CBT_context_engineering/
        
        # Look for parent directory with cbt_api.py (original root)
        for parent in [current.parent] + list(current.parent.parents):
            if (parent / 'cbt_api.py').exists():
                return parent
        
        # Fallback to current directory
        return current
    
    def _initialize_model_paths(self) -> Dict[str, Path]:
        """Initialize ML model paths"""
        base_paths = {
            'classifier': 'BinaryClassifier/cbt_classifier',
            'sequence_regressor': 'cbt_sequence_regressor/cbt_sequence_model',
            'evaluator': 'RegressionEvaluation/cbt_evaluator_simple'
        }
        
        model_paths = {}
        for model_name, relative_path in base_paths.items():
            # Check environment variable first
            env_var = f'CBTAPI_{model_name.upper()}_PATH'
            if env_path := os.environ.get(env_var):
                model_paths[model_name] = Path(env_path)
            else:
                model_paths[model_name] = self.project_root / relative_path
        
        return model_paths
    
    def _initialize_api_config(self) -> Dict[str, str]:
        """Initialize API configuration"""
        return {
            'ollama_base_url': os.environ.get('OLLAMA_BASE_URL', 'http://100.111.94.76:11434'),
            'ollama_model': os.environ.get('OLLAMA_MODEL', 'qwen2.5:14b-instruct'),
            'n8n_base_url': os.environ.get('N8N_BASE_URL', None),  # Disabled by default
            'knowledge_graph_path': os.environ.get('KNOWLEDGE_GRAPH_PATH', 'cbt_knowledge_graph.json'),
            'max_token_budget': int(os.environ.get('MAX_TOKEN_BUDGET', '8192')),
            'log_level': os.environ.get('LOG_LEVEL', 'INFO'),
            'api_host': os.environ.get('API_HOST', '0.0.0.0'),
            'api_port': int(os.environ.get('API_PORT', '4001')),
            # PocketBase Configuration
            'pocketbase_url': os.environ.get('POCKETBASE_URL', 'http://127.0.0.1:8090'),
            'pocketbase_admin_email': os.environ.get('POCKETBASE_ADMIN_EMAIL', 'admin@cbtapi.com'),
            'pocketbase_admin_password': os.environ.get('POCKETBASE_ADMIN_PASSWORD', 'admin123456'),
            'pocketbase_auto_create_collections': os.environ.get('POCKETBASE_AUTO_CREATE_COLLECTIONS', 'true').lower() == 'true',
        }
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path for specific model"""
        if model_name not in self.model_paths:
            raise ValueError(f"Unknown model: {model_name}")
        return self.model_paths[model_name]
    
    def get_api_config(self, key: str) -> str:
        """Get API configuration value"""
        return self.api_config.get(key)
    
    def get_sys_path(self) -> str:
        """Get the path to add to sys.path for imports"""
        return str(self.project_root)
    
    def validate_paths(self) -> Dict[str, bool]:
        """Validate that all required paths exist"""
        validation_results = {}
        
        # Check project root
        validation_results['project_root'] = self.project_root.exists()
        
        # Check model paths
        for model_name, path in self.model_paths.items():
            validation_results[f'model_{model_name}'] = path.exists()
        
        return validation_results
    
    def get_environment_info(self) -> Dict[str, any]:
        """Get comprehensive environment information"""
        return {
            'environment': 'AWS' if self.is_aws else 'Docker' if self.is_docker else 'Local',
            'project_root': str(self.project_root),
            'model_paths': {k: str(v) for k, v in self.model_paths.items()},
            'api_config': self.api_config,
            'path_validation': self.validate_paths(),
            'python_path': os.environ.get('PYTHONPATH', ''),
            'working_directory': str(Path.cwd()),
            'pocketbase_enabled': self.pocketbase_manager is not None,
        }

# ========================================
# POCKETBASE DATABASE MANAGER
# ========================================

class PocketBaseManager:
    """Manages PocketBase database operations for CBT API"""
    
    def __init__(self, api_config: Dict[str, str]):
        if not POCKETBASE_AVAILABLE:
            raise ImportError("PocketBase client not available. Install with: pip install pocketbase")
        
        self.url = api_config.get('pocketbase_url')
        self.admin_email = api_config.get('pocketbase_admin_email')
        self.admin_password = api_config.get('pocketbase_admin_password')
        self.auto_create_collections = api_config.get('pocketbase_auto_create_collections', True)
        
        # Initialize PocketBase client
        self.pb = PocketBase(self.url)
        self._initialized = False
        
        logger.info(f"🗄️ PocketBase Manager initialized - URL: {self.url}")
    
    async def initialize(self) -> bool:
        """Initialize PocketBase connection and collections"""
        try:
            # Authenticate as admin
            await self._authenticate_admin()
            
            # Create collections if needed
            if self.auto_create_collections:
                await self._create_collections()
            
            self._initialized = True
            logger.info("✅ PocketBase initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize PocketBase: {e}")
            return False
    
    async def _authenticate_admin(self):
        """Authenticate as admin user"""
        try:
            self.pb.admins.auth_with_password(self.admin_email, self.admin_password)
            logger.info("✅ PocketBase admin authentication successful")
        except Exception as e:
            logger.error(f"❌ PocketBase admin authentication failed: {e}")
            raise
    
    async def _create_collections(self):
        """Create necessary collections for CBT API"""
        try:
            # Users collection schema
            users_schema = {
                "name": "users",
                "type": "auth",
                "schema": [
                    {
                        "name": "first_name",
                        "type": "text",
                        "required": True,
                    },
                    {
                        "name": "last_name", 
                        "type": "text",
                        "required": True,
                    },
                    {
                        "name": "date_of_birth",
                        "type": "date",
                        "required": False,
                    },
                    {
                        "name": "phone",
                        "type": "text",
                        "required": False,
                    },
                    {
                        "name": "emergency_contact",
                        "type": "json",
                        "required": False,
                    },
                    {
                        "name": "preferences",
                        "type": "json",
                        "required": False,
                    },
                    {
                        "name": "account_status",
                        "type": "select",
                        "required": True,
                        "options": {
                            "values": ["active", "inactive", "suspended", "pending_verification"]
                        }
                    }
                ],
                "options": {
                    "allowEmailAuth": True,
                    "allowUsernameAuth": False,
                    "allowOAuth2Auth": False,
                    "requireEmail": True,
                    "exceptEmailDomains": [],
                    "onlyEmailDomains": [],
                    "minPasswordLength": 8
                }
            }
            
            # Try to create users collection
            try:
                self.pb.collections.create(users_schema)
                logger.info("✅ Created 'users' collection")
            except ClientResponseError as e:
                if "already exists" in str(e).lower():
                    logger.info("ℹ️ 'users' collection already exists")
                else:
                    raise
            
            # Conversations collection schema
            conversations_schema = {
                "name": "conversations",
                "type": "base",
                "schema": [
                    {
                        "name": "user_id",
                        "type": "text",
                        "required": True,
                        "options": {
                            "min": 1,
                            "max": 50,
                            "pattern": ""
                        }
                    },
                    {
                        "name": "conversation_data",
                        "type": "json",
                        "required": True,
                    },
                    {
                        "name": "phase",
                        "type": "text",
                        "required": True,
                    },
                    {
                        "name": "status",
                        "type": "select",
                        "required": True,
                        "options": {
                            "values": ["active", "completed", "paused", "archived"]
                        }
                    },
                    {
                        "name": "last_activity",
                        "type": "date",
                        "required": True,
                    }
                ]
            }
            
            try:
                self.pb.collections.create(conversations_schema)
                logger.info("✅ Created 'conversations' collection")
            except ClientResponseError as e:
                if "already exists" in str(e).lower():
                    logger.info("ℹ️ 'conversations' collection already exists")
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"❌ Failed to create collections: {e}")
            raise
    
    async def signup_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sign up a new user"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create user record
            user_record = self.pb.collection('users').create({
                'email': user_data['email'],
                'password': user_data['password'],
                'passwordConfirm': user_data['password'],
                'first_name': user_data.get('first_name', ''),
                'last_name': user_data.get('last_name', ''),
                'date_of_birth': user_data.get('date_of_birth'),
                'phone': user_data.get('phone'),
                'emergency_contact': user_data.get('emergency_contact'),
                'preferences': user_data.get('preferences', {}),
                'account_status': 'pending_verification'
            })
            
            logger.info(f"✅ User created successfully: {user_record.id}")
            
            return {
                'success': True,
                'user_id': user_record.id,
                'email': user_data['email'],  # Use input data instead of record
                'message': 'User created successfully'
            }
            
        except ClientResponseError as e:
            logger.error(f"❌ User signup failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to create user account'
            }
    
    async def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate user login"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Authenticate user
            auth_data = self.pb.collection('users').auth_with_password(email, password)
            
            # Access user data through authStore.model
            user_model = self.pb.auth_store.model
            
            logger.info(f"✅ User login successful: {user_model.id}")
            
            return {
                'success': True,
                'user_id': user_model.id,
                'email': user_model.email,
                'token': auth_data.token,
                'user_data': {
                    'first_name': getattr(user_model, 'first_name', ''),
                    'last_name': getattr(user_model, 'last_name', ''),
                    'account_status': getattr(user_model, 'account_status', 'active')
                },
                'message': 'Login successful'
            }
            
        except ClientResponseError as e:
            logger.error(f"❌ User login failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Invalid email or password'
            }
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        if not self._initialized:
            await self.initialize()
        
        try:
            user_record = self.pb.collection('users').get_one(user_id)
            return {
                'id': user_record.id,
                'email': user_record.email,
                'first_name': getattr(user_record, 'first_name', ''),
                'last_name': getattr(user_record, 'last_name', ''),
                'account_status': getattr(user_record, 'account_status', 'active'),
                'created': user_record.created,
                'updated': user_record.updated
            }
        except ClientResponseError:
            return None
    
    async def save_conversation(self, user_id: str, conversation_data: Dict[str, Any]) -> bool:
        """Save conversation data to database"""
        if not self._initialized:
            await self.initialize()
        
        try:
            self.pb.collection('conversations').create({
                'user_id': user_id,
                'conversation_data': conversation_data,
                'phase': conversation_data.get('phase', 'chit_chat'),
                'status': 'active',
                'last_activity': datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save conversation: {e}")
            return False
    
    async def ensure_conversations_collection(self) -> bool:
        """Ensure conversations collection exists - can be called manually"""
        try:
            # Try to get the collection first
            try:
                self.pb.collections.get_one('conversations')
                logger.info("✅ Conversations collection already exists")
                return True
            except:
                # Collection doesn't exist, create it
                pass
            
            # Create the collection
            conversations_schema = {
                "name": "conversations",
                "type": "base",
                "schema": [
                    {
                        "name": "user_id",
                        "type": "text",
                        "required": True,
                        "options": {
                            "min": 1,
                            "max": 50,
                            "pattern": ""
                        }
                    },
                    {
                        "name": "conversation_data",
                        "type": "json",
                        "required": True,
                    },
                    {
                        "name": "phase",
                        "type": "text",
                        "required": True,
                    },
                    {
                        "name": "status",
                        "type": "select",
                        "required": True,
                        "options": {
                            "values": ["active", "completed", "archived"]
                        }
                    },
                    {
                        "name": "last_activity",
                        "type": "date",
                        "required": True,
                    }
                ],
                "listRule": "user_id = @request.auth.id",
                "viewRule": "user_id = @request.auth.id",
                "createRule": "",
                "updateRule": "user_id = @request.auth.id",
                "deleteRule": "user_id = @request.auth.id"
            }
            
            self.pb.collections.create(conversations_schema)
            logger.info("✅ Created conversations collection successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to ensure conversations collection: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check PocketBase connection health"""
        try:
            # Simple health check by making a request to the API
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.url}/api/health")
                return {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'url': self.url,
                    'response_code': response.status_code
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'url': self.url,
                'error': str(e)
            }

# Global configuration instance
config = DeploymentConfig()

# ========================================
# PYDANTIC MODELS (from models.py)
# ========================================

# User Authentication Models
class UserSignupRequest(BaseModel):
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=8, description="User's password (minimum 8 characters)")
    first_name: str = Field(..., min_length=1, description="User's first name")
    last_name: str = Field(..., min_length=1, description="User's last name")
    date_of_birth: Optional[str] = Field(None, description="User's date of birth (YYYY-MM-DD)")
    phone: Optional[str] = Field(None, description="User's phone number")
    emergency_contact: Optional[Dict[str, str]] = Field(None, description="Emergency contact information")
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User preferences")

class UserLoginRequest(BaseModel):
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")

class UserAuthResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[str] = None
    email: Optional[str] = None
    token: Optional[str] = None
    user_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class UserProfileResponse(BaseModel):
    id: str
    email: str
    first_name: str
    last_name: str
    account_status: str
    created: str
    updated: str

# Existing CBT API Models
class StartConversationResponse(BaseModel):
    conversation_id: str
    response: str
    phase: str

class SendMessageRequest(BaseModel):
    conversation_id: str
    message: str = Field(..., min_length=1, description="The user's message")

class SendMessageResponse(BaseModel):
    conversation_id: str
    response: str
    phase: str
    current_progress_scores: Optional[Dict[str, float]] = None

class ConversationStateResponse(BaseModel):
    conversation_id: str
    phase: str
    conversation_history: List[Dict[str, Any]]
    cbt_trigger_detected: bool
    trigger_statement: Optional[str] = None
    current_progress_scores: Optional[Dict[str, float]] = None
    progress_summary: Optional[Dict[str, Any]] = None

class ResetConversationRequest(BaseModel):
    conversation_id: str

class ResetConversationResponse(BaseModel):
    conversation_id: str
    response: str
    phase: str
    message: str

class ExportConversationResponse(BaseModel):
    conversation_id: str
    conversation_history: List[Dict[str, Any]]
    cbt_trigger_detected: bool
    trigger_statement: Optional[str] = None
    export_timestamp: str

class HealthResponse(BaseModel):
    status: str
    service: str

class ErrorResponse(BaseModel):
    error: str

class ConfigRequest(BaseModel):
    hf_endpoint_url: str
    hf_api_token: str

class ConfigResponse(BaseModel):
    message: str
    status: str

class FlowTransition(BaseModel):
    conversation_id: str
    from_phase: str
    to_phase: str
    trigger_statement: str
    timestamp: str
    user_message_count: int
    time_in_previous_phase: float  # seconds

class FlowAnalytics(BaseModel):
    total_conversations: int
    chit_chat_only: int
    transitioned_to_cbt: int
    transition_rate: float
    average_messages_before_trigger: float
    common_triggers: List[Dict[str, Any]]

class FlowMappingResponse(BaseModel):
    conversation_flows: List[Dict[str, Any]]
    analytics: FlowAnalytics