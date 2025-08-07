"""
Utils - Configuration and Models with PostgreSQL Support
"""

import os
from pathlib import Path
from typing import Dict, Optional, List, Any, Union
import logging
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime, timezone, timedelta
import asyncio
import httpx
import json
import bcrypt
import jwt
import secrets
import uuid

logger = logging.getLogger(__name__)

# ========================================
# DEPLOYMENT CONFIGURATION
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
        
        # Initialize database managers
        self.postgresql_available = self._check_postgresql()
        
        logger.info(f"🚀 CBT API Configuration:")
        logger.info(f"   Environment: {'AWS' if self.is_aws else 'Docker' if self.is_docker else 'Local'}")
        logger.info(f"   Project Root: {self.project_root}")
        logger.info(f"   Model Paths: {self.model_paths}")
        logger.info(f"   PostgreSQL: {'Available' if self.postgresql_available else 'Not configured'}")
    
    def _check_postgresql(self) -> bool:
        """Check if PostgreSQL is available"""
        try:
            from sqlalchemy import create_engine
            if os.getenv('DATABASE_URL'):
                return True
        except ImportError:
            pass
        return False
    
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
        current = Path(__file__).parent.parent
        
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
            'n8n_base_url': os.environ.get('N8N_BASE_URL', None),
            'hf_api_token': os.environ.get('HF_API_TOKEN', ''),
            'sequence_regressor_hf_model': os.environ.get('SEQUENCE_REGRESSOR_HF_MODEL', 'SaitejaJate/cbt_sequence_regressor'),
            'evaluator_hf_model': os.environ.get('EVALUATOR_HF_MODEL', 'SaitejaJate/Regression_Evaluation'),
            'knowledge_graph_path': os.environ.get('KNOWLEDGE_GRAPH_PATH', 'cbt_knowledge_graph.json'),
            'max_token_budget': int(os.environ.get('MAX_TOKEN_BUDGET', '8192')),
            'log_level': os.environ.get('LOG_LEVEL', 'INFO'),
            'api_host': os.environ.get('API_HOST', '0.0.0.0'),
            'api_port': int(os.environ.get('API_PORT', '8000')),
            # JWT Configuration
            'jwt_secret': os.environ.get('JWT_SECRET', secrets.token_urlsafe(32)),
            'jwt_algorithm': os.environ.get('JWT_ALGORITHM', 'HS256'),
            'jwt_expiration_hours': int(os.environ.get('JWT_EXPIRATION_HOURS', '24')),
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
            'postgresql_available': self.postgresql_available,
        }

# ========================================
# POSTGRESQL DATABASE SUPPORT
# ========================================

try:
    from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, Boolean, Float, ForeignKey
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship, Session
    from sqlalchemy.pool import NullPool
    SQLALCHEMY_AVAILABLE = True
    logger.info("✅ SQLAlchemy available for PostgreSQL")
except ImportError:
    logger.warning("⚠️ SQLAlchemy not available. Install with: pip install sqlalchemy psycopg2-binary")
    SQLALCHEMY_AVAILABLE = False
    Base = None

# PostgreSQL Configuration
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# Create SQLAlchemy components if available
if SQLALCHEMY_AVAILABLE and DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL, poolclass=NullPool, echo=False)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base = declarative_base()
        logger.info("✅ PostgreSQL connection configured")
    except Exception as e:
        logger.error(f"❌ Failed to configure PostgreSQL: {e}")
        engine = None
        SessionLocal = None
        Base = declarative_base() if SQLALCHEMY_AVAILABLE else None
else:
    engine = None
    SessionLocal = None
    if SQLALCHEMY_AVAILABLE:
        Base = declarative_base()

# Database Models (only if SQLAlchemy is available)
if SQLALCHEMY_AVAILABLE:
    class DBUser(Base):
        __tablename__ = 'db_users'
        
        id = Column(String, primary_key=True)
        email = Column(String, unique=True, nullable=False)
        username = Column(String, unique=True)
        password_hash = Column(String, nullable=False)
        first_name = Column(String)
        last_name = Column(String)
        date_of_birth = Column(String)
        phone = Column(String)
        emergency_contact = Column(JSON)
        preferences = Column(JSON)
        account_status = Column(String, default='active')
        created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
        last_login = Column(DateTime)
        user_metadata = Column(JSON)
        knowledge_graph = Column(JSON, nullable=True, default=lambda: {"entities": {}, "relations": []})
        
        conversations = relationship("DBConversation", back_populates="user", cascade="all, delete-orphan")

    class DBConversation(Base):
        __tablename__ = 'db_conversations'
        
        id = Column(String, primary_key=True)
        user_id = Column(String, ForeignKey('db_users.id'), nullable=False)
        started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
        ended_at = Column(DateTime)
        current_phase = Column(String, default='chit_chat')
        progress_scores = Column(JSON)
        user_metadata = Column(JSON)
        
        user = relationship("DBUser", back_populates="conversations")
        messages = relationship("DBMessage", back_populates="conversation", cascade="all, delete-orphan")

    class DBMessage(Base):
        __tablename__ = 'db_messages'
        
        id = Column(String, primary_key=True)
        conversation_id = Column(String, ForeignKey('db_conversations.id'), nullable=False)
        timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
        role = Column(String, nullable=False)  # 'user' or 'assistant'
        content = Column(Text, nullable=False)
        phase = Column(String)
        is_cbt_trigger = Column(Boolean, default=False)
        trigger_confidence = Column(Float)
        
        conversation = relationship("DBConversation", back_populates="messages")

    # Authentication and Database Manager
    class PostgreSQLManager:
        """Manager for PostgreSQL database operations with authentication"""
        
        @staticmethod
        def init_db():
            """Initialize database tables"""
            if engine:
                Base.metadata.create_all(bind=engine)
                logger.info("✅ PostgreSQL tables created successfully")
            else:
                logger.warning("⚠️ No PostgreSQL connection available")
        
        @staticmethod
        def get_db():
            """Get database session"""
            if SessionLocal:
                db = SessionLocal()
                try:
                    yield db
                finally:
                    db.close()
            else:
                yield None
        
        @staticmethod
        def hash_password(password: str) -> str:
            """Hash a password using bcrypt"""
            return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        @staticmethod
        def verify_password(password: str, hashed: str) -> bool:
            """Verify a password against a hash"""
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        
        @staticmethod
        def generate_jwt_token(user_id: str, email: str, config: DeploymentConfig) -> str:
            """Generate JWT token for user"""
            expiration = datetime.now(timezone.utc) + timedelta(hours=config.get_api_config('jwt_expiration_hours'))
            payload = {
                'user_id': user_id,
                'email': email,
                'exp': expiration,
                'iat': datetime.now(timezone.utc)
            }
            return jwt.encode(payload, config.get_api_config('jwt_secret'), algorithm=config.get_api_config('jwt_algorithm'))
        
        @staticmethod
        def decode_jwt_token(token: str, config: DeploymentConfig) -> Optional[Dict[str, Any]]:
            """Decode and verify JWT token"""
            try:
                payload = jwt.decode(token, config.get_api_config('jwt_secret'), algorithms=[config.get_api_config('jwt_algorithm')])
                return payload
            except jwt.ExpiredSignatureError:
                logger.warning("JWT token expired")
                return None
            except jwt.InvalidTokenError:
                logger.warning("Invalid JWT token")
                return None
        
        @staticmethod
        async def signup_user(db: Session, user_data: Dict[str, Any], config: DeploymentConfig) -> Dict[str, Any]:
            """Sign up a new user"""
            try:
                # Check if user already exists
                existing_user = db.query(DBUser).filter(DBUser.email == user_data['email']).first()
                if existing_user:
                    return {
                        'success': False,
                        'message': 'User with this email already exists'
                    }
                
                # Create new user
                user_id = str(uuid.uuid4())
                hashed_password = PostgreSQLManager.hash_password(user_data['password'])
                
                new_user = DBUser(
                    id=user_id,
                    email=user_data['email'],
                    password_hash=hashed_password,
                    first_name=user_data.get('first_name', ''),
                    last_name=user_data.get('last_name', ''),
                    date_of_birth=user_data.get('date_of_birth'),
                    phone=user_data.get('phone'),
                    emergency_contact=user_data.get('emergency_contact'),
                    preferences=user_data.get('preferences', {}),
                    account_status='active'
                )
                
                db.add(new_user)
                db.commit()
                db.refresh(new_user)
                
                # Generate JWT token
                token = PostgreSQLManager.generate_jwt_token(user_id, user_data['email'], config)
                
                logger.info(f"✅ User created successfully: {user_id}")
                
                return {
                    'success': True,
                    'user_id': user_id,
                    'email': user_data['email'],
                    'token': token,
                    'message': 'User created successfully'
                }
                
            except Exception as e:
                logger.error(f"❌ User signup failed: {e}")
                db.rollback()
                return {
                    'success': False,
                    'error': str(e),
                    'message': 'Failed to create user account'
                }
        
        @staticmethod
        async def login_user(db: Session, email: str, password: str, config: DeploymentConfig) -> Dict[str, Any]:
            """Authenticate user login"""
            try:
                # Find user by email
                user = db.query(DBUser).filter(DBUser.email == email).first()
                
                if not user:
                    return {
                        'success': False,
                        'message': 'Invalid email or password'
                    }
                
                # Verify password
                if not PostgreSQLManager.verify_password(password, user.password_hash):
                    return {
                        'success': False,
                        'message': 'Invalid email or password'
                    }
                
                # Update last login
                user.last_login = datetime.now(timezone.utc)
                db.commit()
                
                # Generate JWT token
                token = PostgreSQLManager.generate_jwt_token(user.id, user.email, config)
                
                logger.info(f"✅ User login successful: {user.id}")
                
                return {
                    'success': True,
                    'user_id': user.id,
                    'email': user.email,
                    'token': token,
                    'user_data': {
                        'first_name': user.first_name or '',
                        'last_name': user.last_name or '',
                        'account_status': user.account_status
                    },
                    'message': 'Login successful'
                }
                
            except Exception as e:
                logger.error(f"❌ User login failed: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'message': 'Login failed'
                }
        
        @staticmethod
        async def get_user_by_id(db: Session, user_id: str) -> Optional[Dict[str, Any]]:
            """Get user by ID"""
            try:
                user = db.query(DBUser).filter(DBUser.id == user_id).first()
                if user:
                    return {
                        'id': user.id,
                        'email': user.email,
                        'first_name': user.first_name or '',
                        'last_name': user.last_name or '',
                        'account_status': user.account_status,
                        'created': user.created_at.isoformat(),
                        'updated': user.last_login.isoformat() if user.last_login else None
                    }
                return None
            except Exception as e:
                logger.error(f"Failed to get user: {e}")
                return None
        
        @staticmethod
        def save_conversation_to_db(db: Session, conversation_id: str, user_id: str, phase: str = 'chit_chat'):
            """Save a new conversation to PostgreSQL"""
            if not db:
                return None
            
            try:
                conv = DBConversation(id=conversation_id, user_id=user_id, current_phase=phase)
                db.add(conv)
                db.commit()
                db.refresh(conv)
                return conv
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")
                db.rollback()
                return None
        
        @staticmethod
        def save_message_to_db(db: Session, message_id: str, conversation_id: str, role: str, content: str, phase: str = None):
            """Save a message to PostgreSQL"""
            if not db:
                return None
            
            try:
                msg = DBMessage(
                    id=message_id,
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    phase=phase
                )
                db.add(msg)
                db.commit()
                db.refresh(msg)
                return msg
            except Exception as e:
                logger.error(f"Failed to save message: {e}")
                db.rollback()
                return None

    # Initialize tables on startup if PostgreSQL is available
    if engine:
        try:
            PostgreSQLManager.init_db()
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL tables: {e}")

# Global configuration instance
config = DeploymentConfig()

# ========================================
# PYDANTIC MODELS
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
    updated: Optional[str] = None

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