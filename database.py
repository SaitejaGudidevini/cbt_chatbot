"""
PostgreSQL Database Configuration and Models
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import NullPool
import logging

logger = logging.getLogger(__name__)

# Get database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL')

# Create engine
if DATABASE_URL:
    # Railway provides DATABASE_URL in postgres:// format, SQLAlchemy needs postgresql://
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    
    engine = create_engine(
        DATABASE_URL,
        poolclass=NullPool,  # Recommended for serverless/Railway
        echo=False
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    logger.warning("DATABASE_URL not found. Database features will be disabled.")
    engine = None
    SessionLocal = None

# Create base class for models
Base = declarative_base()

# Define models
class User(Base):
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    current_phase = Column(String, default='chit_chat')
    progress_scores = Column(JSON)
    metadata = Column(JSON)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    phase_transitions = relationship("PhaseTransition", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey('conversations.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    phase = Column(String)
    is_cbt_trigger = Column(Boolean, default=False)
    trigger_confidence = Column(Float)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

class PhaseTransition(Base):
    __tablename__ = 'phase_transitions'
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey('conversations.id'), nullable=False)
    from_phase = Column(String)
    to_phase = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    trigger_message = Column(Text)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="phase_transitions")

# Database initialization
def init_db():
    """Initialize database tables"""
    if engine:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    else:
        logger.warning("Cannot initialize database - no connection available")

# Dependency to get DB session
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

# Database helper functions
class DatabaseManager:
    """Manager class for database operations"""
    
    @staticmethod
    def create_user(db: Session, user_id: str, email: str, username: str = None):
        """Create a new user"""
        user = User(id=user_id, email=email, username=username)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def get_user(db: Session, user_id: str):
        """Get user by ID"""
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def create_conversation(db: Session, conversation_id: str, user_id: str):
        """Create a new conversation"""
        conversation = Conversation(id=conversation_id, user_id=user_id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation
    
    @staticmethod
    def save_message(db: Session, message_id: str, conversation_id: str, role: str, content: str, phase: str = None, is_cbt_trigger: bool = False, trigger_confidence: float = None):
        """Save a message"""
        message = Message(
            id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            phase=phase,
            is_cbt_trigger=is_cbt_trigger,
            trigger_confidence=trigger_confidence
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        return message
    
    @staticmethod
    def update_conversation_phase(db: Session, conversation_id: str, new_phase: str, trigger_message: str = None):
        """Update conversation phase"""
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conversation:
            old_phase = conversation.current_phase
            conversation.current_phase = new_phase
            
            # Record phase transition
            if old_phase != new_phase:
                import uuid
                transition = PhaseTransition(
                    id=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    from_phase=old_phase,
                    to_phase=new_phase,
                    trigger_message=trigger_message
                )
                db.add(transition)
            
            db.commit()
            return conversation
        return None
    
    @staticmethod
    def get_conversation_history(db: Session, conversation_id: str):
        """Get all messages for a conversation"""
        return db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.timestamp).all()