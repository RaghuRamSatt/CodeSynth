"""
Database Manager - Handles database connections and operations
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import datetime
import uuid

import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func

# Configure logging
logger = logging.getLogger(__name__)

# SQLAlchemy Models
Base = declarative_base()

class User(Base):
    """User model for authentication."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    datasets = relationship("Dataset", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"

class Dataset(Base):
    """Dataset model for tracking user datasets."""
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    name = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    size_bytes = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.now())
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="datasets")
    conversation_datasets = relationship("ConversationDataset", back_populates="dataset")
    
    def __repr__(self):
        return f"<Dataset(name='{self.name}', type='{self.file_type}')>"

class Conversation(Base):
    """Conversation model for tracking user-agent interactions."""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")
    conversation_datasets = relationship("ConversationDataset", back_populates="conversation")
    
    def __repr__(self):
        return f"<Conversation(title='{self.title}')>"

class Message(Base):
    """Message model for storing conversation messages."""
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    role = Column(String(50), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    message_type = Column(String(50), default='text')  # 'text', 'code', 'image', etc.
    created_at = Column(DateTime, default=func.now())
    
    # For code messages
    code = Column(Text, nullable=True)
    execution_results = Column(JSON, nullable=True)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(role='{self.role}', type='{self.message_type}')>"

class ConversationDataset(Base):
    """Join table for conversations and datasets."""
    __tablename__ = 'conversation_datasets'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    
    # Relationships
    conversation = relationship("Conversation", back_populates="conversation_datasets")
    dataset = relationship("Dataset", back_populates="conversation_datasets")
    
    def __repr__(self):
        return f"<ConversationDataset(conversation_id={self.conversation_id}, dataset_id={self.dataset_id})>"

class ModelEvaluation(Base):
    """Model for tracking model performance evaluations."""
    __tablename__ = 'model_evaluations'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    benchmark_name = Column(String(100), nullable=False)
    task_type = Column(String(100), nullable=False)
    metrics = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<ModelEvaluation(model='{self.model_name}', benchmark='{self.benchmark_name}')>"

class DatabaseManager:
    """
    Manages database connections and operations.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the database manager with configuration settings.
        
        Args:
            config_path: Path to the configuration file
        """
        import yaml
        
        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        self.db_config = config.get("database", {})
        self.host = os.getenv("MYSQL_HOST", self.db_config.get("host", "localhost"))
        self.user = os.getenv("MYSQL_USER", self.db_config.get("user", "root"))
        self.password = os.getenv("MYSQL_PASSWORD", self.db_config.get("password", ""))
        self.database = os.getenv("MYSQL_DATABASE", self.db_config.get("database", "data_analysis_agent"))
        self.port = self.db_config.get("port", 3306)
        
        self.engine = None
        self.Session = None
        
    def initialize(self) -> bool:
        """
        Initialize the database connection and create tables if they don't exist.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Create database if it doesn't exist
            self._create_database_if_not_exists()
            
            # Create SQLAlchemy engine
            connection_string = f"mysql+mysqlconnector://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            self.engine = create_engine(connection_string, pool_size=5, max_overflow=10)
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(self.engine)
            
            logger.info("Database initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def _create_database_if_not_exists(self) -> None:
        """Create the database if it doesn't exist."""
        try:
            # Connect to MySQL server without specifying a database
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password
            )
            
            cursor = connection.cursor()
            
            # Create database if it doesn't exist
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            
            cursor.close()
            connection.close()
            
            logger.info(f"Ensured database '{self.database}' exists")
        except Error as e:
            logger.error(f"Error creating database: {e}")
            raise
    
    def save_conversation(self, user_id: int, title: str, messages: List[Dict[str, Any]], 
                         dataset_ids: List[int] = None) -> int:
        """
        Save a conversation to the database.
        
        Args:
            user_id: User ID
            title: Conversation title
            messages: List of message dictionaries
            dataset_ids: List of dataset IDs associated with the conversation
            
        Returns:
            Conversation ID
        """
        try:
            session = self.Session()
            
            # Create conversation
            conversation = Conversation(
                user_id=user_id,
                title=title
            )
            session.add(conversation)
            session.flush()  # Get conversation ID
            
            # Add messages
            for msg in messages:
                message = Message(
                    conversation_id=conversation.id,
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    message_type=msg.get("type", "text"),
                    code=msg.get("code") if msg.get("type") == "code" else None,
                    execution_results=msg.get("execution_results") if msg.get("type") == "code" else None
                )
                session.add(message)
            
            # Link datasets
            if dataset_ids:
                for dataset_id in dataset_ids:
                    link = ConversationDataset(
                        conversation_id=conversation.id,
                        dataset_id=dataset_id
                    )
                    session.add(link)
            
            session.commit()
            conversation_id = conversation.id
            session.close()
            
            return conversation_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving conversation: {e}")
            raise
    
    def get_conversation(self, conversation_id: int) -> Dict[str, Any]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation dictionary
        """
        try:
            session = self.Session()
            
            # Get conversation
            conversation = session.query(Conversation).filter_by(id=conversation_id).first()
            
            if not conversation:
                session.close()
                return None
            
            # Get messages
            messages = session.query(Message).filter_by(conversation_id=conversation_id).order_by(Message.created_at).all()
            
            # Get datasets
            dataset_links = session.query(ConversationDataset).filter_by(conversation_id=conversation_id).all()
            dataset_ids = [link.dataset_id for link in dataset_links]
            datasets = session.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all() if dataset_ids else []
            
            result = {
                "id": conversation.id,
                "user_id": conversation.user_id,
                "title": conversation.title,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "messages": [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "type": msg.message_type,
                        "code": msg.code,
                        "execution_results": msg.execution_results,
                        "created_at": msg.created_at
                    }
                    for msg in messages
                ],
                "datasets": [
                    {
                        "id": ds.id,
                        "name": ds.name,
                        "file_path": ds.file_path,
                        "file_type": ds.file_type,
                        "metadata": ds.metadata
                    }
                    for ds in datasets
                ]
            }
            
            session.close()
            return result
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            if 'session' in locals():
                session.close()
            return None
    
    def save_dataset(self, user_id: int, name: str, file_path: str, file_type: str, 
                    size_bytes: int, metadata: Dict[str, Any] = None) -> int:
        """
        Save a dataset to the database.
        
        Args:
            user_id: User ID
            name: Dataset name
            file_path: Path to the dataset file
            file_type: File type (csv, xlsx, etc.)
            size_bytes: File size in bytes
            metadata: Dataset metadata
            
        Returns:
            Dataset ID
        """
        try:
            session = self.Session()
            
            dataset = Dataset(
                user_id=user_id,
                name=name,
                file_path=file_path,
                file_type=file_type,
                size_bytes=size_bytes,
                metadata=metadata
            )
            
            session.add(dataset)
            session.commit()
            
            dataset_id = dataset.id
            session.close()
            
            return dataset_id
        except Exception as e:
            if 'session' in locals():
                session.rollback()
            logger.error(f"Error saving dataset: {e}")
            raise
    
    def save_model_evaluation(self, model_name: str, benchmark_name: str, task_type: str, 
                             metrics: Dict[str, Any]) -> int:
        """
        Save model evaluation results.
        
        Args:
            model_name: Name of the model
            benchmark_name: Name of the benchmark
            task_type: Type of task (code_generation, question_answering, etc.)
            metrics: Evaluation metrics
            
        Returns:
            Evaluation ID
        """
        try:
            session = self.Session()
            
            evaluation = ModelEvaluation(
                model_name=model_name,
                benchmark_name=benchmark_name,
                task_type=task_type,
                metrics=metrics
            )
            
            session.add(evaluation)
            session.commit()
            
            evaluation_id = evaluation.id
            session.close()
            
            return evaluation_id
        except Exception as e:
            if 'session' in locals():
                session.rollback()
            logger.error(f"Error saving model evaluation: {e}")
            raise
    
    def get_model_evaluations(self, model_name: Optional[str] = None, 
                             benchmark_name: Optional[str] = None,
                             task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get model evaluation results.
        
        Args:
            model_name: Filter by model name (optional)
            benchmark_name: Filter by benchmark name (optional)
            task_type: Filter by task type (optional)
            
        Returns:
            List of evaluation dictionaries
        """
        try:
            session = self.Session()
            
            # Build query
            query = session.query(ModelEvaluation)
            
            if model_name:
                query = query.filter_by(model_name=model_name)
            
            if benchmark_name:
                query = query.filter_by(benchmark_name=benchmark_name)
                
            if task_type:
                query = query.filter_by(task_type=task_type)
            
            # Execute query
            evaluations = query.all()
            
            result = [
                {
                    "id": eval.id,
                    "model_name": eval.model_name,
                    "benchmark_name": eval.benchmark_name,
                    "task_type": eval.task_type,
                    "metrics": eval.metrics,
                    "created_at": eval.created_at
                }
                for eval in evaluations
            ]
            
            session.close()
            return result
        except Exception as e:
            logger.error(f"Error getting model evaluations: {e}")
            if 'session' in locals():
                session.close()
            return []