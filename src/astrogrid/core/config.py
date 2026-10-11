"""
AstroGrid Core Configuration Module

Application configuration settings and environment variable parser.
"""

import os
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class DatabaseConfig(BaseModel):
    """Database connection settings."""
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="astrogrid", description="Database name")
    user: str = Field(default="astrogrid_user", description="Database user")
    password: str = Field(default="", description="Database password")
    pool_size: int = Field(default=10, description="Connection pool size")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v


class ObservabilityConfig(BaseModel):
    """Observability and logging settings."""
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json or text")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    tracing_enabled: bool = Field(default=True, description="Enable distributed tracing")
    sample_rate: float = Field(default=0.1, description="Tracing sample rate (0.0-1.0)")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Sample rate must be between 0.0 and 1.0")
        return v


class MessageBrokerConfig(BaseModel):
    """Message broker (Kafka-like) settings."""
    bootstrap_servers: str = Field(default="localhost:9092", description="Broker servers")
    topic_prefix: str = Field(default="astrogrid", description="Topic name prefix")
    consumer_group: str = Field(default="astrogrid-consumers", description="Consumer group ID")
    max_retries: int = Field(default=3, description="Max retry attempts")
    batch_size: int = Field(default=100, description="Batch size for producers")


class StorageConfig(BaseModel):
    """Object storage settings."""
    endpoint: str = Field(default="http://localhost:9000", description="Storage endpoint")
    access_key: str = Field(default="astrogrid", description="Access key")
    secret_key: str = Field(default="", description="Secret key")
    bucket: str = Field(default="astrogrid-data", description="Default bucket name")
    region: str = Field(default="us-east-1", description="Storage region")


class AppConfig(BaseModel):
    """Main application configuration."""
    app_name: str = Field(default="AstroGrid", description="Application name")
    environment: str = Field(default="development", description="Environment: development, staging, production")
    debug: bool = Field(default=False, description="Enable debug mode")
    api_host: str = Field(default="0.0.0.0", description="API bind host")
    api_port: int = Field(default=8000, description="API bind port")
    workers: int = Field(default=4, description="Number of worker processes")
    cors_origins: list = Field(default=["*"], description="Allowed CORS origins")

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    message_broker: MessageBrokerConfig = Field(default_factory=MessageBrokerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid_envs = {"development", "staging", "production"}
        v_lower = v.lower()
        if v_lower not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v_lower

    @field_validator("api_port")
    @classmethod
    def validate_api_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("API port must be between 1 and 65535")
        return v


def _get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve an environment variable with optional default."""
    return os.environ.get(key, default)


def _parse_bool(value: str) -> bool:
    """Parse a string to boolean."""
    return value.lower() in ("true", "1", "yes", "on")


def _parse_int(value: str) -> int:
    """Parse a string to integer."""
    return int(value)


def _parse_float(value: str) -> float:
    """Parse a string to float."""
    return float(value)


def _parse_list(value: str) -> list:
    """Parse a comma-separated string to a list."""
    return [item.strip() for item in value.split(",") if item.strip()]


def load_config_from_env() -> AppConfig:
    """
    Load application configuration from environment variables.
    
    Environment variables follow the pattern: ASTROGRID_<SECTION>_<KEY>
    Examples:
        ASTROGRID_ENVIRONMENT
        ASTROGRID_API_PORT
        ASTROGRID_DATABASE_HOST
        ASTROGRID_DATABASE_PORT
        ASTROGRID_OBSERVABILITY_LOG_LEVEL
        ASTROGRID_MESSAGE_BROKER_BOOTSTRAP_SERVERS
        ASTROGRID_STORAGE_ENDPOINT
    """
    # Top-level settings
    environment = _get_env("ASTROGRID_ENVIRONMENT", "development")
    debug = _parse_bool(_get_env("ASTROGRID_DEBUG", "false"))
    api_host = _get_env("ASTROGRID_API_HOST", "0.0.0.0")
    api_port = _parse_int(_get_env("ASTROGRID_API_PORT", "8000"))
    workers = _parse_int(_get_env("ASTROGRID_WORKERS", "4"))
    cors_origins = _parse_list(_get_env("ASTROGRID_CORS_ORIGINS", "*"))

    # Database settings
    db_host = _get_env("ASTROGRID_DATABASE_HOST", "localhost")
    db_port = _parse_int(_get_env("ASTROGRID_DATABASE_PORT", "5432"))
    db_name = _get_env("ASTROGRID_DATABASE_NAME", "astrogrid")
    db_user = _get_env("ASTROGRID_DATABASE_USER", "astrogrid_user")
    db_password = _get_env("ASTROGRID_DATABASE_PASSWORD", "")
    db_pool_size = _parse_int(_get_env("ASTROGRID_DATABASE_POOL_SIZE", "10"))

    # Observability settings
    log_level = _get_env("ASTROGRID_OBSERVABILITY_LOG_LEVEL", "INFO")
    log_format = _get_env("ASTROGRID_OBSERVABILITY_LOG_FORMAT", "json")
    metrics_enabled = _parse_bool(_get_env("ASTROGRID_OBSERVABILITY_METRICS_ENABLED", "true"))
    tracing_enabled = _parse_bool(_get_env("ASTROGRID_OBSERVABILITY_TRACING_ENABLED", "true"))
    sample_rate = _parse_float(_get_env("ASTROGRID_OBSERVABILITY_SAMPLE_RATE", "0.1"))

    # Message broker settings
    mb_servers = _get_env("ASTROGRID_MESSAGE_BROKER_BOOTSTRAP_SERVERS", "localhost:9092")
    mb_topic_prefix = _get_env("ASTROGRID_MESSAGE_BROKER_TOPIC_PREFIX", "astrogrid")
    mb_consumer_group = _get_env("ASTROGRID_MESSAGE_BROKER_CONSUMER_GROUP", "astrogrid-consumers")
    mb_max_retries = _parse_int(_get_env("ASTROGRID_MESSAGE_BROKER_MAX_RETRIES", "3"))
    mb_batch_size = _parse_int(_get_env("ASTROGRID_MESSAGE_BROKER_BATCH_SIZE", "100"))

    # Storage settings
    st_endpoint = _get_env("ASTROGRID_STORAGE_ENDPOINT", "http://localhost:9000")
    st_access_key = _get_env("ASTROGRID_STORAGE_ACCESS_KEY", "astrogrid")
    st_secret_key = _get_env("ASTROGRID_STORAGE_SECRET_KEY", "")
    st_bucket = _get_env("ASTROGRID_STORAGE_BUCKET", "astrogrid-data")
    st_region = _get_env("ASTROGRID_STORAGE_REGION", "us-east-1")

    return AppConfig(
        app_name="AstroGrid",
        environment=environment,
        debug=debug,
        api_host=api_host,
        api_port=api_port,
        workers=workers,
        cors_origins=cors_origins,
        database=DatabaseConfig(
            host=db_host,
            port=db_port,
            name=db_name,
            user=db_user,
            password=db_password,
            pool_size=db_pool_size,
        ),
        observability=ObservabilityConfig(
            log_level=log_level,
            log_format=log_format,
            metrics_enabled=metrics_enabled,
            tracing_enabled=tracing_enabled,
            sample_rate=sample_rate,
        ),
        message_broker=MessageBrokerConfig(
            bootstrap_servers=mb_servers,
            topic_prefix=mb