"""
Utility Functions for RM Assistant
Includes caching, logging, observability, and helper functions
"""

import asyncio
import functools
import hashlib
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, TypeVar
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


# ============================================================================
# Caching Utilities
# ============================================================================

class LRUCache:
    """Thread-safe LRU cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: dict = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            # Check TTL
            if self._is_expired(key):
                self._remove(key)
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_size:
                    # Remove oldest item
                    oldest = next(iter(self._cache))
                    self._remove(oldest)
            
            self._cache[key] = value
            self._timestamps[key] = datetime.now()
    
    def _is_expired(self, key: str) -> bool:
        """Check if entry is expired"""
        if key not in self._timestamps:
            return True
        age = (datetime.now() - self._timestamps[key]).total_seconds()
        return age > self.ttl_seconds
    
    def _remove(self, key: str) -> None:
        """Remove entry from cache"""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def clear(self) -> None:
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    @property
    def stats(self) -> dict:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3)
        }


class SemanticCache:
    """Cache with semantic similarity matching for LLM responses"""
    
    def __init__(
        self,
        max_size: int = 500,
        similarity_threshold: float = 0.95,
        embedding_fn: Optional[Callable] = None
    ):
        self.cache = LRUCache(max_size=max_size)
        self.similarity_threshold = similarity_threshold
        self.embedding_fn = embedding_fn
        self._embeddings: dict = {}
    
    def _compute_hash(self, text: str) -> str:
        """Compute hash for exact match lookup"""
        return hashlib.md5(text.encode()).hexdigest()
    
    async def get(self, query: str) -> Optional[Any]:
        """Get cached response using semantic similarity"""
        # Try exact match first
        exact_key = self._compute_hash(query)
        result = self.cache.get(exact_key)
        if result:
            return result
        
        # Try semantic match if embedding function provided
        if self.embedding_fn and self._embeddings:
            query_embedding = await self._get_embedding(query)
            for key, cached_embedding in self._embeddings.items():
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                if similarity >= self.similarity_threshold:
                    return self.cache.get(key)
        
        return None
    
    async def set(self, query: str, response: Any) -> None:
        """Cache response with embedding"""
        key = self._compute_hash(query)
        self.cache.set(key, response)
        
        if self.embedding_fn:
            embedding = await self._get_embedding(query)
            self._embeddings[key] = embedding
    
    async def _get_embedding(self, text: str) -> list:
        """Get embedding for text"""
        if asyncio.iscoroutinefunction(self.embedding_fn):
            return await self.embedding_fn(text)
        return self.embedding_fn(text)
    
    def _cosine_similarity(self, a: list, b: list) -> float:
        """Compute cosine similarity between vectors"""
        import math
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(x ** 2 for x in b))
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0


def cached(ttl_seconds: int = 3600, max_size: int = 100):
    """Decorator for caching function results"""
    cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = hashlib.md5(
                json.dumps((args, kwargs), sort_keys=True, default=str).encode()
            ).hexdigest()
            
            result = cache.get(key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        wrapper.cache = cache
        return wrapper
    
    return decorator


def async_cached(ttl_seconds: int = 3600, max_size: int = 100):
    """Decorator for caching async function results"""
    cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            key = hashlib.md5(
                json.dumps((args, kwargs), sort_keys=True, default=str).encode()
            ).hexdigest()
            
            result = cache.get(key)
            if result is not None:
                return result
            
            result = await func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        wrapper.cache = cache
        return wrapper
    
    return decorator


# ============================================================================
# Logging Utilities
# ============================================================================

@dataclass
class AuditLogEntry:
    """Audit log entry for compliance tracking"""
    timestamp: datetime
    action: str
    user_id: Optional[str]
    customer_id: Optional[str]
    details: dict
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    status: str = "success"
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "user_id": self.user_id,
            "customer_id": self.customer_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
            "status": self.status
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class AuditLogger:
    """Audit logger for compliance and regulatory requirements"""
    
    def __init__(self, log_file: str = None, logger_name: str = "audit"):
        self.logger = logging.getLogger(logger_name)
        self.log_file = log_file
        self._entries: list = []
    
    def log(
        self,
        action: str,
        user_id: str = None,
        customer_id: str = None,
        details: dict = None,
        status: str = "success"
    ) -> AuditLogEntry:
        """Log an audit event"""
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            action=action,
            user_id=user_id,
            customer_id=customer_id,
            details=details or {},
            status=status
        )
        
        self.logger.info(entry.to_json())
        self._entries.append(entry)
        
        return entry
    
    def log_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        result: Any,
        user_id: str = None,
        execution_time: float = None
    ) -> AuditLogEntry:
        """Log a tool call for audit trail"""
        return self.log(
            action="tool_call",
            user_id=user_id,
            details={
                "tool": tool_name,
                "arguments": arguments,
                "result_summary": str(result)[:500],
                "execution_time_ms": execution_time
            }
        )
    
    def log_customer_access(
        self,
        customer_id: str,
        access_type: str,
        user_id: str,
        data_accessed: list[str] = None
    ) -> AuditLogEntry:
        """Log customer data access"""
        return self.log(
            action="customer_data_access",
            user_id=user_id,
            customer_id=customer_id,
            details={
                "access_type": access_type,
                "data_fields": data_accessed or []
            }
        )
    
    def get_entries(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        action: str = None,
        user_id: str = None
    ) -> list[AuditLogEntry]:
        """Query audit log entries"""
        entries = self._entries
        
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]
        if action:
            entries = [e for e in entries if e.action == action]
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        
        return entries


class StructuredLogger:
    """Structured JSON logger for observability"""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._context: dict = {}
    
    def set_context(self, **kwargs) -> None:
        """Set persistent context fields"""
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear context"""
        self._context.clear()
    
    def _format_message(self, message: str, extra: dict = None) -> str:
        """Format message as JSON"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            **self._context,
            **(extra or {})
        }
        return json.dumps(log_data)
    
    def info(self, message: str, **extra) -> None:
        self.logger.info(self._format_message(message, extra))
    
    def warning(self, message: str, **extra) -> None:
        self.logger.warning(self._format_message(message, extra))
    
    def error(self, message: str, **extra) -> None:
        self.logger.error(self._format_message(message, extra))
    
    def debug(self, message: str, **extra) -> None:
        self.logger.debug(self._format_message(message, extra))


# ============================================================================
# Timing Utilities
# ============================================================================

@contextmanager
def timer(name: str = "operation"):
    """Context manager for timing operations"""
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"{name} completed in {elapsed:.2f}ms")


def timed(func: Callable) -> Callable:
    """Decorator to time function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"{func.__name__} executed in {elapsed:.2f}ms")
    
    return wrapper


def async_timed(func: Callable) -> Callable:
    """Decorator to time async function execution"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"{func.__name__} executed in {elapsed:.2f}ms")
    
    return wrapper


# ============================================================================
# Retry Utilities
# ============================================================================

def retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retrying failed operations with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = delay_seconds
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}"
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
            
            raise last_exception
        
        return wrapper
    return decorator


def async_retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retrying failed async operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            delay = delay_seconds
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}"
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
            
            raise last_exception
        
        return wrapper
    return decorator


# ============================================================================
# Data Masking Utilities
# ============================================================================

class DataMasker:
    """Utility for masking sensitive data"""
    
    # Default patterns to mask
    DEFAULT_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "account_number": r"\b\d{10,12}\b"
    }
    
    def __init__(self, patterns: dict = None):
        import re
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self._compiled = {
            name: re.compile(pattern) 
            for name, pattern in self.patterns.items()
        }
    
    def mask(self, text: str, mask_char: str = "*") -> str:
        """Mask sensitive data in text"""
        result = text
        for name, pattern in self._compiled.items():
            result = pattern.sub(
                lambda m: self._mask_match(m.group(), mask_char),
                result
            )
        return result
    
    def _mask_match(self, value: str, mask_char: str) -> str:
        """Mask a matched value, preserving some characters"""
        if len(value) <= 4:
            return mask_char * len(value)
        return value[:2] + mask_char * (len(value) - 4) + value[-2:]
    
    def mask_dict(self, data: dict, sensitive_keys: list[str] = None) -> dict:
        """Mask sensitive values in a dictionary"""
        sensitive = sensitive_keys or [
            "password", "secret", "token", "key", "ssn", 
            "account_number", "credit_card"
        ]
        
        result = {}
        for key, value in data.items():
            if any(s in key.lower() for s in sensitive):
                result[key] = "***MASKED***"
            elif isinstance(value, str):
                result[key] = self.mask(value)
            elif isinstance(value, dict):
                result[key] = self.mask_dict(value, sensitive_keys)
            else:
                result[key] = value
        
        return result


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(
        self,
        max_tokens: int = 100,
        refill_rate: float = 10.0,  # tokens per second
        refill_interval: float = 1.0
    ):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.refill_interval = refill_interval
        self._tokens = max_tokens
        self._last_refill = time.time()
        self._lock = threading.Lock()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self.refill_rate
        self._tokens = min(self.max_tokens, self._tokens + tokens_to_add)
        self._last_refill = now
    
    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens, return success"""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False
    
    async def acquire_async(self, tokens: int = 1, timeout: float = None) -> bool:
        """Async version with optional wait"""
        start = time.time()
        while True:
            if self.acquire(tokens):
                return True
            
            if timeout and (time.time() - start) >= timeout:
                return False
            
            await asyncio.sleep(0.1)
    
    @property
    def available_tokens(self) -> float:
        """Get current available tokens"""
        with self._lock:
            self._refill()
            return self._tokens


def rate_limited(limiter: RateLimiter, tokens: int = 1):
    """Decorator for rate limiting function calls"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not limiter.acquire(tokens):
                raise RuntimeError("Rate limit exceeded")
            return func(*args, **kwargs)
        return wrapper
    return decorator


__all__ = [
    # Caching
    "LRUCache",
    "SemanticCache",
    "cached",
    "async_cached",
    
    # Logging
    "AuditLogEntry",
    "AuditLogger",
    "StructuredLogger",
    
    # Timing
    "timer",
    "timed",
    "async_timed",
    
    # Retry
    "retry",
    "async_retry",
    
    # Data masking
    "DataMasker",
    
    # Rate limiting
    "RateLimiter",
    "rate_limited",
]
