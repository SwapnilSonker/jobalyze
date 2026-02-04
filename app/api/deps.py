"""
Shared dependencies for API routes
"""
from app.db.database import get_db
from app.core.security import get_current_user

__all__ = ["get_db", "get_current_user"]
