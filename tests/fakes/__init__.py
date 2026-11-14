"""Stand-ins shared by the tests: a scripted model, a throwaway service, and a small HTTP client."""
from .http import fetch, get_json, send_json
from .models import SALES_QUERY, sales_model
from .services import sales_service

__all__ = ["SALES_QUERY", "fetch", "get_json", "sales_model", "sales_service", "send_json"]
