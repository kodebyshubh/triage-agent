import httpx
from groq import BadRequestError


def bad_request(message: str, body: dict | None = None) -> BadRequestError:
    """A real groq.BadRequestError, built the way the SDK builds one from a 400."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return BadRequestError(message, response=httpx.Response(400, request=request), body=body)


def code_error(code: str) -> str:
    return f"Error code: 400 - {{'error': {{'message': 'x', 'code': '{code}'}}}}"


TOOL_USE_FAILED = "Error code: 400 - {'error': {'code': 'tool_use_failed'}}"
OTHER_BAD_REQUEST = "Error code: 400 - {'error': {'code': 'invalid_request_error'}}"
