import time
from functools import wraps
from utils.logging import info, debug

class TokenTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.process_time = 0
    
    def update_from_llm_response(self, response):
        if not response:
            return
        
        try:
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                token_usage = metadata.get('token_usage', {})
                self.prompt_tokens = token_usage.get('prompt_tokens', 0)
                self.completion_tokens = token_usage.get('completion_tokens', 0)
                self.total_tokens = token_usage.get('total_tokens', 0)
            elif hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                self.prompt_tokens = usage.get('input_tokens', usage.get('prompt_tokens', 0))
                self.completion_tokens = usage.get('output_tokens', usage.get('completion_tokens', 0))
                self.total_tokens = self.prompt_tokens + self.completion_tokens
        except Exception as e:
            debug(f"Failed to extract token usage: {e}")
    
    def log(self):
        info(f"Tokens - Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}, Total: {self.total_tokens}")
        info(f"Process time: {self.process_time:.2f}s")
    
    def get_metadata(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "process_time": round(self.process_time, 2)
        }

def track_llm_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracker = TokenTracker()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        tracker.process_time = time.time() - start_time
        
        if isinstance(result, dict) and 'result' in result:
            tracker.update_from_llm_response(result.get('response', result))
        
        tracker.log()
        return {**result, "_metadata": tracker.get_metadata()} if isinstance(result, dict) else result
    
    return wrapper
