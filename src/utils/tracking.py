import time
from functools import wraps
from langchain_core.callbacks import BaseCallbackHandler

class TokenTrackingCallback(BaseCallbackHandler):
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._token_usage_list = []
        self._usage_metadata_list = []
    
    def on_llm_end(self, response, **kwargs):
        try:
            # Try to get token usage from generations
            if hasattr(response, 'generations') and response.generations:
                for generation_list in response.generations:
                    if generation_list and hasattr(generation_list[0], 'generation_info'):
                        gen_info = generation_list[0].generation_info
                        if gen_info:
                            # Ollama provides token counts in generation_info
                            if 'prompt_eval_count' in gen_info:
                                self.prompt_tokens += gen_info.get('prompt_eval_count', 0)
                            if 'eval_count' in gen_info:
                                self.completion_tokens += gen_info.get('eval_count', 0)
                            self.total_tokens = self.prompt_tokens + self.completion_tokens
                            
                            # Also check for token_usage dict
                            if 'token_usage' in gen_info:
                                usage = gen_info['token_usage']
                                self.prompt_tokens += usage.get('prompt_tokens', 0)
                                self.completion_tokens += usage.get('completion_tokens', 0)
                                self.total_tokens = self.prompt_tokens + self.completion_tokens
        except Exception as e:
            from src.utils.logging import debug
            debug(f"Failed to extract token usage from callback: {e}")
    
    def on_chain_end(self, outputs, **kwargs):
        """Capture token usage from chain outputs if available"""
        try:
            if isinstance(outputs, dict):
                # Check for usage metadata in the output
                if 'usage_metadata' in outputs:
                    usage = outputs['usage_metadata']
                    self.prompt_tokens += usage.get('input_tokens', 0)
                    self.completion_tokens += usage.get('output_tokens', 0)
                    self.total_tokens += usage.get('total_tokens', 0)
                
                # Check for response metadata
                if 'response_metadata' in outputs:
                    metadata = outputs['response_metadata']
                    if 'token_usage' in metadata:
                        usage = metadata['token_usage']
                        self.prompt_tokens += usage.get('prompt_tokens', 0)
                        self.completion_tokens += usage.get('completion_tokens', 0)
                        self.total_tokens += usage.get('total_tokens', 0)
        except Exception as e:
            from src.utils.logging import debug
            debug(f"Failed to extract token usage from chain end: {e}")
    
    def update_from_usage_metadata(self, input_tokens, output_tokens):
        """Update tokens from usage_metadata dict"""
        self.prompt_tokens = input_tokens
        self.completion_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
    
    def get_metadata(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "process_time": 0
        }

class TokenTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.process_time = 0
        self.callback = TokenTrackingCallback()
    
    def get_callbacks(self):
        return [self.callback]
    
    def update_from_response(self):
        """Update token counts from the callback"""
        self.prompt_tokens = self.callback.prompt_tokens
        self.completion_tokens = self.callback.completion_tokens
        self.total_tokens = self.callback.total_tokens
    
    def log(self):
        from src.utils.logging import info
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
        tracker.update_from_response()
        
        tracker.log()
        return {**result, "_metadata": tracker.get_metadata()} if isinstance(result, dict) else result
    
    return wrapper
