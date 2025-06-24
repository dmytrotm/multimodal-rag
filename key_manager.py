from collections import deque
import threading
import time

class SmartAPIKeyManager:
    def __init__(self, api_keys, queries_per_minute=15, safety_buffer=2.0):
        self.api_keys = api_keys
        self.queries_per_minute = queries_per_minute
        self.min_interval = (60.0 / queries_per_minute) * safety_buffer
        self.key_request_times = {key: deque() for key in api_keys}
        self.key_failures = {key: 0 for key in api_keys}
        self.key_cooldown = {key: 0 for key in api_keys}
        self.key_token_usage = {key: deque() for key in api_keys}
        self.lock = threading.Lock()
        self.failure_cooldown_time = 120
        self.max_tokens_per_minute = 10000
        print(f"📊 Rate limiting: {queries_per_minute} queries/min per key (min interval: {self.min_interval:.2f}s)")

    def _clean_old_requests(self, key, current_time):
        request_times = self.key_request_times[key]
        while request_times and current_time - request_times[0] > 60:
            request_times.popleft()
        
        token_times = self.key_token_usage[key]
        while token_times and current_time - token_times[0][0] > 60:
            token_times.popleft()

    def _estimate_tokens(self, text):
        """Rough token estimation (1 token ≈ 4 characters for most models)"""
        return len(text) // 4

    def _can_make_request(self, key, current_time, estimated_tokens=0):
        self._clean_old_requests(key, current_time)
        
        if current_time - self.key_cooldown[key] < self.failure_cooldown_time:
            return False, "failure_cooldown"
        
        request_times = self.key_request_times[key]
        if len(request_times) >= self.queries_per_minute:
            return False, "rate_limit"
        
        if request_times and current_time - request_times[-1] < self.min_interval:
            return False, "min_interval"
        
        current_tokens = sum(tokens for _, tokens in self.key_token_usage[key])
        if current_tokens + estimated_tokens > self.max_tokens_per_minute:
            return False, "token_limit"
        
        return True, "ready"

    def get_best_api_key(self, estimated_tokens=0, thread_id=None):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            with self.lock:
                current_time = time.time()
                key_status = {}
                
                for key in self.api_keys:
                    can_use, reason = self._can_make_request(key, current_time, estimated_tokens)
                    key_status[key] = {
                        'can_use': can_use, 
                        'reason': reason, 
                        'failures': self.key_failures[key], 
                        'requests_count': len(self.key_request_times[key]), 
                        'last_request': self.key_request_times[key][-1] if self.key_request_times[key] else 0,
                        'token_usage': sum(tokens for _, tokens in self.key_token_usage[key])
                    }
                
                available_keys = [k for k, s in key_status.items() if s['can_use']]
                
                if available_keys:
                    best_key = min(available_keys, key=lambda k: (
                        key_status[k]['failures'], 
                        key_status[k]['token_usage'],
                        key_status[k]['requests_count']
                    ))
                    
                    self.key_request_times[best_key].append(current_time)
                    self.key_token_usage[best_key].append((current_time, estimated_tokens))
                    return best_key
                
                wait_times = []
                for key, status in key_status.items():
                    if status['reason'] == 'failure_cooldown':
                        wait_times.append(self.failure_cooldown_time - (current_time - self.key_cooldown[key]))
                    elif status['reason'] == 'min_interval':
                        wait_times.append(self.min_interval - (current_time - status['last_request']))
                    elif status['reason'] == 'rate_limit':
                        wait_times.append(60 - (current_time - self.key_request_times[key][0]))
                    elif status['reason'] == 'token_limit':
                        wait_times.append(30)
                
                min_wait = min(wait_times) if wait_times else 5.0
                min_wait = max(1.0, min_wait)
            
            if thread_id:
                print(f"⏳ Thread {thread_id}: All keys busy. Waiting {min_wait:.1f}s... (attempt {retry_count + 1}/{max_retries})")
            time.sleep(min_wait)
            retry_count += 1
        
        raise Exception(f"Thread {thread_id}: Could not get available API key after maximum retries")

    def mark_key_failed(self, api_key, thread_id=None):
        with self.lock:
            self.key_failures[api_key] += 1
            self.key_cooldown[api_key] = time.time()
            if thread_id:
                print(f"❌ Thread {thread_id}: API key failed. Failures: {self.key_failures[api_key]}")

    def mark_key_success(self, api_key):
        with self.lock:
            self.key_failures[api_key] = max(0, self.key_failures[api_key] - 1)

