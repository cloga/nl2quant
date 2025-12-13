import threading
import time


class RateLimiter:
    """Simple token bucket style rate limiter.

    limit: max calls per interval
    interval: seconds per window (e.g., 60 for per-minute)
    """

    def __init__(self, limit: int = 500, interval: float = 60.0):
        self.limit = limit
        self.interval = interval
        self._lock = threading.Lock()
        self._tokens = limit
        self._last_refill = time.time()

    def _refill(self):
        now = time.time()
        elapsed = now - self._last_refill
        if elapsed >= self.interval:
            # reset bucket each interval
            self._tokens = self.limit
            self._last_refill = now

    def acquire(self):
        """Block until a token is available."""
        while True:
            with self._lock:
                self._refill()
                if self._tokens > 0:
                    self._tokens -= 1
                    return
            # no tokens; sleep briefly and retry
            time.sleep(0.05)


# Global limiter instance (shared across module imports)
GLOBAL_LIMITER = RateLimiter(limit=500, interval=60.0)
