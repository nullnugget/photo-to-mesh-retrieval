"""HTTP downloads with connection reuse, byte ranges, retries, and safe file writes.

Downloads from the ABO bucket (in the US) are slow for each connection, but many connections in
parallel reach the full line speed. A new HTTPS connection also costs about 1 second, against
about 0.5 seconds for a request on an open connection. Thus:

- each worker thread keeps its own open connection (one `requests.Session` per thread);
- a failed request is tried again after a growing wait (1 s, 2 s, 4 s, ...), because the server
  can answer "503 Slow Down" or the connection can break;
- a file is written to `<name>.part` first and renamed at the end, so that an interrupted
  download never leaves a half-written file that looks complete.
"""

import threading
import time
from pathlib import Path

import requests

RETRY_STATUS = {429, 500, 502, 503, 504}  # answers that can succeed on a later attempt


class NotFound(Exception):
    """The server says that the file does not exist (HTTP 403 or 404). A retry does not help."""


class Fetcher:
    """Download bytes or files. One instance can be shared by many threads."""

    def __init__(self, retries: int = 5, timeout: float = 60.0, first_wait: float = 1.0):
        """
        Args:
            retries: Attempts for each request before the error is raised.
            timeout: Seconds to wait for the server before an attempt fails.
            first_wait: Wait before the second attempt; it doubles for each further attempt.
        """
        self.retries = retries
        self.timeout = timeout
        self.first_wait = first_wait
        self._local = threading.local()

    def _session(self) -> requests.Session:
        """Return the open connection of the calling thread (made on first use)."""
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def get(self, url: str, start: int | None = None, end: int | None = None) -> bytes:
        """Download a whole file, or the bytes [start, end) of it.

        Raises:
            NotFound: For HTTP 403 or 404.
            requests.RequestException: If all attempts fail.
        """
        headers = {} if start is None else {"Range": f"bytes={start}-{end - 1}"}  # HTTP ranges include the end
        for attempt in range(self.retries):
            try:
                response = self._session().get(url, headers=headers, timeout=self.timeout)
                if response.status_code in (403, 404):
                    raise NotFound(f"HTTP {response.status_code}: {url}")
                if response.status_code in RETRY_STATUS:
                    raise requests.HTTPError(f"HTTP {response.status_code}: {url}")
                response.raise_for_status()
                body = response.content
                if start is not None and response.status_code == 200:  # the server ignored the range
                    body = body[start:end]
                if start is not None and len(body) != end - start:
                    raise requests.HTTPError(f"expected {end - start} bytes, got {len(body)}: {url}")
                return body
            except NotFound:
                raise
            except requests.RequestException:
                if attempt == self.retries - 1:
                    raise
                self._local.__dict__.pop("session", None)  # open a fresh connection for the next attempt
                time.sleep(self.first_wait * 2**attempt)
        raise AssertionError("unreachable")

    def save(self, url: str, path: Path) -> int:
        """Download a whole file to `path` (through `<name>.part`). Returns the number of bytes."""
        body = self.get(url)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        temporary.write_bytes(body)
        temporary.replace(path)
        return len(body)
