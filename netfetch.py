"""Polite HTTP downloads shared by every rebuild script (standard library only).

- One pacing clock per host: at least CORPORA_HF_ROWS_GAP seconds (default 5) between requests to the
  Hugging Face rows API, CORPORA_MIN_GAP seconds (default 1) for every other host.
- A Hugging Face token is sent to huggingface.co hosts when one is available (HF_TOKEN,
  HUGGING_FACE_HUB_TOKEN, or the token file written by `huggingface-cli login`). Anonymous requests have
  the lowest rate limit; a free token raises it. The token is never printed.
- HTTP 429 / 5xx and network errors are retried: the server's Retry-After is honoured when present,
  otherwise exponential backoff with jitter (up to 5 minutes). After a 429 the host's pacing gap doubles
  (up to 30 s) for the rest of the run, so a rebuild slows down instead of hammering the server.
- 404 and other client errors fail at once (retrying cannot fix them).
- Downloads stream to "<path>.tmp" and are renamed into place only when complete, so an interrupted
  run can simply be restarted with the same --cache directory: finished files are not fetched again.
"""
import email.utils, os, random, sys, time, urllib.error, urllib.parse, urllib.request

UA = "llm-eval-corpora/1.1 (corpus rebuild; python-urllib)"
ROWS_HOST = "datasets-server.huggingface.co"
_gap = {ROWS_HOST: float(os.environ.get("CORPORA_HF_ROWS_GAP", "5"))}
_default_gap = float(os.environ.get("CORPORA_MIN_GAP", "1"))
_last = {}
_token_cache = []


def _hf_token():
    if not _token_cache:
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not tok:
            home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
            path = os.environ.get("HF_TOKEN_PATH") or os.path.join(home, "token")
            try:
                with open(path, encoding="utf-8") as f:
                    tok = f.read().strip()
            except OSError:
                tok = None
        _token_cache.append(tok or None)
        if not tok:
            print("note: no Hugging Face token found; anonymous requests have the lowest rate limit "
                  "(set HF_TOKEN to raise it)", file=sys.stderr)
    return _token_cache[0]


def _pace(host):
    wait = _last.get(host, 0.0) + _gap.get(host, _default_gap) - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last[host] = time.monotonic()


def _retry_after(err):
    value = err.headers.get("Retry-After") if getattr(err, "headers", None) else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, email.utils.parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


def download(url, path, tries=10, timeout=120, check=None):
    """Fetch url into path (atomic rename). check(path) may raise to reject the body and retry."""
    host = urllib.parse.urlparse(url).hostname or ""
    use_token = host == "huggingface.co" or host.endswith(".huggingface.co")
    tmp = path + ".tmp"
    for attempt in range(1, tries + 1):
        _pace(host)
        headers = {"User-Agent": UA}
        token = _hf_token() if use_token else None
        if token:
            headers["Authorization"] = "Bearer " + token
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r, \
                    open(tmp, "wb") as out:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    out.write(chunk)
            if check:
                check(tmp)
            os.replace(tmp, path)
            return path
        except urllib.error.HTTPError as e:
            if e.code in (401, 403) and token:
                print(f"HTTP {e.code} with the Hugging Face token; retrying anonymously", file=sys.stderr)
                _token_cache[:] = [None]
                continue
            if e.code != 429 and e.code < 500:
                raise SystemExit(f"download failed (HTTP {e.code}): {url}")
            wait = _retry_after(e)
            if e.code == 429:
                _gap[host] = min(30.0, max(_gap.get(host, _default_gap), 1.0) * 2)
            reason = f"HTTP {e.code}"
        except Exception as e:  # network errors, timeouts, rejected bodies
            wait, reason = None, f"{type(e).__name__}: {e}"
        if attempt == tries:
            break
        if wait is None:
            wait = min(300.0, 10.0 * 2 ** (attempt - 1)) * (0.5 + random.random() / 2)
        print(f"{reason} from {host}; waiting {wait:.0f} s (attempt {attempt}/{tries})", file=sys.stderr)
        time.sleep(wait)
    try:
        os.remove(tmp)
    except OSError:
        pass
    raise SystemExit(f"download failed after {tries} attempts: {url} (rerun with the same --cache to resume)")
