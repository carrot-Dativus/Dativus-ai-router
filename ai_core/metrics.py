"""경량 런타임 메트릭 수집기 — 평가 스크립트(eval_test.py)가 읽어 가는 전역 카운터."""
import threading

_lock = threading.Lock()
_total_llm = 0      # _invoke_with_backoff 통과 성공 횟수
_fallback = 0       # Ollama 폴백 발생 횟수


def record_llm_call(is_fallback: bool = False) -> None:
    global _total_llm, _fallback
    with _lock:
        _total_llm += 1
        if is_fallback:
            _fallback += 1


def get_stats() -> dict:
    with _lock:
        total = _total_llm
        fb    = _fallback
    return {
        "total_llm_calls": total,
        "fallback_calls":  fb,
        "fallback_ratio":  fb / total if total else 0.0,
    }


def reset() -> None:
    global _total_llm, _fallback
    with _lock:
        _total_llm = 0
        _fallback  = 0
