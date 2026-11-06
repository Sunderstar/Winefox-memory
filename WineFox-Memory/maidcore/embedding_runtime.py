"""Process-local embedding model pool with nonblocking warm-up.

No network access is requested by the loader: weights must already exist in
Hugging Face's local cache. Only this module imports sentence_transformers,
so the ~7s torch/transformers import can happen before the first chat turn.
"""
from __future__ import annotations

import threading


class EmbeddingModelPool:
    _lock = threading.RLock()
    _entries: dict[str, dict] = {}

    @staticmethod
    def _load_model(model_name):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, local_files_only=True)

    @classmethod
    def _do_load(cls, name, entry):
        try:
            instance=cls._load_model(name)
        except Exception as exc:
            # Background failures must never crash a chat session. Do not
            # retry every turn; a changed setting or process restart retries.
            with cls._lock:
                entry['state']='failed'
                entry['error']=type(exc).__name__
        else:
            with cls._lock:
                entry['state']='ready'
                entry['model']=instance
        finally:
            entry['finished'].set()

    @classmethod
    def start(cls, name, *, background=True):
        """Start exactly one load per model per process, or reuse its result."""
        with cls._lock:
            entry=cls._entries.get(name)
            if entry is not None:
                return cls.status(name)
            entry={'state':'loading', 'model':None, 'error':None,
                   'finished':threading.Event(), 'encode_lock':threading.Lock()}
            cls._entries[name]=entry
        if background:
            threading.Thread(target=cls._do_load,args=(name,entry),
                             name='winefox-embedding-warmup',daemon=True).start()
        else:
            cls._do_load(name,entry)
        return cls.status(name)

    @classmethod
    def get(cls, name, *, nonblocking_if_loading=True):
        """Return (model, state). During startup fall back to lexical search."""
        with cls._lock:
            entry=cls._entries.get(name)
        if entry is None:
            cls.start(name,background=False)
            with cls._lock:entry=cls._entries[name]
        if entry['state']=='loading' and not nonblocking_if_loading:
            entry['finished'].wait()
        if entry['state']=='ready':
            return entry['model'], 'ready'
        if entry['state']=='loading':
            return None, 'loading'
        return None, 'failed: '+str(entry['error'])

    @classmethod
    def encode(cls, name, model, texts, **kwargs):
        """Serialize one model's encode() calls across concurrent UI requests."""
        with cls._lock:
            entry=cls._entries.get(name)
        if entry is None:
            return model.encode(texts,**kwargs)
        with entry['encode_lock']:
            return model.encode(texts,**kwargs)

    @classmethod
    def status(cls,name):
        with cls._lock:
            entry=cls._entries.get(name)
            if entry is None:return {'model':name,'state':'not_started'}
            result={'model':name,'state':entry['state']}
            if entry['error']:result['error_type']=entry['error']
            return result
