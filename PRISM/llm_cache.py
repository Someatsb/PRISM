import os
import pickle
import time


LOCK_RETRY_INTERVAL = 0.1
LOCK_TIMEOUT = 30.0


def get_cache_path(args):
    cache_dir = os.path.join(args.main_path, "dataset", f"{args.dataset}_llm_cache")
    os.makedirs(cache_dir, exist_ok=True)

    file_name = f"p{args.perspective}_k{args.num_nodes_per_view}_mmr{args.mmr_lambda}.pkl"
    return os.path.join(cache_dir, file_name)


class _file_lock:
    def __init__(self, lock_path, timeout=LOCK_TIMEOUT):
        self.lock_path = lock_path
        self.timeout = timeout
        self._fd = None

    def __enter__(self):
        deadline = time.time() + self.timeout

        while True:
            try:
                self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return self
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(f"타임아웃: 락 파일을 획득하지 못했습니다: {self.lock_path}")
                time.sleep(LOCK_RETRY_INTERVAL)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._fd is not None:
            os.close(self._fd)
        try:
            os.remove(self.lock_path)
        except FileNotFoundError:
            pass


def load_cache(args):
    cache_path = get_cache_path(args)

    if not os.path.exists(cache_path):
        return {}

    with open(cache_path, "rb") as f:
        return pickle.load(f)


def save_new_entries(args, new_results_by_node):
    if not new_results_by_node:
        return

    cache_path = get_cache_path(args)
    lock_path = cache_path + ".lock"

    with _file_lock(lock_path):
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)

        cache.update(new_results_by_node)

        tmp_path = cache_path + f".tmp{os.getpid()}"
        with open(tmp_path, "wb") as f:
            pickle.dump(cache, f)
        os.replace(tmp_path, cache_path)
