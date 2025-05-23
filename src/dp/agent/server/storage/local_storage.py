import hashlib
import os
import shutil

from .base_storage import BaseStorage


class LocalStorage(BaseStorage):
    def __init__(self):
        pass

    def _upload(self, key, path):
        os.makedirs(os.path.dirname(key), exist_ok=True)
        shutil.copy(path, key)
        return os.path.abspath(key)

    def _download(self, key, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copy(key, path)
        return path

    def list(self, prefix, recursive=False):
        if os.path.isfile(prefix):
            return [prefix]
        if recursive:
            keys = []
            for f in os.listdir(prefix):
                if os.path.isdir(os.path.join(prefix, f)):
                    keys += self.list(os.path.join(prefix, f), recursive=True)
                elif os.path.isfile(os.path.join(prefix, f)):
                    keys.append(os.path.join(prefix, f))
            return keys
        else:
            return [os.path.join(prefix, f) for f in os.listdir(prefix)]

    def copy(self, src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    def get_md5(self, key):
        md5 = hashlib.md5()
        with open(key, "rb") as fd:
            for chunk in iter(lambda: fd.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()
