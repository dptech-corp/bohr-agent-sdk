import os
import shutil
from typing import Optional

import requests

from .base_storage import BaseStorage

config = {
    "plugin_type": os.environ.get("HTTP_PLUGIN_TYPE"),
}


class HTTPStorage(BaseStorage):
    scheme = "http"

    def __init__(
        self,
        plugin: dict = None,
        max_upload_size: Optional[int] = None,
        max_download_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(max_upload_size=max_upload_size, max_download_size=max_download_size)
        self.plugin = None
        if plugin is None and config["plugin_type"] is not None:
            plugin = {"type": config["plugin_type"]}
        if plugin is not None:
            from . import storage_dict
            storage_type = plugin.pop("type")
            self.plugin = storage_dict[storage_type](**plugin)

    def get_size(self, key: str) -> Optional[int]:
        url = self.scheme + "://" + key
        try:
            r = requests.head(url, verify=False, timeout=10)
            if r.ok and "Content-Length" in r.headers:
                return int(r.headers["Content-Length"])
        except requests.RequestException:
            pass
        return None

    def _upload(self, key, path):
        if self.plugin is not None:
            key = self.plugin._upload(key, path)
            url = self.plugin.get_http_url(key)
            return url.split("://")[1]
        else:
            raise NotImplementedError()

    def _download(self, key, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sess = requests.session()
        url = self.scheme + "://" + key
        with sess.get(url, stream=True, verify=False) as req:
            req.raise_for_status()
            with open(path, 'w') as f:
                shutil.copyfileobj(req.raw, f.buffer)
        return path

    def list(self, prefix, recursive=False):
        return [prefix]

    def copy(self, src, dst):
        raise NotImplementedError()

    def get_md5(self, key):
        raise NotImplementedError()


class HTTPSStorage(HTTPStorage):
    scheme = "https"
