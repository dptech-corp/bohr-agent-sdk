import os
import tarfile
from abc import ABC, abstractmethod
from typing import List, Optional


class StorageLimitExceededError(Exception):
    """Raised when upload or download exceeds configured size limit."""

    def __init__(self, message: str, size: int, limit: int, direction: str):
        self.size = size
        self.limit = limit
        self.direction = direction  # "upload" or "download"
        super().__init__(message)


class BaseStorage(ABC):
    """Base storage with optional transfer size limits (default: no limit)."""

    def __init__(
        self,
        max_upload_size: Optional[int] = None,
        max_download_size: Optional[int] = None,
    ):
        """
        Args:
            max_upload_size: Max size in bytes for a single upload; None = no limit.
            max_download_size: Max size in bytes for a single download; None = no limit.
        """
        self._max_upload_size = max_upload_size
        self._max_download_size = max_download_size

    def get_size(self, key: str) -> Optional[int]:
        """
        Return size in bytes for the object at key, or None if unknown.
        Override in subclasses that can provide size (e.g. for download limit check).
        """
        return None

    def _check_upload_limit(self, path: str) -> None:
        """Raise StorageLimitExceededError if path size exceeds max_upload_size. No-op if no limit."""
        if self._max_upload_size is None:
            return
        size = os.path.getsize(path)
        if size > self._max_upload_size:
            raise StorageLimitExceededError(
                f"Upload size {size} exceeds limit {self._max_upload_size} bytes",
                size=size,
                limit=self._max_upload_size,
                direction="upload",
            )

    def _check_download_limit(self, key: str) -> None:
        """Raise StorageLimitExceededError if object size exceeds max_download_size. No-op if no limit or size unknown."""
        if self._max_download_size is None:
            return
        size = self.get_size(key)
        if size is not None and size > self._max_download_size:
            raise StorageLimitExceededError(
                f"Download size {size} exceeds limit {self._max_download_size} bytes",
                size=size,
                limit=self._max_download_size,
                direction="download",
            )

    @abstractmethod
    def _upload(self, key: str, path: str) -> str:
        """
        Upload a file from path to key
        """
        pass

    @abstractmethod
    def _download(self, key: str, path: str) -> str:
        """
        Download a file from key to path
        """
        pass

    @abstractmethod
    def list(self, prefix: str, recursive: bool = False) -> List[str]:
        pass

    @abstractmethod
    def copy(self, src: str, dst: str) -> None:
        pass

    @abstractmethod
    def get_md5(self, key: str) -> str:
        pass

    def download(self, key: str, path: str) -> str:
        objs = self.list(prefix=key, recursive=True)
        if objs == [key]:
            path = os.path.join(path, os.path.basename(key.split("?")[0]))
            self._check_download_limit(key)
            self._download(key=key, path=path)
            if path[-4:] == ".tgz":
                path = extract(path)
        else:
            for obj in objs:
                self._check_download_limit(obj)
                rel_path = obj[len(key):]
                if rel_path[:1] == "/":
                    rel_path = rel_path[1:]
                file_path = os.path.join(path, rel_path)
                self._download(key=obj, path=file_path)
        return path

    def upload(self, key: str, path: str) -> str:
        if os.path.isfile(path):
            self._check_upload_limit(path)
            key = os.path.join(key, os.path.basename(path))
            key = self._upload(key, path)
        elif os.path.isdir(path):
            cwd = os.getcwd()
            if os.path.dirname(path):
                os.chdir(os.path.dirname(path))
            fname = os.path.basename(path)
            tgz_path = fname + ".tgz"
            with tarfile.open(tgz_path, "w:gz", dereference=True) as tf:
                tf.add(fname)
            os.chdir(cwd)
            full_tgz = "%s.tgz" % path
            try:
                self._check_upload_limit(full_tgz)
            except StorageLimitExceededError:
                os.remove(full_tgz)
                raise
            key = os.path.join(key, fname + ".tgz")
            key = self._upload(key, full_tgz)
            os.remove(full_tgz)
        return key


def extract(path):
    with tarfile.open(path, "r:gz") as tf:
        common = os.path.commonpath(tf.getnames())
        tf.extractall(os.path.dirname(path))

    os.remove(path)
    path = os.path.dirname(path)
    # if the tarfile contains only one directory,
    # return its path
    if common != "":
        return os.path.join(path, common)
    else:
        return path
