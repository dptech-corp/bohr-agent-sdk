import asyncio
import os
import tarfile
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import List


class BaseStorage(ABC):
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
            self._download(key=key, path=path)
            if path[-4:] == ".tgz":
                path = extract(path)
        else:
            for obj in objs:
                rel_path = obj[len(key):]
                if rel_path[:1] == "/":
                    rel_path = rel_path[1:]
                file_path = os.path.join(path, rel_path)
                self._download(key=obj, path=file_path)
        return path

    def upload(self, key: str, path: str) -> str:
        if os.path.isfile(path):
            key = os.path.join(key, os.path.basename(path))
            key = self._upload(key, path)
        elif os.path.isdir(path):
            cwd = os.getcwd()
            if os.path.dirname(path):
                os.chdir(os.path.dirname(path))
            fname = os.path.basename(path)
            with tarfile.open(fname + ".tgz", "w:gz", dereference=True) as tf:
                tf.add(fname)
            os.chdir(cwd)
            key = os.path.join(key, fname + ".tgz")
            key = self._upload(key, "%s.tgz" % path)
            os.remove("%s.tgz" % path)
        return key

    async def _async_upload(self, key, path):
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(1) as pool:
            key = await loop.run_in_executor(pool, self._upload, key, path)
        return key

    async def _async_download(self, key, path):
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(1) as pool:
            path = await loop.run_in_executor(pool, self._download, key, path)
        return path

    async def async_download(self, key: str, path: str) -> str:
        objs = self.list(prefix=key, recursive=True)
        if objs == [key]:
            path = os.path.join(path, os.path.basename(key.split("?")[0]))
            await self._async_download(key=key, path=path)
            if path[-4:] == ".tgz":
                path = extract(path)
        else:
            for obj in objs:
                rel_path = obj[len(key):]
                if rel_path[:1] == "/":
                    rel_path = rel_path[1:]
                file_path = os.path.join(path, rel_path)
                await self._async_download(key=obj, path=file_path)
        return path

    async def async_upload(self, key: str, path: str) -> str:
        if os.path.isfile(path):
            key = os.path.join(key, os.path.basename(path))
            key = await self._async_upload(key, path)
        elif os.path.isdir(path):
            cwd = os.getcwd()
            if os.path.dirname(path):
                os.chdir(os.path.dirname(path))
            fname = os.path.basename(path)
            with tarfile.open(fname + ".tgz", "w:gz", dereference=True) as tf:
                tf.add(fname)
            os.chdir(cwd)
            key = os.path.join(key, fname + ".tgz")
            key = await self._async_upload(key, "%s.tgz" % path)
            os.remove("%s.tgz" % path)
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
