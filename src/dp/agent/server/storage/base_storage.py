import os
import shutil
import tarfile
import tempfile
from abc import ABC, abstractmethod
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
            path = os.path.join(path, os.path.basename(key))
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


def merge_dir(src, dst):
    for f in os.listdir(src):
        src_file = os.path.join(src, f)
        dst_file = os.path.join(dst, f)
        if os.path.isdir(src_file):
            if os.path.isdir(dst_file):
                merge_dir(src_file, dst_file)
            else:
                shutil.move(src_file, dst_file)
        elif os.path.isfile(src_file):
            shutil.move(src_file, dst_file)


def extract(path):
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(tmpdir)

        os.remove(path)
        path = os.path.dirname(path)

        # if the tarfile contains only one directory, merge the
        # directory with the target directory
        ld = os.listdir(tmpdir)
        if len(ld) == 1 and os.path.isdir(os.path.join(tmpdir, ld[0])):
            merge_dir(os.path.join(tmpdir, ld[0]), path)
        else:
            merge_dir(tmpdir, path)
    return path
