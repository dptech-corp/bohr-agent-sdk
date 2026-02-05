import json
import os
from typing import Optional

import requests
import tiefblue

from .base_storage import BaseStorage

succ_code = [0, "0000"]
config = {
    "bohrium_url": os.environ.get("BOHRIUM_BOHRIUM_URL",
                                  "https://bohrium.dp.tech"),
    "username": os.environ.get("BOHRIUM_USERNAME"),
    "phone": os.environ.get("BOHRIUM_PHONE"),
    "password": os.environ.get("BOHRIUM_PASSWORD"),
    "authorization": os.environ.get("BOHRIUM_AUTHORIZATION"),
    "project_id": os.environ.get("BOHRIUM_PROJECT_ID"),
    "tiefblue_url": os.environ.get("BOHRIUM_TIEFBLUE_URL",
                                   "https://tiefblue.dp.tech"),
    "ticket": os.environ.get("BOHRIUM_TICKET"),
    "upload_progress": os.environ.get("BOHRIUM_UPLOAD_PROGRESS") in [
        "true", "1"],
    "access_key": os.environ.get("BOHRIUM_ACCESS_KEY"),
    "openapi_url": os.environ.get("BOHRIUM_OPENAPI_URL",
                                  "https://openapi.dp.tech"),
    "app_key": os.environ.get("BOHRIUM_APP_KEY", "agent"),
}


def _raise_error(res, op):
    if res["code"] not in succ_code:
        if "error" in res:
            if isinstance(res["error"], str):
                raise RuntimeError("%s failed: %s" % (op, res["error"]))
            else:
                raise RuntimeError("%s failed: %s" % (op, res["error"]["msg"]))
        elif "message" in res:
            raise RuntimeError("%s failed: %s" % (op, res["message"]))
        else:
            raise RuntimeError("%s failed" % op)


def login(username=None, phone=None, password=None, bohrium_url=None):
    if username is None:
        username = config["username"]
    if phone is None:
        phone = config["phone"]
    if password is None:
        password = config["password"]
    if bohrium_url is None:
        bohrium_url = config["bohrium_url"]
    authorization = _login(
        bohrium_url + "/account/login", username, phone, password)
    return authorization


def _login(login_url=None, username=None, phone=None, password=None):
    data = {
        "username": username,
        "phone": phone,
        "password": password,
    }
    rsp = requests.post(login_url, headers={
                        "Content-type": "application/json"}, json=data)
    res = json.loads(rsp.text)
    _raise_error(res, "login")
    return res["data"]["token"]


class BohriumStorage(BaseStorage):
    def __init__(
            self,
            bohrium_url: Optional[str] = None,
            username: Optional[str] = None,
            phone: Optional[str] = None,
            password: Optional[str] = None,
            authorization: Optional[str] = None,
            project_id: Optional[str] = None,
            token: Optional[str] = None,
            prefix: Optional[str] = None,
            sharePath: Optional[str] = None,
            userSharePath: Optional[str] = None,
            tiefblue_url: Optional[str] = None,
            ticket: Optional[str] = None,
            access_key: Optional[str] = None,
            openapi_url: Optional[str] = None,
            app_key: Optional[str] = None,
            max_upload_size: Optional[int] = None,
            max_download_size: Optional[int] = None,
            **kwargs,
    ) -> None:
        """Bohrium storage interface

        Args:
            bohrium_url: The base URL of bohrium, https://bohrium.dp.tech by
                default
            username: The username of bohrium
            password: The password of bohrium
            authorization: The login authorization of bohrium
            project_id: The project ID of bohrium
            prefix: Artifact storage prefix in user's personal storage or
                project storage
            ticket: The ticket of bohrium
            max_upload_size: Max upload size in bytes; None = no limit
            max_download_size: Max download size in bytes; None = no limit
        """
        super().__init__(max_upload_size=max_upload_size, max_download_size=max_download_size)
        self.bohrium_url = bohrium_url if bohrium_url is not None else \
            config["bohrium_url"]
        self.username = username if username is not None else \
            config["username"]
        self.phone = phone if phone is not None else config["phone"]
        self.password = password if password is not None else \
            config["password"]
        self.authorization = authorization if authorization is not None else \
            config["authorization"]
        self.ticket = ticket if ticket is not None else config["ticket"]
        self.project_id = project_id if project_id is not None else \
            config["project_id"]
        self.tiefblue_url = tiefblue_url if tiefblue_url is not None else \
            config["tiefblue_url"]
        self.token = token
        self.prefix = prefix
        self.sharePath = sharePath
        self.userSharePath = userSharePath
        self.access_key = access_key if access_key is not None else \
            config["access_key"]
        self.openapi_url = openapi_url if openapi_url is not None else \
            config["openapi_url"]
        self.app_key = app_key if app_key is not None else config["app_key"]
        if self.token is None:
            self.get_token()

    def get_token(self, retry=1):
        url = self.bohrium_url + "/brm/v1/storage/token"
        headers = {
            "Content-type": "application/json",
        }
        params = {
            "projectId": self.project_id,
        }
        if self.access_key is not None:
            url = self.openapi_url + "/openapi/v1/storage/token"
            headers = {"x-app-key": self.app_key}
            params["accessKey"] = self.access_key
        elif self.ticket is not None:
            headers["Brm-Ticket"] = config["ticket"]
        else:
            if self.authorization is None:
                self.authorization = login(
                    self.username, self.phone, self.password, self.bohrium_url)
            headers["Authorization"] = "Bearer " + self.authorization
        rsp = requests.get(url, headers=headers, params=params)
        if not rsp.text:
            if retry > 0:
                self.authorization = None
                self.get_token(retry=retry-1)
                return
            raise RuntimeError("Bohrium unauthorized")
        res = json.loads(rsp.text)
        _raise_error(res, "get storage token")
        self.token = res["data"]["token"]
        self.prefix = res["data"]["path"]
        self.sharePath = res["data"]["sharePath"]
        self.userSharePath = res["data"]["userSharePath"]

    def prefixing(self, key):
        if not key.startswith(self.prefix):
            return self.prefix + key
        return key

    def _upload(self, key, path):
        key = self.prefixing(key)
        client = tiefblue.Client(base_url=self.tiefblue_url, token=self.token)
        try:
            client.upload_from_file(
                key, path, progress_bar=config["upload_progress"])
        except tiefblue.client.TiefblueException as e:
            if e.code == 190001:
                self.get_token()
                client = tiefblue.Client(base_url=self.tiefblue_url,
                                         token=self.token)
                client.upload_from_file(
                    key, path, progress_bar=config["upload_progress"])
            else:
                raise e
        return key

    def _download(self, key, path):
        key = self.prefixing(key)
        client = tiefblue.Client(base_url=self.tiefblue_url, token=self.token)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            client.download_from_file(key, path)
        except tiefblue.client.TiefblueException as e:
            if e.code == 190001:
                self.get_token()
                client = tiefblue.Client(base_url=self.tiefblue_url,
                                         token=self.token)
                client.download_from_file(key, path)
            else:
                raise e
        return path

    def list(self, prefix, recursive=False):
        prefix = self.prefixing(prefix)
        client = tiefblue.Client(base_url=self.tiefblue_url, token=self.token)
        keys = []
        next_token = ""
        while True:
            try:
                res = client.list(prefix=prefix, recursive=recursive,
                                  next_token=next_token)
            except tiefblue.client.TiefblueException as e:
                if e.code == 190001:
                    self.get_token()
                    client = tiefblue.Client(base_url=self.tiefblue_url,
                                             token=self.token)
                    res = client.list(prefix=prefix, recursive=recursive,
                                      next_token=next_token)
                else:
                    raise e
            for obj in res["objects"]:
                if (recursive or obj["path"] == prefix) and \
                        obj["path"].endswith("/"):
                    continue
                keys.append(obj["path"])
            if not res["hasNext"]:
                break
            next_token = res["nextToken"]
        return keys

    def copy(self, src, dst):
        src = self.prefixing(src)
        dst = self.prefixing(dst)
        client = tiefblue.Client(base_url=self.tiefblue_url, token=self.token)
        try:
            client.copy(src, dst)
        except tiefblue.client.TiefblueException as e:
            if e.code == 190001:
                self.get_token()
                client = tiefblue.Client(base_url=self.tiefblue_url,
                                         token=self.token)
                client.copy(src, dst)
            else:
                raise e

    def get_md5(self, key):
        key = self.prefixing(key)
        client = tiefblue.Client(base_url=self.tiefblue_url, token=self.token)
        try:
            meta = client.meta(key)
        except tiefblue.client.TiefblueException as e:
            if e.code == 190001:
                self.get_token()
                client = tiefblue.Client(base_url=self.tiefblue_url,
                                         token=self.token)
                meta = client.meta(key)
            else:
                raise e
        return meta["entityTag"] if "entityTag" in meta else ""

    def get_size(self, key: str) -> Optional[int]:
        key = self.prefixing(key)
        client = tiefblue.Client(base_url=self.tiefblue_url, token=self.token)
        try:
            meta = client.meta(key)
        except tiefblue.client.TiefblueException as e:
            if e.code == 190001:
                self.get_token()
                client = tiefblue.Client(base_url=self.tiefblue_url,
                                         token=self.token)
                meta = client.meta(key)
            else:
                raise e
        return meta.get("size") or meta.get("contentLength")

    def get_http_url(self, key):
        key = self.prefixing(key)
        url = self.tiefblue_url + "/api/setacl"
        headers = {
            "Content-type": "application/json",
            "Authorization": "Bearer " + self.token,
        }
        params = {
            "acl": "public-read",
            "path": key,
        }
        rsp = requests.post(url, headers=headers, json=params)
        res = json.loads(rsp.text)
        _raise_error(res, "get HTTP url")
        return res["data"]["url"]
