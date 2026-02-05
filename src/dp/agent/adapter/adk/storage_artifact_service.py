
import logging
import mimetypes
import os
import tempfile
from typing import Optional

from google.adk.artifacts import BaseArtifactService
from google.genai import types
from typing_extensions import override

from ...server.storage import BaseStorage

logger = logging.getLogger(__name__)


class StorageArtifactService(BaseArtifactService):
    """An artifact service implementation using storage plugin."""
    def __init__(self, storage: BaseStorage):
        self.storage = storage

    def _file_has_user_namespace(self, filename: str) -> bool:
        """Checks if the filename has a user namespace.

        Args:
            filename: The filename to check.

        Returns:
            True if the filename has a user namespace (starts with "user:"),
            False otherwise.
        """
        return filename.startswith("user:")

    def _get_key(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: int,
    ) -> str:
        """Constructs the key.

        Args:
            app_name: The name of the application.
            user_id: The ID of the user.
            session_id: The ID of the session.
            filename: The name of the artifact file.
            version: The version of the artifact.

        Returns:
            The constructed key.
        """
        if self._file_has_user_namespace(filename):
            return f"{app_name}/{user_id}/user/{filename}/{version}"
        return f"{app_name}/{user_id}/{session_id}/{filename}/{version}"

    @override
    async def save_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        artifact: types.Part,
    ) -> int:
        versions = await self.list_versions(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
        )
        version = 0 if not versions else max(versions) + 1

        key = self._get_key(
            app_name, user_id, session_id, filename, version
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, filename)
            with open(path, "wb") as f:
                f.write(artifact.inline_data.data)
            self.storage._check_upload_limit(path)
            self.storage._upload(key, path)

        return version

    @override
    async def load_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: Optional[int] = None,
    ) -> Optional[types.Part]:
        if version is None:
            versions = await self.list_versions(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
            )
            if not versions:
                return None
            version = max(versions)

        key = self._get_key(
            app_name, user_id, session_id, filename, version
        )
        self.storage._check_download_limit(key)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, filename)
            self.storage._download(key, path)
            with open(path, "rb") as f:
                artifact_bytes = f.read()

        mime_type, _ = mimetypes.guess_type(filename)
        artifact = types.Part.from_bytes(
            data=artifact_bytes, mime_type=mime_type
        )
        return artifact

    @override
    async def list_artifact_keys(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> list[str]:
        filenames = set()

        session_prefix = f"{app_name}/{user_id}/{session_id}/"
        keys = self.storage.list(session_prefix)
        for key in keys:
            _, _, _, filename, _ = key.split("/")[-5:]
            filenames.add(filename)

        user_namespace_prefix = f"{app_name}/{user_id}/user/"
        user_namespace_keys = self.storage.list(user_namespace_prefix)
        for key in user_namespace_keys:
            _, _, _, filename, _ = key.split("/")[-5:]
            filenames.add(filename)

        return sorted(list(filenames))

    @override
    async def delete_artifact(
        self, *, app_name: str, user_id: str, session_id: str, filename: str
    ) -> None:
        raise NotImplementedError()

    @override
    async def list_versions(
        self, *, app_name: str, user_id: str, session_id: str, filename: str
    ) -> list[int]:
        prefix = self._get_key(app_name, user_id, session_id, filename, "")
        keys = self.storage.list(prefix)
        versions = []
        for key in keys:
            _, _, _, _, version = key.split("/")[-5:]
            versions.append(int(version))
        return versions

    async def get_permanent_read_url(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: Optional[int] = None,
    ) -> str:
        if version is None:
            versions = await self.list_versions(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                filename=filename,
            )
            if not versions:
                return None
            version = max(versions)

        key = self._get_key(
            app_name, user_id, session_id, filename, version
        )
        return self.storage.get_http_url(key)
