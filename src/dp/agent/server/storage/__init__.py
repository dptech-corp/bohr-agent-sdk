from .base_storage import BaseStorage, StorageLimitExceededError
from .bohrium_storage import BohriumStorage
from .local_storage import LocalStorage
from .oss_storage import OSSStorage
from .http_storage import HTTPStorage, HTTPSStorage

__all__ = ["BaseStorage", "StorageLimitExceededError"]
storage_dict = {
    "bohrium": BohriumStorage,
    "local": LocalStorage,
    "oss": OSSStorage,
    "http": HTTPStorage,
    "https": HTTPSStorage,
}
