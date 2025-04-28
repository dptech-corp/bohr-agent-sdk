from .base_storage import BaseStorage
from .bohrium_storage import BohriumStorage
from .local_storage import LocalStorage
from .oss_storage import OSSStorage

__all__ = ["BaseStorage"]
storage_dict = {
    "bohrium": BohriumStorage,
    "local": LocalStorage,
    "oss": OSSStorage,
}
