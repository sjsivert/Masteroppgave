import hashlib
from pathlib import Path


def generate_file_hash(path: Path) -> str:
    hash_sha1 = hashlib.sha1()
    # Split into chunks to combat high use of memory
    chunk_size = 4096
    with open(path, "rb") as f:
        chunk = f.read(chunk_size)
        while len(chunk) > 0:
            hash_sha1.update(chunk)
            chunk = f.read(chunk_size)
    return hash_sha1.hexdigest()
