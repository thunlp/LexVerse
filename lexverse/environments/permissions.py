"""Task mount permissions and resource validation for Docker execution."""

READ_ONLY = ("/inputs", "/skills")
READ_WRITE = ("/workspace", "/artifacts")
RUNTIME = "/runtime"


def validate_roots(readonly, writable):
    """A writable parent would make a protected tree replaceable."""
    for path in (*readonly, *writable):
        if path.is_symlink():
            raise ValueError(f"Task resource root must not be a symlink: {path}")
    for protected in readonly:
        for path in writable:
            if protected.resolve().is_relative_to(path.resolve()) or path.resolve().is_relative_to(protected.resolve()):
                raise ValueError("Read-only resources and writable task directories must not overlap")
    for root in readonly:
        if any(path.is_symlink() or (path.is_file() and path.stat().st_nlink > 1) for path in root.rglob("*")):
            raise ValueError("Read-only task resources must not contain symbolic or hard links")
