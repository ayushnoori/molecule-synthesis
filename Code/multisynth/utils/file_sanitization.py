import string
from unicodedata import normalize
import os


def os_path_separators() -> list[str]:
    """Returns a list of path separators for the current operating system."""
    seps: list[str] = []
    for sep in os.path.sep, os.path.altsep:
        if sep:
            seps.append(sep)
    return seps


def sanitize_filesystem_name(potential_file_path_name: str) -> str:
    """Turns a string into a valid file name for the current operating system."""
    # Sort out unicode characters
    valid_filename = (
        normalize("NFKD", potential_file_path_name)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    # Replace path separators with underscores
    for sep in os_path_separators():
        valid_filename = valid_filename.replace(sep, "_")
    # Ensure only valid characters
    valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
    valid_filename = "".join(ch for ch in valid_filename if ch in valid_chars)
    # Ensure at least one letter or number to ignore names such as '..'
    valid_chars = f"{string.ascii_letters}{string.digits}"
    test_filename = "".join(ch for ch in potential_file_path_name if ch in valid_chars)
    if len(test_filename) == 0:
        # Replace empty file name or file path part with the following
        valid_filename = "(Empty Name)"
    return valid_filename
