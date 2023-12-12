"""A module to print to the IPython notebook."""

from typing import Callable, Union

from IPython.display import display, Markdown, Latex, TextDisplayObject


class IPythonPrinter:
    """A class to print to the IPython notebook."""

    cache: list[Union[TextDisplayObject, str]] = []
    last_type: Union[Callable[..., TextDisplayObject], None] = None
    last: str = ""

    def _clear_last(self):
        """Clears the last string."""
        self.last_type = None
        self.last = ""

    def clear(self):
        """Clears the cache."""
        self.cache = []
        self._clear_last()

    def _instantiate_and_append(self):
        """Appends the last string to the cache as the last type."""
        if self.last_type is not None:
            self.cache.append(self.last_type(self.last))
        self._clear_last()

    def flush(self):
        """Outputs the cache to the notebook and clears it."""
        self._instantiate_and_append()
        display(*self.cache)
        self.clear()

    def print(self, string: str, flush: bool = False, no_new_line: bool = False):
        """Prints a string to the notebook."""
        if self.last_type is not str and self.last_type is not None:
            self._instantiate_and_append()

        self.last_type = str
        self.last += string + ("\n" if not no_new_line else "")

        if flush:
            self.flush()

    def print_latex(self, string: str, flush: bool = False, no_new_line: bool = False):
        """Prints a LaTeX string to the notebook."""
        if self.last_type is not Latex and self.last_type is not None:
            self._instantiate_and_append()

        self.last_type = Latex
        self.last += string + ("\n" if not no_new_line else "")

        if flush:
            self.flush()

    def print_markdown(
        self, string: str, flush: bool = False, no_new_line: bool = False
    ):
        """Prints a Markdown string to the notebook."""
        if self.last_type is not Markdown and self.last_type is not None:
            self._instantiate_and_append()

        self.last_type = Markdown
        self.last += string + ("\n" if not no_new_line else "")

        if flush:
            self.flush()
