

def highlight_source_bash(filename):
    """For use inside an IPython notebook: given a filename, print the source code. Bash version."""

    from pygments import highlight
    from pygments.lexers import BashLexer
    from pygments.formatters import HtmlFormatter
    from IPython.core.display import HTML

    with open (filename, "r") as myfile:
        data = myfile.read()

    return HTML(highlight(data, BashLexer(), HtmlFormatter(full=True)))