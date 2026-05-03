"""Entry point: ``python -m examples.code_review``.

Launches the Streamlit UI for the code-review example app. Mirrors the
incident-management entrypoint so the two apps share the same boot shape.
"""
import sys
from streamlit.web.cli import main as streamlit_main


def main() -> None:
    # Streamlit's CLI expects argv[0] to be its own command and
    # argv[1] to be the script path.
    sys.argv = ["streamlit", "run", __file__.replace("__main__.py", "ui.py")]
    streamlit_main()


if __name__ == "__main__":
    main()
