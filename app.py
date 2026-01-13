# Root entrypoint for Hugging Face Spaces
from app.app import demo

if __name__ == "__main__":
    demo.launch()