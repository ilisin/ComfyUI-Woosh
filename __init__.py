from .nodes import WooshGenerate

NODE_CLASS_MAPPINGS = {"WooshLocalGenerate": WooshGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"WooshLocalGenerate": "Woosh 音效 / 环境声生成"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
