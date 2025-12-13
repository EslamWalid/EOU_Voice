from livekit.agents import Plugin
from .runner import ArabicEOURunner

class ArabicEOUPlugin(Plugin):
    def __init__(self):
        super().__init__("arabic_eou", "1.0.0", __package__)
        self._runner_class = ArabicEOURunner()

# Register plugin
plugin = ArabicEOUPlugin()
