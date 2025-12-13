import json
from livekit.plugins.turn_detector.base import _EUORunnerBase, EOUModelBase, MAX_HISTORY_TURNS
from .sdk import ArabicEOUDetector
from .log import logger

class ArabicEOURunner(_EUORunnerBase, EOUModelBase):
    @classmethod
    def model_type(cls):
        return "arabic"
    
    def __init__(self, *args, **kwargs):
        kwargs["load_languages"] = False
        super().__init__(*args, **kwargs)
        # Initialize model and preprocessor here
        from transformers import pipeline
        from arabert.preprocess import ArabertPreprocessor

        self.classifier = pipeline(model="EslamWalid/bert-classifier")
        self.arabert_prep = ArabertPreprocessor("aubmindlab/bert-base-arabertv02")

    def _inference_method(self) -> str:
        # just return any string; usually the runner uses this to register the method
        return "lk_end_of_utterance_arabic"

    def initialize(self):
        # Initialize the SDK
        self.detector = ArabicEOUDetector()
        
        

    def _format_chat_ctx(self, chat_ctx: list[dict]) -> str:
        text = chat_ctx.items[-1].content[-1]
        logger.debug(f"Formatted chat context text: {text}")
        # texts = [msg.get("content", "") for msg in chat_ctx if msg.get("content")]
        return text

    async def supports_language(self, language: str | None) -> bool:
        """
        LiveKit calls this before using the turn detector.
        Return True only if your model supports the given language.
        """
        if not language:
            return False
        return language.lower() in ("ar", "ara")  # Arabic

    def run(self, data: bytes) -> bytes | None:
        data_json = json.loads(data)
        chat_ctx = data_json.get("chat_ctx", [])
        text = self._format_chat_ctx(chat_ctx)
        probability = self.detector.will_end_utterance(text)
        logger.debug(f"EOU probability for text '{text}': {(round(probability, 2))*100}%")
        return json.dumps({"eou_probability": probability, "input": text}).encode()
    
    async def predict_end_of_turn(
        self,
        chat_ctx: list[dict],
        *,
        timeout: float | None = 3,
    ) -> float:
        """
        LiveKit calls this to get probability of end-of-utterance.
        """
        text = self._format_chat_ctx(chat_ctx)
        text = self.arabert_prep.preprocess(text)

        result = self.classifier(text)
        label = result[0]["label"]
        score = result[0]["score"]
        probability = score if label == "EOU" else 1 - score
        logger.debug(f"EOU probability for text '{text}': {(round(probability, 2))*100}%")
        return float(probability)