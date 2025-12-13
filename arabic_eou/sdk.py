
from arabert.preprocess import ArabertPreprocessor

class ArabicEOUDetector:
    """
    Wraps your Hugging Face Arabic EOU classifier
    """
    def __init__(self, model_name="EslamWalid/bert-classifier"):
        from transformers import pipeline
        self.classifier = pipeline(model=model_name)
        self.preprocessor = ArabertPreprocessor("aubmindlab/bert-base-arabertv02")

    def will_end_utterance(self, text: str) -> float:
        text = self.preprocessor.preprocess(text)
        result = self.classifier(text)
        label, score = result[0]["label"], result[0]["score"]
        probability = score if label.upper() == "EOU" else 1 - score
        return float(probability)
