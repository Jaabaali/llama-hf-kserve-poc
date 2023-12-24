import kserve
from transformers import AutoTokenizer, pipeline
import torch

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Configure transformers logging to show download progress
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)

class KServeLlamaModel(kserve.Model):

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.ready = True

    def predict(self, request: dict, headers: dict) -> dict:
        input_text = request["text"]
        sequences = self.pipeline(
            input_text,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=200,
        )
        return {"generated_text": sequences[0]['generated_text']}

if __name__ == "__main__":
    model = KServeLlamaModel("llama-chat")
    model.load()
    server = kserve.ModelServer(workers=3)
    server.start([model])
