import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import transformers
from transformers.generation.streamers import TextStreamer


@dataclass
class GenerationStrategyResult:
    predicted_tokens: List[int]
    acceptance_rate: Optional[float] = None


@dataclass
class GenerationResult:
    decoded_prediction: str
    num_tokens_generated: int
    time_per_token: float
    total_time: float
    tokens_per_second: float


class LlamaGenerator:
    def __init__(
        self,
        model: transformers.LlamaForCausalLM,
        tokenizer: transformers.LlamaTokenizer,
        streamer: Optional[TextStreamer],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.streamer = streamer

    def generate(self, prompt, max_length=500) -> GenerationResult:
        """
        autoregressive generation
        """
        example = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        with torch.inference_mode():
            start = time.time()
            output_ids = self.model.generate(
                input_ids=example["input_ids"],
                attention_mask=example["attention_mask"],
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                streamer=self.streamer,
            )
            total_time = time.time() - start

        decoded_prediction = self.tokenizer.decode(
            output_ids[0], streamer=self.streamer, skip_special_tokens=True
        )
        num_tokens_generated = output_ids.shape[1] - example["input_ids"].shape[1]
        assert num_tokens_generated >= 0
        time_per_token = total_time / num_tokens_generated
        return GenerationResult(
            decoded_prediction=decoded_prediction,
            num_tokens_generated=num_tokens_generated,
            time_per_token=time_per_token,
            tokens_per_second=num_tokens_generated / total_time,
            total_time=total_time,
        )
