from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextStreamer

from src.generator import LlamaGenerator

if __name__ == "__main__":
    model_id = "facebook/layerskip-llama3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    streamer = TextStreamer(tokenizer)
    generator = LlamaGenerator(model, tokenizer, streamer=streamer)
    prompt = "Once upon a time"
    response = generator.generate(prompt, max_length=50)
    num_tokens = response.num_tokens_generated
    print(f"Generated {num_tokens} tokens.")
    print(f"Total time: {response.total_time:.2f} seconds")
    print(f"Time per token: {response.time_per_token: .3f}s")
    print(f"Tokens per second: {response.tokens_per_second:.2f} tokens/s")
    streamer.end()
