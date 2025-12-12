import asyncio
import base64
from collections.abc import AsyncGenerator
from pathlib import Path
import random

from faker import Faker
import tiktoken

from openmockllm.settings import settings

UTILS_DIR = Path(__file__).parent  # The directory where this file is located

tokenizer = tiktoken.get_encoding(settings.tiktoken_encoder)
fake = Faker(settings.faker_langage)
fake.seed_instance(settings.faker_seed)


def get_base64_jpeg_image() -> str:
    # Use absolute path based on the package directory
    image_path = UTILS_DIR / "assets" / "ocr.jpg"
    with open(file=image_path, mode="rb") as f:
        bytes = f.read()
        image = base64.b64encode(bytes).decode("utf-8")
        image = f"data:image/jpeg;base64,{image}"
    return image


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def check_max_context_length(prompt: str, max_context_length: int) -> int:
    return len(tokenizer.encode(prompt)) <= max_context_length


def generate_text(input_tokens: int, max_tokens: int | None = None) -> str:
    """
    Generate text based on the input tokens and the max tokens.

    Args:
        input_tokens (int): Number of tokens in the input prompt.
        max_tokens (int | None): Number of tokens to be generated. Default is random between 100 and 1000.

    Returns:
        str: Generated text.
    """

    max_tokens = max_tokens if max_tokens is not None else random.randint(100, 1000)

    # Number of tokens generated ≈ proportional to max_tokens
    # + small boost if prompt long
    prompt_boost = min(1.5, 1 + input_tokens / 2000)  # +50% max
    target_chars = int(max_tokens * 4 * prompt_boost)
    target_chars = min(target_chars, 8000)

    # Number of paragraphs based mostly on the generated length
    num_paragraphs = max(1, target_chars // 800)

    response_parts = []
    current_length = 0

    while current_length < target_chars and len(response_parts) < num_paragraphs:
        text = fake.paragraph(nb_sentences=random.randint(2, 6))
        response_parts.append(text)
        current_length += len(text)

    return "\n\n".join(response_parts)


def get_realistic_ttft(input_tokens: int, inflight_requests: int = 1) -> float:
    """
    Compute a realistic Time To First Token (TTFT) using a normal distribution.

    Inspired by behaviors observed in major providers (OpenAI, Anthropic, etc.).
    TTFT includes request preparation time, scheduling, and the start of the first forward pass.

    Assumptions:
    - We start from a base TTFT (settings.reference_ttft_mean), defined for
      a "medium" prompt (e.g., ~500 tokens) and 1 parallel request.
    - The variance is approximately 30% of this base TTFT.
    - Prompt size slightly increases TTFT (context preparation).
    - Concurrent requests increase TTFT in a quasi-linear fashion.

    Args:
        input_tokens (int): Number of tokens in the prompt.
        inflight_requests (int): Number of concurrent requests (>= 1).

    Returns:
        float: Realistic TTFT in seconds (>= 1 ms).
    """

    # 1) Base TTFT (for a "reference" size prompt and 1 request)
    # Ex: settings.reference_ttft_mean = 0.6  # 600 ms
    base_ttft_mean = settings.reference_ttft_mean  # in seconds

    # Base variance (~30% of TTFT)
    base_ttft_std = base_ttft_mean * 0.30

    # 2) Input size effect
    # We assume that beyond a certain threshold, context preparation
    # costs a bit more (but in a sub-linear manner).
    reference_prompt_tokens = getattr(settings, "reference_prompt_tokens", 500)

    # Overhead factor: for each "block" of prompt_reference, we add ~10% TTFT
    # clamped to avoid becoming absurd on gigantic prompts
    size_factor = min(2.0, 1.0 + 0.10 * max(0, (input_tokens - reference_prompt_tokens) / reference_prompt_tokens))

    size_adjusted_mean = base_ttft_mean * size_factor
    size_adjusted_std = base_ttft_std * size_factor  # variance also increases slightly

    # 3) Concurrent requests effect (queue):
    # Each additional request increases TTFT by approximately 15–25% of base TTFT.
    # We use 20% here.
    concurrency_factor_per_request = 0.20

    if inflight_requests <= 1:
        queue_delay_mean = 0.0
        queue_delay_std = 0.0
    else:
        # Base TTFT being a typical "wait time", we use it as a unit
        queue_delay_mean = (inflight_requests - 1) * base_ttft_mean * concurrency_factor_per_request
        queue_delay_std = queue_delay_mean * 0.30  # queue more variable

    # 4) Distribution combination (total TTFT = size + queue)
    total_mean = size_adjusted_mean + queue_delay_mean
    total_std = (size_adjusted_std**2 + queue_delay_std**2) ** 0.5

    # 5) Gaussian sampling, protection against negative values
    ttft = random.gauss(total_mean, total_std)
    return max(0.001, ttft)  # min 1 ms to avoid 0


def get_realistic_itl(output_tokens: int, inflight_requests: int = 1) -> float:
    """
    Compute realistic Inter-Token Latency (ITL) using normal distribution.

    Based on benchmarks from major LLM providers (OpenAI GPT-4, Anthropic Claude, Meta LLaMA).
    The parameters are derived from reference_output_tokens_per_second.

    Args:
        output_tokens (int): Number of tokens to be generated.
        inflight_requests (int): Number of concurrent requests.

    Returns:
        Realistic Inter-Token Latency (nTL) in seconds.
    """
    # Reference throughput for generation
    reference_throughput = settings.reference_output_tokens_per_second

    # Average time per token
    time_per_token_mean = 1.0 / reference_throughput
    # Variance (generation = less variable than input)
    time_per_token_std = time_per_token_mean * 0.10

    # Total generation time
    generation_mean = output_tokens * time_per_token_mean
    generation_std = output_tokens * time_per_token_std

    # Queue delay: stronger than for input
    # +25% per additional concurrent request (realistic approximation)
    reference_processing_time = output_tokens / reference_throughput
    queue_delay_mean = (inflight_requests - 1) * reference_processing_time * 0.25
    queue_delay_std = (inflight_requests - 1) * reference_processing_time * 0.10

    # Combine variances
    total_mean = generation_mean + queue_delay_mean
    total_std = (generation_std**2 + queue_delay_std**2) ** 0.5

    # Gaussian sample, avoid negative values
    itl = random.gauss(total_mean, total_std)
    return max(0.001, itl)


async def generate_unstreamed_chat_content(prompt: str, max_tokens: int | None = None) -> str:
    input_tokens = count_tokens(prompt)
    text = generate_text(input_tokens=input_tokens, max_tokens=max_tokens)

    if settings.simulate_latency:
        ttft = get_realistic_ttft(input_tokens=input_tokens, inflight_requests=1)
        await asyncio.sleep(ttft)
        completion_tokens = count_tokens(text)
        itl = get_realistic_itl(output_tokens=completion_tokens, inflight_requests=1)
        await asyncio.sleep(itl * completion_tokens)

    return text


async def generate_stream_chat_content(prompt: str, max_tokens: int | None = None) -> AsyncGenerator[str, None]:
    input_tokens = count_tokens(prompt)
    text = generate_text(input_tokens=input_tokens, max_tokens=max_tokens)

    chunks = text.split(" ")
    itl = 0.0  # Default ITL when latency simulation is disabled
    if settings.simulate_latency:
        ttft = get_realistic_ttft(input_tokens=input_tokens, inflight_requests=1)
        await asyncio.sleep(ttft)
        completion_tokens = count_tokens(text)
        itl = get_realistic_itl(output_tokens=completion_tokens, inflight_requests=1)

    for chunk in chunks:
        if settings.simulate_latency:
            await asyncio.sleep(itl)
        yield f"{chunk}\n\n"

    yield "[DONE]\n\n"
