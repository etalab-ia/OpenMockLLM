from typing import Any

from openmockllm.vllm.schemas.core import VllmBaseModel


class FunctionCall(VllmBaseModel):
    """Deprecated function call format (use tool_calls instead)"""

    name: str
    arguments: str


class ToolCall(VllmBaseModel):
    """Tool call in assistant message"""

    id: str
    type: str
    function: dict[str, Any]


class Message(VllmBaseModel):
    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    function_call: FunctionCall | None = None  # Deprecated
    refusal: str | None = None


class ChatRequest(VllmBaseModel):
    # Required fields
    messages: list[Message]

    # Model selection
    model: str | None = None

    # Standard OpenAI parameters
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = 1
    stream: bool | None = False
    stop: str | list[str] | None = []
    max_tokens: int | None = None  # deprecated but still supported
    max_completion_tokens: int | None = None
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    user: str | None = None

    # Logprobs parameters
    logprobs: bool | None = False
    top_logprobs: int | None = 0

    # Response format and tools
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = "none"
    parallel_tool_calls: bool | None = False

    # Reproducibility
    seed: int | None = None

    # Streaming options
    stream_options: dict[str, Any] | None = None

    # vLLM specific parameters
    best_of: int | None = None
    use_beam_search: bool | None = False
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    length_penalty: float | None = 1.0
    stop_token_ids: list[int] | None = []
    include_stop_str_in_output: bool | None = False
    ignore_eos: bool | None = False
    min_tokens: int | None = 0
    skip_special_tokens: bool | None = True
    spaces_between_special_tokens: bool | None = True
    truncate_prompt_tokens: int | None = None
    prompt_logprobs: int | None = None

    # Token control
    allowed_token_ids: list[int] | None = None
    bad_words: list[str] | None = None

    # Message processing
    echo: bool | None = False
    add_generation_prompt: bool | None = True
    continue_final_message: bool | None = False
    add_special_tokens: bool | None = False

    # RAG and context
    documents: list[dict[str, Any]] | None = None

    # Template control
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    # Multimodal
    mm_processor_kwargs: dict[str, Any] | None = None

    # Guided decoding
    guided_json: str | dict[str, Any] | None = None
    guided_regex: str | None = None
    guided_choice: list[str] | None = None
    guided_grammar: str | None = None
    structural_tag: str | None = None
    guided_decoding_backend: str | None = None
    guided_whitespace_pattern: str | None = None

    # Advanced options
    priority: int | None = 0
    request_id: str | None = None
    logits_processors: list[Any] | None = None
    return_tokens_as_token_ids: bool | None = None
    cache_salt: str | None = None
    kv_transfer_params: dict[str, Any] | None = None
    vllm_xargs: dict[str, Any] | None = None


class LogprobContent(VllmBaseModel):
    """Log probability information for a token"""

    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[dict[str, Any]] | None = None


class ChoiceLogprobs(VllmBaseModel):
    """Log probability information for the choice"""

    content: list[LogprobContent] | None = None
    refusal: list[LogprobContent] | None = None


class ChatResponseChoice(VllmBaseModel):
    index: int
    message: Message
    finish_reason: str | None = None
    logprobs: ChoiceLogprobs | None = None
    stop_reason: int | str | None = None  # vLLM specific


class CompletionTokensDetails(VllmBaseModel):
    """Breakdown of completion tokens"""

    reasoning_tokens: int | None = None
    accepted_prediction_tokens: int | None = None
    rejected_prediction_tokens: int | None = None


class PromptTokensDetails(VllmBaseModel):
    """Breakdown of prompt tokens"""

    cached_tokens: int | None = None
    audio_tokens: int | None = None


class Usage(VllmBaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


class ChatResponse(VllmBaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatResponseChoice]
    usage: Usage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


class StreamDelta(VllmBaseModel):
    """Delta object in streaming response"""

    role: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    function_call: dict[str, Any] | None = None
    refusal: str | None = None


class ChatStreamResponseChoice(VllmBaseModel):
    index: int
    delta: StreamDelta
    finish_reason: str | None = None
    logprobs: ChoiceLogprobs | None = None
    stop_reason: int | str | None = None  # vLLM specific


class ChatStreamResponse(VllmBaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatStreamResponseChoice]
    usage: Usage | None = None  # Present in final chunk when stream_options.include_usage=true
    system_fingerprint: str | None = None
    service_tier: str | None = None
