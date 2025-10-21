from typing import Any, Dict, List, Optional, Union

from openmockllm.vllm.schemas.core import VllmBaseModel


class FunctionCall(VllmBaseModel):
    """Deprecated function call format (use tool_calls instead)"""

    name: str
    arguments: str


class ToolCall(VllmBaseModel):
    """Tool call in assistant message"""

    id: str
    type: str
    function: Dict[str, Any]


class Message(VllmBaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    function_call: Optional[FunctionCall] = None  # Deprecated
    refusal: Optional[str] = None


class ChatRequest(VllmBaseModel):
    # Required fields
    messages: List[Message]

    # Model selection
    model: Optional[str] = None

    # Standard OpenAI parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = []
    max_tokens: Optional[int] = None  # deprecated but still supported
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    # Logprobs parameters
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0

    # Response format and tools
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "none"
    parallel_tool_calls: Optional[bool] = False

    # Reproducibility
    seed: Optional[int] = None

    # Streaming options
    stream_options: Optional[Dict[str, Any]] = None

    # vLLM specific parameters
    best_of: Optional[int] = None
    use_beam_search: Optional[bool] = False
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = 1.0
    stop_token_ids: Optional[List[int]] = []
    include_stop_str_in_output: Optional[bool] = False
    ignore_eos: Optional[bool] = False
    min_tokens: Optional[int] = 0
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    truncate_prompt_tokens: Optional[int] = None
    prompt_logprobs: Optional[int] = None

    # Token control
    allowed_token_ids: Optional[List[int]] = None
    bad_words: Optional[List[str]] = None

    # Message processing
    echo: Optional[bool] = False
    add_generation_prompt: Optional[bool] = True
    continue_final_message: Optional[bool] = False
    add_special_tokens: Optional[bool] = False

    # RAG and context
    documents: Optional[List[Dict[str, Any]]] = None

    # Template control
    chat_template: Optional[str] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None

    # Multimodal
    mm_processor_kwargs: Optional[Dict[str, Any]] = None

    # Guided decoding
    guided_json: Optional[Union[str, Dict[str, Any]]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[List[str]] = None
    guided_grammar: Optional[str] = None
    structural_tag: Optional[str] = None
    guided_decoding_backend: Optional[str] = None
    guided_whitespace_pattern: Optional[str] = None

    # Advanced options
    priority: Optional[int] = 0
    request_id: Optional[str] = None
    logits_processors: Optional[List[Any]] = None
    return_tokens_as_token_ids: Optional[bool] = None
    cache_salt: Optional[str] = None
    kv_transfer_params: Optional[Dict[str, Any]] = None
    vllm_xargs: Optional[Dict[str, Any]] = None


class LogprobContent(VllmBaseModel):
    """Log probability information for a token"""

    token: str
    logprob: float
    bytes: Optional[List[int]] = None
    top_logprobs: Optional[List[Dict[str, Any]]] = None


class ChoiceLogprobs(VllmBaseModel):
    """Log probability information for the choice"""

    content: Optional[List[LogprobContent]] = None
    refusal: Optional[List[LogprobContent]] = None


class ChatResponseChoice(VllmBaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None
    logprobs: Optional[ChoiceLogprobs] = None
    stop_reason: Optional[Union[int, str]] = None  # vLLM specific


class CompletionTokensDetails(VllmBaseModel):
    """Breakdown of completion tokens"""

    reasoning_tokens: Optional[int] = None
    accepted_prediction_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None


class PromptTokensDetails(VllmBaseModel):
    """Breakdown of prompt tokens"""

    cached_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None


class Usage(VllmBaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[PromptTokensDetails] = None
    completion_tokens_details: Optional[CompletionTokensDetails] = None


class ChatResponse(VllmBaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatResponseChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None


class StreamDelta(VllmBaseModel):
    """Delta object in streaming response"""

    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None
    refusal: Optional[str] = None


class ChatStreamResponseChoice(VllmBaseModel):
    index: int
    delta: StreamDelta
    finish_reason: Optional[str] = None
    logprobs: Optional[ChoiceLogprobs] = None
    stop_reason: Optional[Union[int, str]] = None  # vLLM specific


class ChatStreamResponse(VllmBaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatStreamResponseChoice]
    usage: Optional[Usage] = None  # Present in final chunk when stream_options.include_usage=true
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None
