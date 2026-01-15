from mistralai.models import ChatCompletionRequest as MistralChatCompletionRequest
from pydantic import ConfigDict


class ChatCompletionRequest(MistralChatCompletionRequest):
    model_config = ConfigDict(extra="forbid")
