from openmockllm.mistral.exceptions import BadRequestError, NotFoundError
from openmockllm.utils import check_max_context_length as _check_max_context_length


def check_model_not_found(called_model: str, current_model: str):
    if called_model != current_model:
        raise NotFoundError


def check_max_context_length(prompt: str, max_context_length: int):
    if not _check_max_context_length(prompt=prompt, max_context_length=max_context_length):
        raise BadRequestError
