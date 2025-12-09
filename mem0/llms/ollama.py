import json
from typing import Dict, List, Optional, Union

from openai import OpenAI

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.ollama import OllamaConfig
from mem0.llms.base import LLMBase


class OllamaLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, OllamaConfig, Dict]] = None):
        # Normalize config to OllamaConfig
        if config is None:
            config = OllamaConfig()
        elif isinstance(config, dict):
            config = OllamaConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, OllamaConfig):
            config = OllamaConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                http_client_proxies=config.http_client,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = "llama3.1:70b"

        base_url = self.config.ollama_base_url or "http://localhost:11434"
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        # api_key not required by Ollama, but OpenAI client expects one
        self.client = OpenAI(base_url=base_url, api_key=self.config.api_key or "ollama")

    def _parse_response(self, response, tools):
        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls = getattr(msg, "tool_calls", None) or []

        if tools and tool_calls:
            normalized_calls = []
            for tc in tool_calls:
                args_raw = tc.function.arguments
                try:
                    args_parsed = json.loads(args_raw)
                except Exception:
                    args_parsed = args_raw
                # If arguments contain nested JSON as string for entities, parse it
                if isinstance(args_parsed, dict):
                    if isinstance(args_parsed.get("entities"), str):
                        try:
                            args_parsed["entities"] = json.loads(args_parsed["entities"])
                        except Exception:
                            pass
                    if isinstance(args_parsed.get("relations"), str):
                        try:
                            args_parsed["relations"] = json.loads(args_parsed["relations"])
                        except Exception:
                            pass

                normalized_calls.append(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "name": tc.function.name,
                        "function": {
                            "name": tc.function.name,
                            "arguments": args_raw,
                        },
                        "arguments": args_parsed,
                    }
                )
            return {"content": content, "tool_calls": normalized_calls}

        return content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }

        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, tools)
