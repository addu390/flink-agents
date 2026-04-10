################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
from typing import ClassVar, List

try:
    from typing import override
except ImportError:
    from typing_extensions import override
from uuid import UUID

from flink_agents.api.agents.types import OutputSchema
from flink_agents.api.chat_message import ChatMessage
from flink_agents.api.events.event import Event


class ChatRequestEvent(Event):
    """Event representing a request to chat model.

    Attributes:
    ----------
    model : str
        The name of the chat model to be chatted with.
    messages : List[ChatMessage]
        The input to the chat model.
    output_schema: OutputSchema | None
        The expected output schema of the chat model final response. Optional.
    """

    EVENT_TYPE: ClassVar[str] = "_chat_request_event"

    def __init__(
        self,
        model: str,
        messages: List[ChatMessage],
        output_schema: OutputSchema | None = None,
    ) -> None:
        super().__init__(
            type=ChatRequestEvent.EVENT_TYPE,
            attributes={
                "model": model,
                "messages": messages,
                "output_schema": output_schema,
            },
        )

    @classmethod
    @override
    def from_event(cls, event: Event) -> "ChatRequestEvent":
        assert "model" in event.attributes
        assert "messages" in event.attributes
        return ChatRequestEvent(
            model=event.attributes["model"],
            messages=event.attributes["messages"],
            output_schema=event.attributes.get("output_schema"),
        )

    @property
    def model(self) -> str:
        return self.attributes["model"]

    @property
    def messages(self) -> List[ChatMessage]:
        return self.attributes["messages"]

    @property
    def output_schema(self) -> OutputSchema | None:
        return self.attributes.get("output_schema")


class ChatResponseEvent(Event):
    """Event representing a response from chat model.

    Attributes:
    ----------
    request_id : UUID
        The id of the request event.
    response : ChatMessage
        The response from the chat model.
    retry_count : int
        The total number of retries across all tool call rounds.
    total_retry_wait_sec : int
        The total time spent waiting during retries in seconds.
    """

    EVENT_TYPE: ClassVar[str] = "_chat_response_event"

    def __init__(
        self,
        request_id: UUID,
        response: ChatMessage,
        retry_count: int = 0,
        total_retry_wait_sec: int = 0,
    ) -> None:
        super().__init__(
            type=ChatResponseEvent.EVENT_TYPE,
            attributes={
                "request_id": request_id,
                "response": response,
                "retry_count": retry_count,
                "total_retry_wait_sec": total_retry_wait_sec,
            },
        )

    @classmethod
    @override
    def from_event(cls, event: Event) -> "ChatResponseEvent":
        assert "request_id" in event.attributes
        assert "response" in event.attributes
        return ChatResponseEvent(
            request_id=event.attributes["request_id"],
            response=event.attributes["response"],
            retry_count=event.attributes.get("retry_count", 0),
            total_retry_wait_sec=event.attributes.get("total_retry_wait_sec", 0),
        )

    @property
    def request_id(self) -> UUID:
        return self.attributes["request_id"]

    @property
    def response(self) -> ChatMessage:
        return self.attributes["response"]

    @property
    def retry_count(self) -> int:
        return self.attributes.get("retry_count", 0)

    @property
    def total_retry_wait_sec(self) -> int:
        return self.attributes.get("total_retry_wait_sec", 0)
