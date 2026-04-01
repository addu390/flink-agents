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
import logging
from datetime import datetime
from typing import Any, Dict, List, Type

from pydantic import ConfigDict, PrivateAttr
from typing_extensions import override

from flink_agents.api.chat_message import ChatMessage
from flink_agents.api.memory.long_term_memory import (
    CompactionConfig,
    DatetimeRange,
    ItemType,
    MemorySet,
    MemorySetItem,
)
from flink_agents.api.vector_stores.vector_store import _maybe_cast_to_list
from flink_agents.runtime.memory.internal_base_long_term_memory import (
    InternalBaseLongTermMemory,
)
from flink_agents.runtime.python_java_utils import from_java_chat_message

logger = logging.getLogger(__name__)


class JavaLongTermMemory(InternalBaseLongTermMemory):
    """Long-Term Memory implementation that delegates to Java's
    VectorStoreLongTermMemory.

    This class provides Python-side compatibility by wrapping the Java LTM
    implementation, performing parameter and result conversion as needed.
    All actual storage and compaction operations are handled by the Java side.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _j_ltm: Any = PrivateAttr()
    _j_resource_adapter: Any = PrivateAttr()
    _j_memory_sets: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(
        self, *, j_ltm: Any, j_resource_adapter: Any, **kwargs: Any
    ) -> None:
        """Initialize with a Java VectorStoreLongTermMemory instance.

        Args:
            j_ltm: The Java VectorStoreLongTermMemory object (Pemja handle).
            j_resource_adapter: The Java JavaResourceAdapter for type
                conversion helpers.
        """
        super().__init__(**kwargs)
        self._j_ltm = j_ltm
        self._j_resource_adapter = j_resource_adapter
        self._j_memory_sets = {}

    @override
    def switch_context(self, key: str) -> None:
        self._j_ltm.switchContext(key)

    @override
    def get_or_create_memory_set(
        self,
        name: str,
        item_type: type[str] | Type[ChatMessage],
        capacity: int,
        compaction_config: CompactionConfig,
    ) -> MemorySet:
        # Convert Python type to Java Class<?> via adapter
        j_item_type = self._j_resource_adapter.resolveMemoryItemType(
            _python_type_to_name(item_type)
        )
        # Convert Python CompactionConfig to Java CompactionConfig
        j_cc = self._j_resource_adapter.createCompactionConfig(
            compaction_config.model,
            compaction_config.prompt
            if isinstance(compaction_config.prompt, str)
            else None,
            compaction_config.limit,
        )
        # Call Java LTM
        j_memory_set = self._j_ltm.getOrCreateMemorySet(
            name, j_item_type, capacity, j_cc
        )
        # Store Java reference for later pass-back
        self._j_memory_sets[name] = j_memory_set

        return MemorySet(
            name=name,
            item_type=item_type,
            capacity=capacity,
            compaction_config=compaction_config,
            ltm=self,
        )

    @override
    def get_memory_set(self, name: str) -> MemorySet:
        j_memory_set = self._j_ltm.getMemorySet(name)
        self._j_memory_sets[name] = j_memory_set

        # Extract fields from Java MemorySet via getters
        item_type = _java_type_name_to_python(
            j_memory_set.getItemType().getName()
        )
        capacity = j_memory_set.getCapacity()
        j_cc = j_memory_set.getCompactionConfig()
        compaction_config = CompactionConfig(
            model=j_cc.getModel(),
            prompt=j_cc.getPrompt(),
            limit=j_cc.getLimit(),
        )

        return MemorySet(
            name=name,
            item_type=item_type,
            capacity=capacity,
            compaction_config=compaction_config,
            ltm=self,
        )

    @override
    def delete_memory_set(self, name: str) -> bool:
        result = self._j_ltm.deleteMemorySet(name)
        self._j_memory_sets.pop(name, None)
        return result

    @override
    def size(self, memory_set: MemorySet) -> int:
        j_memory_set = self._get_java_memory_set(memory_set)
        return self._j_ltm.size(j_memory_set)

    @override
    def add(
        self,
        memory_set: MemorySet,
        memory_items: ItemType | List[ItemType],
        ids: str | List[str] | None = None,
        metadatas: Dict[str, Any] | List[Dict[str, Any]] | None = None,
    ) -> List[str]:
        memory_items = _maybe_cast_to_list(memory_items)
        ids = _maybe_cast_to_list(ids)
        metadatas = _maybe_cast_to_list(metadatas)

        j_memory_set = self._get_java_memory_set(memory_set)

        # Convert ChatMessage items to Java ChatMessage objects
        if memory_set.item_type == ChatMessage:
            j_items = [
                self._j_resource_adapter.fromPythonChatMessage(item)
                for item in memory_items
            ]
        else:
            j_items = list(memory_items)

        return list(
            self._j_ltm.add(j_memory_set, j_items, ids, metadatas)
        )

    @override
    def get(
        self, memory_set: MemorySet, ids: str | List[str] | None = None
    ) -> List[MemorySetItem]:
        ids = _maybe_cast_to_list(ids)
        j_memory_set = self._get_java_memory_set(memory_set)
        j_items = self._j_ltm.get(j_memory_set, ids)
        return [
            self._from_java_memory_set_item(j_item, memory_set.item_type)
            for j_item in j_items
        ]

    @override
    def delete(
        self, memory_set: MemorySet, ids: str | List[str] | None = None
    ) -> None:
        ids = _maybe_cast_to_list(ids)
        j_memory_set = self._get_java_memory_set(memory_set)
        self._j_ltm.delete(j_memory_set, ids)

    @override
    def search(
        self, memory_set: MemorySet, query: str, limit: int, **kwargs: Any
    ) -> List[MemorySetItem]:
        j_memory_set = self._get_java_memory_set(memory_set)
        j_items = self._j_ltm.search(
            j_memory_set, query, limit, dict(kwargs)
        )
        return [
            self._from_java_memory_set_item(j_item, memory_set.item_type)
            for j_item in j_items
        ]

    @override
    def close(self) -> None:
        # Java side manages its own lifecycle via RunnerContextImpl.close().
        # Do NOT call j_ltm.close() here to avoid double-close.
        pass

    def _get_java_memory_set(self, memory_set: MemorySet) -> Any:
        """Get the stored Java MemorySet reference for the given Python
        MemorySet."""
        j_ms = self._j_memory_sets.get(memory_set.name)
        if j_ms is None:
            err_msg = (
                f"Java MemorySet reference not found for "
                f"'{memory_set.name}'. Ensure get_or_create_memory_set() "
                f"or get_memory_set() was called first."
            )
            raise ValueError(err_msg)
        return j_ms

    def _from_java_memory_set_item(
        self, j_item: Any, item_type: type
    ) -> MemorySetItem:
        """Convert a Java MemorySetItem to a Python MemorySetItem."""
        # Convert value based on item type
        value = j_item.getValue()
        if item_type == ChatMessage:
            value = from_java_chat_message(value)
        # For str type, Pemja auto-converts Java String to Python str

        compacted = j_item.isCompacted()

        # Convert dates via Java-side helper (LocalDateTime → ISO string)
        if compacted:
            j_range = j_item.getCreatedTime()
            created_time = DatetimeRange(
                start=datetime.fromisoformat(
                    self._j_resource_adapter.formatDateTime(
                        j_range.getStart()
                    )
                ),
                end=datetime.fromisoformat(
                    self._j_resource_adapter.formatDateTime(
                        j_range.getEnd()
                    )
                ),
            )
        else:
            created_time = datetime.fromisoformat(
                self._j_resource_adapter.formatDateTime(
                    j_item.getCreatedTime()
                )
            )

        last_accessed_time = datetime.fromisoformat(
            self._j_resource_adapter.formatDateTime(
                j_item.getLastAccessedTime()
            )
        )

        metadata = j_item.getMetadata()
        # Pemja auto-converts Map<String, Object> to dict

        return MemorySetItem(
            memory_set_name=j_item.getMemorySetName(),
            id=j_item.getId(),
            value=value,
            compacted=compacted,
            created_time=created_time,
            last_accessed_time=last_accessed_time,
            additional_metadata=metadata if metadata else None,
        )


def _python_type_to_name(item_type: type) -> str:
    """Convert Python type to a string name for Java resolution."""
    if item_type == str:
        return "str"
    if item_type == ChatMessage:
        return "ChatMessage"
    err_msg = f"Unsupported LTM item type: {item_type}"
    raise ValueError(err_msg)


def _java_type_name_to_python(type_name: str) -> type:
    """Convert Java Class name to Python type."""
    if type_name == "java.lang.String":
        return str
    if "ChatMessage" in type_name:
        return ChatMessage
    err_msg = f"Unsupported Java type name: {type_name}"
    raise ValueError(err_msg)