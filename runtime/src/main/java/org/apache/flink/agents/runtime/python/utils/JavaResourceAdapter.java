/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.flink.agents.runtime.python.utils;

import org.apache.flink.agents.api.chat.messages.ChatMessage;
import org.apache.flink.agents.api.chat.messages.MessageRole;
import org.apache.flink.agents.api.memory.compaction.CompactionConfig;
import org.apache.flink.agents.api.resource.Resource;
import org.apache.flink.agents.api.resource.ResourceType;
import org.apache.flink.agents.api.vectorstores.Document;
import org.apache.flink.agents.api.vectorstores.VectorStoreQuery;
import org.apache.flink.agents.api.vectorstores.VectorStoreQueryMode;
import pemja.core.PythonInterpreter;
import pemja.core.object.PyObject;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Map;
import java.util.function.BiFunction;

/** Adapter for managing Java resources and facilitating Python-Java interoperability. */
public class JavaResourceAdapter {
    private final BiFunction<String, ResourceType, Resource> getResource;

    private final transient PythonInterpreter interpreter;

    public JavaResourceAdapter(
            BiFunction<String, ResourceType, Resource> getResource, PythonInterpreter interpreter) {
        this.getResource = getResource;
        this.interpreter = interpreter;
    }

    /**
     * Retrieves a Java resource by name and type value. This method is intended for use by the
     * Python interpreter.
     *
     * @param name the name of the resource to retrieve
     * @param typeValue the type value of the resource
     * @return the resource
     * @throws Exception if the resource cannot be retrieved
     */
    public Resource getResource(String name, String typeValue) throws Exception {
        return getResource.apply(name, ResourceType.fromValue(typeValue));
    }

    /**
     * Convert a Python chat message to a Java chat message. This method is intended for use by the
     * Python interpreter.
     *
     * @param pythonChatMessage the Python chat message
     * @return the Java chat message
     */
    public ChatMessage fromPythonChatMessage(Object pythonChatMessage) {
        // TODO: Delete this method after the pemja findClass method is fixed.
        ChatMessage chatMessage = new ChatMessage();
        if (interpreter == null) {
            throw new IllegalStateException("Python interpreter is not set.");
        }
        String roleValue =
                (String)
                        interpreter.invoke(
                                "python_java_utils.update_java_chat_message",
                                pythonChatMessage,
                                chatMessage);
        chatMessage.setRole(MessageRole.fromValue(roleValue));
        return chatMessage;
    }

    @SuppressWarnings("unchecked")
    public Document fromPythonDocument(PyObject pythonDocument) {
        // TODO: Delete this method after the pemja findClass method is fixed.
        return new Document(
                pythonDocument.getAttr("content").toString(),
                (Map<String, Object>) pythonDocument.getAttr("metadata", Map.class),
                pythonDocument.getAttr("id").toString());
    }

    @SuppressWarnings("unchecked")
    public VectorStoreQuery fromPythonVectorStoreQuery(PyObject pythonVectorStoreQuery) {
        // TODO: Delete this method after the pemja findClass method is fixed.
        String modeValue =
                (String)
                        interpreter.invoke(
                                "python_java_utils.get_mode_value", pythonVectorStoreQuery);
        return new VectorStoreQuery(
                VectorStoreQueryMode.fromValue(modeValue),
                (String) pythonVectorStoreQuery.getAttr("query_text"),
                pythonVectorStoreQuery.getAttr("limit", Integer.class),
                (String) pythonVectorStoreQuery.getAttr("collection_name"),
                (Map<String, Object>) pythonVectorStoreQuery.getAttr("extra_args", Map.class));
    }

    /**
     * Resolves a Python type name to a Java Class for LTM item types. Called from Python to create
     * Java Class<?> objects that can be passed to VectorStoreLongTermMemory.getOrCreateMemorySet().
     *
     * @param typeName "str" or "ChatMessage"
     * @return the corresponding Java Class
     */
    public Class<?> resolveMemoryItemType(String typeName) {
        if ("str".equals(typeName)) {
            return String.class;
        }
        if ("ChatMessage".equals(typeName)) {
            return ChatMessage.class;
        }
        throw new IllegalArgumentException("Unsupported LTM item type: " + typeName);
    }

    /**
     * Creates a Java CompactionConfig from primitive arguments. Called from Python since Pemja
     * cannot directly construct Java objects.
     *
     * @param model the chat model resource name for summarization
     * @param prompt optional prompt resource name (may be null)
     * @param limit max number of summarization topics
     * @return the Java CompactionConfig
     */
    public CompactionConfig createCompactionConfig(String model, Object prompt, int limit) {
        return new CompactionConfig(model, prompt, limit);
    }

    /**
     * Formats a Java LocalDateTime to an ISO-8601 string. Called from Python since Pemja cannot
     * directly call toString() on Java temporal objects.
     *
     * @param dateTime a LocalDateTime instance
     * @return ISO-8601 formatted string
     */
    public String formatDateTime(Object dateTime) {
        if (dateTime instanceof LocalDateTime) {
            return ((LocalDateTime) dateTime).format(DateTimeFormatter.ISO_DATE_TIME);
        }
        throw new IllegalArgumentException(
                "Expected LocalDateTime but got: " + dateTime.getClass().getName());
    }
}
