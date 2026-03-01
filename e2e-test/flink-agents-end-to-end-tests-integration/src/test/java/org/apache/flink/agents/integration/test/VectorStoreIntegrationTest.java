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

package org.apache.flink.agents.integration.test;

import org.apache.flink.agents.api.AgentsExecutionEnvironment;
import org.apache.flink.agents.api.resource.ResourceDescriptor;
import org.apache.flink.agents.api.resource.ResourceName;
import org.apache.flink.agents.api.vectorstores.Document;
import org.apache.flink.agents.integrations.vectorstores.chroma.ChromaVectorStore;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.DataStreamSource;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.CloseableIterator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Parameterized integration test for Vector Stores. Validates Elasticsearch and ChromaDB backends.
 *
 * <p>Environment variables required for <b>Elasticsearch</b>:
 *
 * <ul>
 *   <li>ES_HOST (e.g., http://localhost:9200)
 *   <li>ES_INDEX (e.g., my_documents)
 *   <li>ES_VECTOR_FIELD (e.g., content_vector)
 *   <li>ES_DIMS (optional; defaults to 768)
 * </ul>
 *
 * <p>Environment variables required for <b>ChromaDB</b>:
 *
 * <ul>
 *   <li>CHROMA_HOST (e.g., localhost) — defaults to localhost
 *   <li>CHROMA_PORT (e.g., 8000) — defaults to 8000
 *   <li>CHROMA_COLLECTION (e.g., my_collection) — defaults to flink_agents_vs_test
 * </ul>
 *
 * If the required environment variables are not provided, the test for that backend will be
 * skipped.
 */
public class VectorStoreIntegrationTest {

    @ParameterizedTest
    @ValueSource(strings = {"ELASTICSEARCH", "CHROMA"})
    public void testVectorStoreSemanticQuery(String backend) throws Exception {
        if ("ELASTICSEARCH".equals(backend)) {
            String host = getEnvOrProperty("ES_HOST");
            String index = getEnvOrProperty("ES_INDEX");
            String vectorField = getEnvOrProperty("ES_VECTOR_FIELD");

            Assumptions.assumeTrue(host != null && !host.isEmpty(), "ES_HOST is not set");
            Assumptions.assumeTrue(index != null && !index.isEmpty(), "ES_INDEX is not set");
            Assumptions.assumeTrue(
                    vectorField != null && !vectorField.isEmpty(), "ES_VECTOR_FIELD is not set");
        } else if ("CHROMA".equals(backend)) {
            String host = getEnvOrProperty("CHROMA_HOST");
            Assumptions.assumeTrue(host != null && !host.isEmpty(), "CHROMA_HOST is not set");
        } else {
            Assumptions.abort("Unknown backend: " + backend);
        }

        System.setProperty("VECTOR_STORE_PROVIDER", backend);

        // For ChromaDB, seed the collection with test documents before querying.
        // Elasticsearch tests expect a pre-populated index; ChromaDB needs explicit seeding.
        String chromaCollection = null;
        ChromaVectorStore seedStore = null;
        if ("CHROMA".equals(backend)) {
            String chromaHost = getEnvOrProperty("CHROMA_HOST");
            String portStr = getEnvOrProperty("CHROMA_PORT");
            int chromaPort =
                    (portStr != null && !portStr.isEmpty()) ? Integer.parseInt(portStr) : 8000;
            chromaCollection =
                    getEnvOrProperty("CHROMA_COLLECTION") != null
                            ? getEnvOrProperty("CHROMA_COLLECTION")
                            : "flink_agents_vs_test";

            seedStore =
                    new ChromaVectorStore(
                            ResourceDescriptor.Builder.newBuilder(
                                            ResourceName.VectorStore.CHROMA_VECTOR_STORE)
                                    .addInitialArgument("host", chromaHost)
                                    .addInitialArgument("port", chromaPort)
                                    .addInitialArgument("collection", chromaCollection)
                                    .build(),
                            (n, t) -> null);

            seedStore.getOrCreateCollection(chromaCollection, Collections.emptyMap());
            seedChromaCollection(seedStore, chromaCollection);
        }

        try {
            StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
            env.setParallelism(1);

            final DataStreamSource<String> inputStream = env.fromData("What is Apache Flink");

            final AgentsExecutionEnvironment agentEnv =
                    AgentsExecutionEnvironment.getExecutionEnvironment(env);

            final DataStream<Object> outputStream =
                    agentEnv.fromDataStream(
                                    inputStream, (KeySelector<String, String>) value -> value)
                            .apply(new VectorStoreIntegrationAgent())
                            .toDataStream();

            final CloseableIterator<Object> results = outputStream.collectAsync();

            agentEnv.execute();

            checkResult(results);
        } finally {
            // Clean up ChromaDB collection after the test
            if (seedStore != null && chromaCollection != null) {
                try {
                    seedStore.deleteCollection(chromaCollection);
                } catch (Exception ignored) {
                }
            }
        }
    }

    /**
     * Seeds a ChromaDB collection with test documents containing random 768-dim embeddings. The
     * exact embedding values don't matter — ChromaDB nearest-neighbor search returns the closest
     * matches regardless of distance, so the query will find these documents.
     */
    private void seedChromaCollection(ChromaVectorStore store, String collection) throws Exception {
        Random rng = new Random(42);
        List<Document> docs =
                List.of(
                        new Document(
                                "Apache Flink is a distributed stream processing framework",
                                Map.of("topic", "flink"),
                                "seed-1"),
                        new Document(
                                "Flink supports event-driven applications and real-time analytics",
                                Map.of("topic", "flink"),
                                "seed-2"),
                        new Document(
                                "Vector stores enable semantic search over document embeddings",
                                Map.of("topic", "vector-stores"),
                                "seed-3"));

        for (Document doc : docs) {
            float[] emb = new float[768];
            for (int i = 0; i < emb.length; i++) {
                emb[i] = rng.nextFloat();
            }
            doc.setEmbedding(emb);
        }

        // Documents already have embeddings set, so add() won't invoke the embedding model.
        store.add(docs, collection, Collections.emptyMap());
    }

    @SuppressWarnings("unchecked")
    private void checkResult(CloseableIterator<Object> results) {
        Assertions.assertTrue(
                results.hasNext(), "No output received from VectorStoreIntegrationAgent");

        Object obj = results.next();
        Assertions.assertInstanceOf(Map.class, obj, "Output must be a Map");

        java.util.Map<String, Object> res = (java.util.Map<String, Object>) obj;
        Assertions.assertEquals("PASSED", res.get("test_status"));

        Object count = res.get("retrieved_count");
        Assertions.assertNotNull(count, "retrieved_count must exist");
        if (count instanceof Number) {
            Assertions.assertTrue(((Number) count).intValue() >= 1, "retrieved_count must be >= 1");
        }

        Object preview = res.get("first_doc_preview");
        Assertions.assertTrue(
                preview instanceof String && !((String) preview).trim().isEmpty(),
                "first_doc_preview must be a non-empty string");

        Object firstId = res.get("first_doc_id");
        if (firstId != null) {
            Assertions.assertTrue(
                    firstId instanceof String && !((String) firstId).trim().isEmpty(),
                    "first_doc_id when present must be a non-empty string");
        }
    }

    private static String getEnvOrProperty(String key) {
        String val = System.getenv(key);
        if (val == null || val.isEmpty()) {
            val = System.getProperty(key);
        }
        return val;
    }
}
