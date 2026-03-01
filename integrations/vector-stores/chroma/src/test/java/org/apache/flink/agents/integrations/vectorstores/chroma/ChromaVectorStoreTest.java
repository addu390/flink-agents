package org.apache.flink.agents.integrations.vectorstores.chroma;

import org.apache.flink.agents.api.resource.Resource;
import org.apache.flink.agents.api.resource.ResourceDescriptor;
import org.apache.flink.agents.api.resource.ResourceType;
import org.apache.flink.agents.api.vectorstores.CollectionManageableVectorStore;
import org.apache.flink.agents.api.vectorstores.Document;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.function.BiFunction;

import static org.junit.jupiter.api.Assertions.*;

class ChromaVectorStoreTest {

    private static ChromaVectorStore store;

    @BeforeAll
    static void setup() {
        // Skip if ChromaDB isn't running
        try {
            new tech.amikos.chromadb.Client("http://localhost:8000").heartbeat();
        } catch (Exception e) {
            Assumptions.assumeTrue(false, "ChromaDB not running at localhost:8000");
        }

        BiFunction<String, ResourceType, Resource> noopGetResource = (n, t) -> null;

        ResourceDescriptor desc =
                ResourceDescriptor.Builder.newBuilder(ChromaVectorStore.class.getName())
                        .addInitialArgument("host", "localhost")
                        .addInitialArgument("port", 8000)
                        .addInitialArgument("collection", "test_collection")
                        .build();

        store = new ChromaVectorStore(desc, noopGetResource);
    }

    @Test
    void testCollectionManagement() throws Exception {
        CollectionManageableVectorStore.Collection col =
                store.getOrCreateCollection("junit_test", Map.of("env", "test"));
        assertEquals("junit_test", col.getName());

        CollectionManageableVectorStore.Collection fetched = store.getCollection("junit_test");
        assertEquals("junit_test", fetched.getName());

        store.deleteCollection("junit_test");
    }

    @Test
    void testAddGetDelete() throws Exception {
        store.getOrCreateCollection("crud_test", Collections.emptyMap());

        List<Document> docs =
                List.of(
                        new Document(
                                "ChromaDB is a vector database", Map.of("src", "test"), "doc1"),
                        new Document(
                                "Flink Agents is an AI framework", Map.of("src", "test"), "doc2"));

        // Simulate pre-computed embeddings
        docs.get(0).setEmbedding(new float[] {0.1f, 0.2f, 0.3f, 0.4f, 0.5f});
        docs.get(1).setEmbedding(new float[] {0.5f, 0.4f, 0.3f, 0.2f, 0.1f});

        Map<String, Object> kwargs = new HashMap<>();
        kwargs.put("collection", "crud_test");
        kwargs.put("create_collection_if_not_exists", true);

        List<String> ids = store.addEmbedding(docs, "crud_test", kwargs);
        assertEquals(2, ids.size());

        // Get all
        List<Document> retrieved = store.get(null, "crud_test", Collections.emptyMap());
        assertEquals(2, retrieved.size());

        // Get by ID
        retrieved = store.get(List.of("doc1"), "crud_test", Collections.emptyMap());
        assertEquals(1, retrieved.size());
        assertEquals("doc1", retrieved.get(0).getId());

        // Delete one
        store.delete(List.of("doc1"), "crud_test", Collections.emptyMap());
        retrieved = store.get(null, "crud_test", Collections.emptyMap());
        assertEquals(1, retrieved.size());

        // Delete all
        store.delete(null, "crud_test", Collections.emptyMap());
        retrieved = store.get(null, "crud_test", Collections.emptyMap());
        assertEquals(0, retrieved.size());

        store.deleteCollection("crud_test");
    }

    @Test
    void testQueryEmbedding() throws Exception {
        store.getOrCreateCollection("query_test", Collections.emptyMap());

        List<Document> docs =
                List.of(
                        new Document("cats are fluffy", Map.of("topic", "animals"), "d1"),
                        new Document("java is a language", Map.of("topic", "code"), "d2"));
        docs.get(0).setEmbedding(new float[] {1.0f, 0.0f, 0.0f});
        docs.get(1).setEmbedding(new float[] {0.0f, 1.0f, 0.0f});

        Map<String, Object> kwargs = new HashMap<>();
        kwargs.put("collection", "query_test");
        kwargs.put("create_collection_if_not_exists", true);
        store.addEmbedding(docs, "query_test", kwargs);

        // Query with an embedding close to d1
        List<Document> results =
                store.queryEmbedding(new float[] {0.9f, 0.1f, 0.0f}, 1, "query_test", kwargs);
        assertEquals(1, results.size());
        assertEquals("d1", results.get(0).getId());

        store.deleteCollection("query_test");
    }
}
