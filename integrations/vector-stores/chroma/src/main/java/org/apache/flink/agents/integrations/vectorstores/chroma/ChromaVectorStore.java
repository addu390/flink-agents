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

package org.apache.flink.agents.integrations.vectorstores.chroma;

import com.google.gson.Gson;
import org.apache.flink.agents.api.resource.Resource;
import org.apache.flink.agents.api.resource.ResourceDescriptor;
import org.apache.flink.agents.api.resource.ResourceType;
import org.apache.flink.agents.api.vectorstores.BaseVectorStore;
import org.apache.flink.agents.api.vectorstores.CollectionManageableVectorStore;
import org.apache.flink.agents.api.vectorstores.Document;
import tech.amikos.chromadb.Client;
import tech.amikos.chromadb.handler.ApiClient;
import tech.amikos.chromadb.handler.ApiException;
import tech.amikos.chromadb.handler.DefaultApi;
import tech.amikos.chromadb.model.AddEmbedding;
import tech.amikos.chromadb.model.DeleteEmbedding;
import tech.amikos.chromadb.model.GetEmbedding;
import tech.amikos.chromadb.model.QueryEmbedding;

import javax.annotation.Nullable;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.function.BiFunction;

/**
 * ChromaDB-backed implementation of a vector store.
 *
 * <p>Connects to a running ChromaDB server via HTTP using the community Java client ({@code
 * io.github.amikos-tech:chromadb-java-client}). Supports collection management, document CRUD, and
 * approximate nearest neighbor vector queries.
 *
 * <p>For data operations ({@code get}, {@code delete}, {@code add}, {@code query}) the
 * implementation works directly with the low-level {@link DefaultApi} and request model objects
 * ({@link GetEmbedding}, {@link DeleteEmbedding}, {@link AddEmbedding}, {@link QueryEmbedding}).
 * This avoids the {@link tech.amikos.chromadb.Collection} high-level wrapper which couples an
 * {@code EmbeddingFunction} into every query call — our framework already computes embeddings
 * externally via {@link org.apache.flink.agents.api.embedding.model.BaseEmbeddingModelSetup}.
 *
 * <p>Configuration arguments (via {@link ResourceDescriptor}):
 *
 * <ul>
 *   <li>{@code host} (optional): ChromaDB server host. Defaults to {@code localhost}.
 *   <li>{@code port} (optional): ChromaDB server port. Defaults to {@code 8000}.
 *   <li>{@code collection} (optional): Default collection name. Defaults to {@link
 *       #DEFAULT_COLLECTION}.
 *   <li>{@code collection_metadata} (optional): Default metadata for new collections.
 *   <li>{@code create_collection_if_not_exists} (optional): Auto-create on add/query. Defaults to
 *       {@code true}.
 * </ul>
 *
 * <p>Example:
 *
 * <pre>{@code
 * ResourceDescriptor desc = ResourceDescriptor.Builder
 *     .newBuilder(ResourceName.VectorStore.CHROMA_VECTOR_STORE)
 *     .addInitialArgument("embedding_model", "textEmbedder")
 *     .addInitialArgument("host", "localhost")
 *     .addInitialArgument("port", 8000)
 *     .addInitialArgument("collection", "my_docs")
 *     .build();
 * }</pre>
 */
public class ChromaVectorStore extends BaseVectorStore implements CollectionManageableVectorStore {

    public static final String DEFAULT_COLLECTION = "flink_agents_chroma_collection";

    /**
     * Max documents per add batch. ChromaDB enforces a server-side limit; mirrors the Python
     * client's constant.
     */
    public static final int MAX_CHUNK_SIZE = 41665;

    private static final Gson GSON = new Gson();

    private final String host;
    private final int port;
    private final String collectionName;
    private final Map<String, String> collectionMetadata;
    private final boolean createCollectionIfNotExists;

    /** High-level client used for collection management (create, get, delete, list). */
    private Client client;

    /** Low-level API used for data operations (get, add, delete, query by embedding). */
    private DefaultApi api;

    public ChromaVectorStore(
            ResourceDescriptor descriptor, BiFunction<String, ResourceType, Resource> getResource) {
        super(descriptor, getResource);

        this.host = Objects.requireNonNullElse(descriptor.getArgument("host"), "localhost");
        Integer portArg = descriptor.getArgument("port");
        this.port = (portArg != null) ? portArg : 8000;
        this.collectionName =
                Objects.requireNonNullElse(
                        descriptor.getArgument("collection"), DEFAULT_COLLECTION);

        Map<String, String> metaArg = descriptor.getArgument("collection_metadata");
        this.collectionMetadata = (metaArg != null) ? metaArg : Collections.emptyMap();

        Boolean createArg = descriptor.getArgument("create_collection_if_not_exists");
        this.createCollectionIfNotExists = (createArg != null) ? createArg : true;
    }

    private String getBasePath() {
        return String.format("http://%s:%d", this.host, this.port);
    }

    private Client getClient() {
        if (this.client == null) {
            this.client = new Client(getBasePath());
        }
        return this.client;
    }

    /** Returns the low-level API for direct data operations. */
    private DefaultApi getApi() {
        if (this.api == null) {
            ApiClient apiClient = new ApiClient();
            apiClient.setBasePath(getBasePath());
            this.api = new DefaultApi(apiClient);
        }
        return this.api;
    }

    @Override
    public void close() throws Exception {
        // ChromaDB HTTP client does not require explicit close.
    }

    @Override
    public Collection getOrCreateCollection(String name, Map<String, Object> metadata)
            throws Exception {
        try {
            Map<String, String> chromaMeta = ensureEmbeddingFunctionKey(toStringMap(metadata));
            tech.amikos.chromadb.Collection col =
                    getClient().createCollection(name, chromaMeta, true, null);
            Map<String, Object> returnedMeta = col.getMetadata();
            return new Collection(
                    name,
                    returnedMeta != null
                            ? returnedMeta
                            : (metadata != null ? metadata : Collections.emptyMap()));
        } catch (ApiException e) {
            throw new RuntimeException("Failed to get or create ChromaDB collection: " + name, e);
        }
    }

    @Override
    public Collection getCollection(String name) throws Exception {
        try {
            tech.amikos.chromadb.Collection col = getClient().getCollection(name, null);
            Map<String, Object> meta = col.getMetadata();
            return new Collection(name, meta != null ? meta : Collections.emptyMap());
        } catch (ApiException e) {
            throw new RuntimeException(String.format("ChromaDB collection %s not found", name), e);
        }
    }

    @Override
    public Collection deleteCollection(String name) throws Exception {
        Collection collection = getCollection(name);
        try {
            getClient().deleteCollection(name);
        } catch (ApiException e) {
            throw new RuntimeException("Failed to delete ChromaDB collection: " + name, e);
        }
        return collection;
    }

    @Override
    public Map<String, Object> getStoreKwargs() {
        Map<String, Object> m = new HashMap<>();
        m.put("collection", this.collectionName);
        m.put("collection_metadata", this.collectionMetadata);
        m.put("create_collection_if_not_exists", this.createCollectionIfNotExists);
        return m;
    }

    @Override
    public long size(@Nullable String collection) throws Exception {
        String colName = resolveCollection(collection);
        try {
            tech.amikos.chromadb.Collection chromaCol = getClient().getCollection(colName, null);
            return chromaCol.count();
        } catch (ApiException e) {
            throw new RuntimeException("Failed to count documents in collection: " + colName, e);
        }
    }

    /**
     * Retrieves documents from ChromaDB.
     *
     * <p>Uses the low-level {@link DefaultApi} so that {@code limit} and {@code offset} from {@code
     * extraArgs} are properly forwarded (the high-level {@code Collection.get()} method does not
     * expose these parameters).
     */
    @Override
    public List<Document> get(
            @Nullable List<String> ids, @Nullable String collection, Map<String, Object> extraArgs)
            throws IOException {
        String colName = resolveCollection(collection);
        try {
            String collectionId = getCollectionId(colName);

            GetEmbedding req = new GetEmbedding();
            if (ids != null) {
                req.ids(ids);
            }
            Object where = extraArgs.get("where");
            if (where != null) {
                req.where(where);
            }
            Object whereDoc = extraArgs.get("where_document");
            if (whereDoc != null) {
                req.whereDocument(whereDoc);
            }
            Integer limit = (Integer) extraArgs.get("limit");
            if (limit != null) {
                req.limit(limit);
            }
            Integer offset = (Integer) extraArgs.get("offset");
            if (offset != null) {
                req.offset(offset);
            }

            Object rawResponse = getApi().get(req, collectionId);
            tech.amikos.chromadb.Collection.GetResult result =
                    deserialize(rawResponse, tech.amikos.chromadb.Collection.GetResult.class);

            return convertGetResult(result);
        } catch (ApiException e) {
            throw new IOException("Failed to get documents from ChromaDB", e);
        }
    }

    @Override
    public void delete(
            @Nullable List<String> ids, @Nullable String collection, Map<String, Object> extraArgs)
            throws IOException {
        String colName = resolveCollection(collection);
        try {
            String collectionId = getCollectionId(colName);

            Map<String, Object> where = castToObjectMap(extraArgs.get("where"));
            Map<String, Object> whereDoc = castToObjectMap(extraArgs.get("where_document"));

            if (ids == null && where == null && whereDoc == null) {
                // Delete all: fetch IDs first (mirrors Python behavior).
                GetEmbedding getAll = new GetEmbedding();
                Object rawAll = getApi().get(getAll, collectionId);
                tech.amikos.chromadb.Collection.GetResult all =
                        deserialize(rawAll, tech.amikos.chromadb.Collection.GetResult.class);
                ids = all.getIds();
                if (ids == null || ids.isEmpty()) {
                    return;
                }
            }

            DeleteEmbedding req = new DeleteEmbedding();
            if (ids != null) {
                req.ids(ids);
            }
            if (where != null) {
                req.where(where);
            }
            if (whereDoc != null) {
                req.whereDocument(whereDoc);
            }

            getApi().delete(req, collectionId);
        } catch (ApiException e) {
            throw new IOException("Failed to delete documents from ChromaDB", e);
        }
    }

    /**
     * Performs a vector search using a pre-computed embedding.
     *
     * <p>Builds a {@link QueryEmbedding} request with the raw embedding vector and calls {@link
     * DefaultApi#getNearestNeighbors} directly so that the embedding function coupling in {@code
     * Collection.query()} is bypassed.
     */
    @Override
    protected List<Document> queryEmbedding(
            float[] embedding, int limit, @Nullable String collection, Map<String, Object> args) {
        try {
            String collectionId = resolveOrCreateCollectionId(collection, args);

            // Convert float[] → List<Float> for JSON serialization
            List<Float> embeddingList = new ArrayList<>(embedding.length);
            for (float v : embedding) {
                embeddingList.add(v);
            }

            QueryEmbedding body = new QueryEmbedding();
            body.queryEmbeddings(Collections.singletonList(embeddingList));
            body.nResults(limit);
            body.include(
                    Arrays.asList(
                            QueryEmbedding.IncludeEnum.DOCUMENTS,
                            QueryEmbedding.IncludeEnum.METADATAS));

            Map<String, Object> where = castToObjectMap(args.get("where"));
            if (where != null) {
                body.where(where);
            }

            Object rawResponse = getApi().getNearestNeighbors(body, collectionId);
            tech.amikos.chromadb.Collection.QueryResponse response =
                    deserialize(rawResponse, tech.amikos.chromadb.Collection.QueryResponse.class);

            return convertQueryResponse(response);
        } catch (ApiException e) {
            throw new RuntimeException("Error performing vector search in ChromaDB", e);
        }
    }

    /**
     * Adds documents with pre-computed embeddings to ChromaDB.
     *
     * <p>Builds an {@link AddEmbedding} request with raw embedding arrays and calls {@link
     * DefaultApi#add} directly. This avoids the {@code Collection.add()} method which requires
     * {@link tech.amikos.chromadb.Embedding} objects.
     */
    @Override
    @SuppressWarnings("unchecked")
    protected List<String> addEmbedding(
            List<Document> documents, @Nullable String collection, Map<String, Object> extraArgs)
            throws IOException {
        try {
            String collectionId = resolveOrCreateCollectionId(collection, extraArgs);

            List<String> allIds = new ArrayList<>();

            // Chunk to respect ChromaDB's batch size limit
            for (int start = 0; start < documents.size(); start += MAX_CHUNK_SIZE) {
                int end = Math.min(start + MAX_CHUNK_SIZE, documents.size());
                List<Document> chunk = documents.subList(start, end);

                List<String> ids = new ArrayList<>(chunk.size());
                List<String> docs = new ArrayList<>(chunk.size());
                List<Object> embeddings = new ArrayList<>(chunk.size());
                List<Map<String, Object>> metadatas = new ArrayList<>(chunk.size());

                for (Document doc : chunk) {
                    String id =
                            (doc.getId() != null && !doc.getId().isEmpty())
                                    ? doc.getId()
                                    : UUID.randomUUID().toString();
                    ids.add(id);
                    docs.add(doc.getContent());

                    float[] emb = doc.getEmbedding();
                    if (emb != null) {
                        List<Float> embList = new ArrayList<>(emb.length);
                        for (float v : emb) {
                            embList.add(v);
                        }
                        embeddings.add(embList);
                    } else {
                        embeddings.add(Collections.emptyList());
                    }

                    metadatas.add(
                            doc.getMetadata() != null ? doc.getMetadata() : Collections.emptyMap());
                }

                AddEmbedding req = new AddEmbedding();
                req.setEmbeddings(embeddings);
                req.setMetadatas(metadatas);
                req.setDocuments(docs);
                req.setIds(ids);
                req.incrementIndex(true);

                getApi().add(req, collectionId);
                allIds.addAll(ids);
            }

            return allIds;
        } catch (ApiException e) {
            throw new IOException("Failed to add documents to ChromaDB", e);
        }
    }

    private String resolveCollection(@Nullable String collection) {
        return (collection != null) ? collection : this.collectionName;
    }

    /**
     * Fetches the ChromaDB-internal collection ID for a collection name. Required by {@link
     * DefaultApi} methods which identify collections by ID, not name.
     */
    private String getCollectionId(String collectionName) throws ApiException {
        tech.amikos.chromadb.Collection col = getClient().getCollection(collectionName, null);
        return col.getId();
    }

    /**
     * Gets or creates a collection (based on the {@code create_collection_if_not_exists} flag) and
     * returns its ChromaDB-internal ID. Used by {@code addEmbedding} and {@code queryEmbedding} to
     * mirror the Python behaviour.
     */
    private String resolveOrCreateCollectionId(
            @Nullable String collection, Map<String, Object> args) throws ApiException {
        String colName = (String) args.getOrDefault("collection", resolveCollection(collection));
        boolean create =
                (boolean)
                        args.getOrDefault(
                                "create_collection_if_not_exists",
                                this.createCollectionIfNotExists);

        tech.amikos.chromadb.Collection col;
        if (create) {
            Map<String, String> meta =
                    ensureEmbeddingFunctionKey(
                            castToStringMap(
                                    args.getOrDefault(
                                            "collection_metadata", this.collectionMetadata)));
            col = getClient().createCollection(colName, meta, true, null);
        } else {
            col = getClient().getCollection(colName, null);
        }
        return col.getId();
    }

    /**
     * Deserializes a raw API response (returned as {@code Object} by {@link DefaultApi}) into the
     * target type via Gson round-trip. This mirrors the pattern used internally by {@link
     * tech.amikos.chromadb.Collection}.
     */
    private static <T> T deserialize(Object rawResponse, Class<T> clazz) {
        String json = GSON.toJson(rawResponse);
        return GSON.fromJson(json, clazz);
    }

    /**
     * Converts a {@code GetResult} (single-level lists) into framework {@link Document} objects.
     */
    @SuppressWarnings("unchecked")
    private static List<Document> convertGetResult(
            tech.amikos.chromadb.Collection.GetResult result) {
        List<String> ids = result.getIds();
        if (ids == null || ids.isEmpty()) {
            return Collections.emptyList();
        }

        List<String> docs = result.getDocuments();
        List<Map<String, Object>> metas =
                (List<Map<String, Object>>) (List<?>) result.getMetadatas();

        List<Document> documents = new ArrayList<>(ids.size());
        for (int i = 0; i < ids.size(); i++) {
            String content = (docs != null && i < docs.size()) ? docs.get(i) : "";
            Map<String, Object> meta =
                    (metas != null && i < metas.size() && metas.get(i) != null)
                            ? metas.get(i)
                            : Collections.emptyMap();
            documents.add(new Document(content, meta, ids.get(i)));
        }
        return documents;
    }

    /**
     * Converts a {@code QueryResponse} (nested lists — one per query embedding) into framework
     * {@link Document} objects. We always send a single query embedding, so we read index 0.
     */
    @SuppressWarnings("unchecked")
    private static List<Document> convertQueryResponse(
            tech.amikos.chromadb.Collection.QueryResponse response) {
        if (response.getIds() == null || response.getIds().isEmpty()) {
            return Collections.emptyList();
        }

        List<String> ids = response.getIds().get(0);
        List<String> docs =
                (response.getDocuments() != null && !response.getDocuments().isEmpty())
                        ? response.getDocuments().get(0)
                        : Collections.emptyList();
        List<Map<String, Object>> metas =
                (response.getMetadatas() != null && !response.getMetadatas().isEmpty())
                        ? (List<Map<String, Object>>) (List<?>) response.getMetadatas().get(0)
                        : Collections.emptyList();

        List<Document> documents = new ArrayList<>(ids.size());
        for (int i = 0; i < ids.size(); i++) {
            String content = (i < docs.size()) ? docs.get(i) : "";
            Map<String, Object> meta =
                    (i < metas.size() && metas.get(i) != null)
                            ? metas.get(i)
                            : Collections.emptyMap();
            documents.add(new Document(content, meta, ids.get(i)));
        }
        return documents;
    }

    /**
     * The chromadb-java-client {@code Client.createCollection} internally calls {@code
     * ef.getClass().getName()} when the metadata map is null, empty, or lacks the {@code
     * "embedding_function"} key. Since we manage embeddings externally and pass {@code null} for
     * the EmbeddingFunction parameter, this causes a NullPointerException. Adding a placeholder
     * value for the key makes the Client skip that code path.
     */
    private static Map<String, String> ensureEmbeddingFunctionKey(
            @Nullable Map<String, String> meta) {
        Map<String, String> mutable = (meta != null) ? new HashMap<>(meta) : new HashMap<>();
        mutable.putIfAbsent("embedding_function", "none");
        return mutable;
    }

    /** Converts a {@code Map<String, Object>} to {@code Map<String, String>} for ChromaDB. */
    private static Map<String, String> toStringMap(@Nullable Map<String, Object> map) {
        if (map == null || map.isEmpty()) {
            return Collections.emptyMap();
        }
        Map<String, String> result = new HashMap<>();
        for (Map.Entry<String, Object> entry : map.entrySet()) {
            if (entry.getValue() != null) {
                result.put(entry.getKey(), String.valueOf(entry.getValue()));
            }
        }
        return result;
    }

    /** Best-effort cast of an arbitrary object to {@code Map<String, String>}. */
    @SuppressWarnings("unchecked")
    @Nullable
    private static Map<String, String> castToStringMap(@Nullable Object obj) {
        if (obj == null) {
            return null;
        }
        if (obj instanceof Map) {
            Map<String, String> result = new HashMap<>();
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) obj).entrySet()) {
                result.put(String.valueOf(entry.getKey()), String.valueOf(entry.getValue()));
            }
            return result;
        }
        return null;
    }

    /** Best-effort cast of an arbitrary object to {@code Map<String, Object>}. */
    @SuppressWarnings("unchecked")
    @Nullable
    private static Map<String, Object> castToObjectMap(@Nullable Object obj) {
        if (obj == null) {
            return null;
        }
        if (obj instanceof Map) {
            return (Map<String, Object>) obj;
        }
        return null;
    }
}
