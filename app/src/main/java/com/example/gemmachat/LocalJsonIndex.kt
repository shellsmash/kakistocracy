package com.example.gemmachat

import android.content.res.AssetManager
import com.google.ai.edge.litertlm.Tool
import com.google.ai.edge.litertlm.ToolParam
import com.google.ai.edge.litertlm.ToolSet
import java.io.IOException
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import org.json.JSONTokener

internal data class SearchResult(
    val source: String,
    val text: String,
)

internal class LocalJsonIndex private constructor(
    private val chunks: List<IndexedChunk>,
    val sourceCount: Int,
) {
    val chunkCount: Int
        get() = chunks.size

    fun search(query: String, limit: Int = 4): List<SearchResult> {
        val queryTerms = tokenize(query).toSet()
        if (queryTerms.isEmpty() || chunks.isEmpty()) return emptyList()

        val averageLength = chunks.sumOf { it.terms.size }.toDouble() / chunks.size
        val scores = chunks.mapIndexedNotNull { index, chunk ->
            var score = 0.0
            queryTerms.forEach { term ->
                val termFrequency = chunk.termFrequencies[term] ?: return@forEach
                val documentFrequency = documentFrequencies[term] ?: return@forEach
                val inverseDocumentFrequency =
                    ln(1.0 + (chunks.size - documentFrequency + 0.5) / (documentFrequency + 0.5))
                val lengthNormalization = 1.5 *
                    (1.0 - 0.75 + 0.75 * chunk.terms.size / averageLength)
                score += inverseDocumentFrequency *
                    (termFrequency * (1.5 + 1.0)) / (termFrequency + lengthNormalization)
            }
            if (score > 0.0) index to score else null
        }

        return scores
            .sortedByDescending { it.second }
            .take(limit)
            .map { (index, _) ->
                val chunk = chunks[index]
                SearchResult(chunk.source, chunk.text)
            }
    }

    private val documentFrequencies: Map<String, Int> by lazy {
        val counts = mutableMapOf<String, Int>()
        chunks.forEach { chunk ->
            chunk.termFrequencies.keys.forEach { term ->
                counts[term] = (counts[term] ?: 0) + 1
            }
        }
        counts
    }

    private data class IndexedChunk(
        val source: String,
        val text: String,
        val terms: List<String>,
        val termFrequencies: Map<String, Int>,
    )

    companion object {
        private const val ASSET_DIRECTORY = "rag"
        private const val CHUNK_SIZE = 1000
        private const val CHUNK_OVERLAP = 120
        private val tokenPattern = Regex("[\\p{L}\\p{N}]{2,}")

        fun build(assetManager: AssetManager): LocalJsonIndex {
            val jsonPaths = collectJsonPaths(assetManager, ASSET_DIRECTORY)
            val indexedChunks = mutableListOf<IndexedChunk>()
            var indexedSources = 0

            jsonPaths.forEach { assetPath ->
                val jsonValue = try {
                    assetManager.open(assetPath).bufferedReader(Charsets.UTF_8).use { reader ->
                        JSONTokener(reader.readText()).nextValue()
                    }
                } catch (exception: JSONException) {
                    throw IOException("Invalid JSON in app/src/main/assets/$assetPath", exception)
                }

                val records = if (jsonValue is JSONArray) {
                    (0 until jsonValue.length()).map(jsonValue::get)
                } else {
                    listOf(jsonValue)
                }
                val sourceChunks = records.flatMap { record ->
                    val fields = mutableListOf<String>()
                    appendFields(record, "", fields)
                    splitIntoChunks(fields.joinToString("\n"))
                }
                if (sourceChunks.isNotEmpty()) indexedSources++

                sourceChunks.forEach { text ->
                    val terms = tokenize(text)
                    if (terms.isNotEmpty()) {
                        indexedChunks += IndexedChunk(
                            source = assetPath.removePrefix("$ASSET_DIRECTORY/"),
                            text = text,
                            terms = terms,
                            termFrequencies = terms.groupingBy { it }.eachCount(),
                        )
                    }
                }
            }

            return LocalJsonIndex(indexedChunks, indexedSources)
        }

        private fun collectJsonPaths(assetManager: AssetManager, directory: String): List<String> {
            val children = assetManager.list(directory).orEmpty()
            return children.flatMap { child ->
                val path = "$directory/$child"
                val nested = assetManager.list(path)
                when {
                    nested != null && nested.isNotEmpty() -> collectJsonPaths(assetManager, path)
                    child.endsWith(".json", ignoreCase = true) -> listOf(path)
                    else -> emptyList()
                }
            }
        }

        private fun appendFields(value: Any?, path: String, fields: MutableList<String>) {
            when (value) {
                is JSONObject -> {
                    val keys = value.keys().asSequence().toList().sorted()
                    keys.forEach { key ->
                        val childPath = if (path.isEmpty()) key else "$path.$key"
                        appendFields(value.get(key), childPath, fields)
                    }
                }
                is JSONArray -> {
                    for (index in 0 until value.length()) {
                        appendFields(value.get(index), "$path[$index]", fields)
                    }
                }
                JSONObject.NULL -> Unit
                null -> Unit
                else -> {
                    val text = value.toString().trim()
                    if (text.isNotEmpty()) {
                        fields += if (path.isEmpty()) text else "$path: $text"
                    }
                }
            }
        }

        private fun splitIntoChunks(text: String): List<String> {
            if (text.isBlank()) return emptyList()
            val result = mutableListOf<String>()
            var start = 0
            while (start < text.length) {
                var end = min(start + CHUNK_SIZE, text.length)
                if (end < text.length) {
                    val boundary = text.lastIndexOf(' ', end)
                    if (boundary > start + CHUNK_SIZE / 2) end = boundary
                }
                text.substring(start, end).trim().takeIf(String::isNotEmpty)?.let(result::add)
                if (end == text.length) break
                start = max(start + 1, end - CHUNK_OVERLAP)
            }
            return result
        }

        private fun tokenize(text: String): List<String> =
            tokenPattern.findAll(text.lowercase()).map { it.value }.toList()
    }
}

internal class LocalDataSearchTool(private val index: LocalJsonIndex) : ToolSet {

    val hasIndexed: Boolean
        get() = true

    val indexedSourceCount: Int
        get() = index.sourceCount

    @Tool(
        description = "Search the app's bundled local JSON knowledge base for facts needed to " +
            "answer the user's question. Call this only when the question asks about specific " +
            "information that may be in the app's local data. The query should be a short " +
            "description of the information to find. Never use this tool for general knowledge.",
    )
    fun searchLocalData(
        @ToolParam(description = "The specific fact or topic to find in the local JSON data.")
        query: String,
    ): String {
        val matches = index.search(query)
        if (matches.isEmpty()) {
            return "No relevant records were found in the local JSON data. Do not invent an answer from the data."
        }

        return matches.joinToString("\n\n") { result ->
            "Source: ${result.source}\n${result.text}"
        }
    }
}
