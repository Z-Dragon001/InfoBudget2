# InfoBudget LLM Memory Extraction Design

## 1. Why the current code keeps a heuristic fallback

`infobudget/extractors/mock_joint.py` is a mock extractor, not a real LLM extractor.

Its purpose in the current codebase is now to:

- keep the full pipeline runnable end to end when local/API models are unavailable
- let segmentation, scoring, routing, storage, and evaluation be tested without GPU or API dependencies
- provide fallback `MemoryEntry` objects with the same schema produced by the LLM extractor

That fallback path still uses heuristic functions such as:

- `extract_entities(...)`
- `extract_preferences(...)`
- `extract_constraints(...)`
- `summarize_text(...)`

The primary runtime path is now `TieredJointExtractor`, which delegates to `LocalJointExtractor` for local model tiers and `APIJointExtractor` for API tiers.

## 2. Target extraction architecture

InfoBudget should use **joint long-term memory extraction by LLMs**, not heuristic entity extraction, for the real system.

Recommended design:

1. `small` tier
   - local deployment
   - optimized for precision and low cost
   - extracts only the highest-value memories
   - example role: routine segments with lower information score

2. `medium` tier
   - local deployment
   - balanced recall and cost
   - default extractor for most meaningful segments

3. `large` tier
   - API model
   - highest recall and best temporal/event synthesis
   - used only for the most valuable segments

This matches the InfoBudget idea:

- segment first
- compute information score
- route by budget/value
- use different model scales for memory extraction

## 3. Why joint extraction is the right design

InfoBudget should not separately run:

- one extractor for entities
- another extractor for facts
- another extractor for episodes
- another extractor for summary

Instead, one LLM call should jointly produce:

- retrieval-oriented summary
- semantic memory
- episodic memory
- importance score

Benefits:

- lower total token cost
- fewer consistency conflicts across extracted fields
- better alignment between summary, semantic facts, and episodic events
- easier storage and evaluation

## 4. Memory schema to keep fixed

The prompt should target the existing project schema already defined in `infobudget/schemas.py`:

- `MemoryEntry`
- `SemanticMemory`
- `SemanticEntity`
- `SemanticFact`
- `Preference`
- `Constraint`
- `EpisodicMemory`
- `Episode`

The LLM should therefore output a JSON object that can be parsed into:

```json
{
  "topic": "...",
  "summary": "...",
  "semantic_memory": {
    "entities": [],
    "facts": [],
    "preferences": [],
    "constraints": []
  },
  "episodic_memory": {
    "episodes": []
  },
  "importance": 0.0
}
```

## 5. Prompt design principles

All three tiers should use prompts that are:

- in English
- JSON-only
- schema-constrained
- grounded in the segment text only
- explicit about semantic memory vs episodic memory
- explicit about timestamp handling
- explicit about image-description handling

The prompts added to `configs/prompts/` are:

- `joint_memory_extraction_small.txt`
- `joint_memory_extraction_medium.txt`
- `joint_memory_extraction_large.txt`
- `joint_memory_extraction.txt` as a generic fallback

## 6. Tier-specific prompt behavior

### Small model prompt

Use when:

- the router sends a segment to `small`
- precision and speed matter most

Characteristics:

- strict limits on number of facts and episodes
- conservative inference
- extracts only the most future-useful items

### Medium model prompt

Use when:

- the router sends a segment to `medium`
- the segment is important enough to deserve fuller extraction but not the most expensive model

Characteristics:

- balanced completeness
- stable canonicalization
- stronger support for durable facts plus time-aware events

### Large model prompt

Use when:

- the router sends a segment to `large`
- the segment has high information value

Characteristics:

- highest recall
- better multi-turn event synthesis
- stronger handling of updates, temporal progression, and cross-turn event integration

## 7. Semantic memory extraction requirements

Semantic memory should include:

- entities: people, organizations, places, datasets, tools, products, projects, papers, works
- facts: stable or reusable information
- preferences: durable likes, tool preferences, style preferences, research interests
- constraints: explicit limitations, environment restrictions, deadlines, budget constraints, unsupported capabilities

Examples:

- `"InfoBudget" is a research project`
- `LoCoMo` and `LongMemEval` are selected evaluation datasets
- the user prefers turn-level timestamps aligned with LightMem
- the project must log every change to `CHANGELOG.md`

## 8. Episodic memory extraction requirements

Episodic memory should capture:

- actions
- event order
- timestamps
- cause/result
- updates or state transitions

Examples:

- the user added raw datasets to the project
- the project changed LoCoMo turn timestamps to synthesized turn-level timestamps
- LongMemEval was updated to 500ms turn-level increments

For LoCoMo and LongMemEval, turn-level timestamps are especially important because episodic retrieval depends on temporal ordering.

## 9. Timestamp handling in prompts

The segment text already contains LightMem-style lines such as:

```text
[2024-01-07T17:24:00.000, Sun] 0.Tim: ...
```

This is the right format for prompting because it lets the model:

- detect temporal order
- distinguish mention time from event time
- normalize phrases like "last week" or "next month"
- build event sequence in episodic memory

Prompt instructions should therefore require:

- use absolute timestamp if grounded
- otherwise preserve relative phrase anchored to the mention date
- assign episode `sequence` in chronological order inside the segment

## 10. Local vs API model strategy

Recommended execution strategy:

1. `small`
   - local inference backend
   - strict JSON mode if available
   - low temperature

2. `medium`
   - local inference backend
   - strict JSON mode if available
   - low temperature

3. `large`
   - API backend
   - use structured outputs or JSON mode
   - low temperature

Suggested decoding defaults:

- `temperature = 0.0` or `0.1`
- `top_p = 0.9` or lower
- schema-first output
- one-shot JSON object

## 11. Runtime implementation after prompts

The current repo supports tier-specific prompt loading in the runtime:

- `small` -> `joint_memory_extraction_small.txt`
- `medium` -> `joint_memory_extraction_medium.txt`
- `large` -> `joint_memory_extraction_large.txt`

The current runtime also includes:

1. `LocalJointExtractor` for local OpenAI-compatible inference servers
2. `APIJointExtractor` for remote OpenAI-compatible APIs
3. `TieredJointExtractor` to dispatch by the routed model tier
4. JSON parsing and schema coercion into `MemoryEntry`
5. runtime token usage and latency logging when the provider returns usage metadata
6. optional mock fallback controlled by `extractor.fallback_to_mock`

## 12. Parsing and validation recommendation

The LLM output should be validated before storage:

1. parse JSON
2. verify required top-level keys exist
3. coerce missing arrays to `[]`
4. clip `importance` and confidence values to `[0, 1]`
5. ensure episode `sequence` is contiguous
6. reject or repair malformed items

Recommended fallback flow:

1. primary extraction call
2. if JSON parse fails, run a cheap JSON-repair prompt
3. if still invalid, drop to empty memory object and log failure

## 13. Evaluation implications

After real LLM extraction is connected:

- semantic memory quality can be evaluated through QA accuracy and evidence recall
- episodic memory quality can be evaluated through temporal and session-scoped questions
- cost analysis should use real response `usage` from runtime calls
- preprocessing `token_count` should remain as an offline budget feature, not as a replacement for runtime token accounting

## 14. Short conclusion

The heuristic extractor now exists only as a fallback to make offline development and tests runnable.

The implemented design for InfoBudget is:

- route segments by information value
- use local small/medium models and an API large model
- run one joint English prompt per segment
- output a fixed JSON schema containing summary, semantic memory, episodic memory, and importance
- validate and store the result in the existing `MemoryEntry` structure
