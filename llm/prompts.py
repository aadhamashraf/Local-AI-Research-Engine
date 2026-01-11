"""
Prompt Templates for LLM Operations
Contains all prompts used throughout the research engine.
"""

# ============================================================================
# SEMANTIC CHUNKING
# ============================================================================

SEMANTIC_CHUNKING_PROMPT = """You are a document analysis expert. Your task is to split the following text into self-contained knowledge units.

Each chunk should:
- Represent a complete thought or concept
- Be between 300-800 words
- Have clear boundaries (don't split mid-sentence)
- Maintain context (include necessary background)

Text to split:
{text}

Return the chunks separated by "---CHUNK---" markers. Do not add any other commentary."""

# ============================================================================
# ENTITY EXTRACTION
# ============================================================================

ENTITY_EXTRACTION_PROMPT = """Extract key entities from the following text. Identify:

1. **Concepts**: Key ideas, theories, or principles (e.g., "Machine Learning", "Gradient Descent")
2. **Methods**: Algorithms, techniques, or approaches (e.g., "EM Algorithm", "Backpropagation")
3. **Authors**: People mentioned (e.g., "Rabiner", "Bishop")
4. **Papers**: Referenced works (e.g., "Attention Is All You Need")
5. **Tools**: Software, libraries, or frameworks (e.g., "TensorFlow", "PyTorch")

Text:
{text}

Return a JSON object with this structure:
{{
  "concepts": ["concept1", "concept2"],
  "methods": ["method1", "method2"],
  "authors": ["author1", "author2"],
  "papers": ["paper1", "paper2"],
  "tools": ["tool1", "tool2"]
}}

Only return the JSON, no other text."""

# ============================================================================
# RELATIONSHIP EXTRACTION
# ============================================================================

RELATIONSHIP_EXTRACTION_PROMPT = """Identify relationships between entities in the following text.

Entities:
{entities}

Text:
{text}

For each relationship, specify:
- source: The entity that initiates the relationship
- target: The entity that receives the relationship
- type: One of [uses, extends, contradicts, compares_to, implements, cites]
- evidence: A brief quote supporting this relationship

Return a JSON array:
[
  {{
    "source": "entity1",
    "target": "entity2",
    "type": "uses",
    "evidence": "brief quote from text"
  }}
]

Only return the JSON array, no other text."""

# ============================================================================
# ANSWER SYNTHESIS
# ============================================================================

ANSWER_SYNTHESIS_PROMPT = """You are a research assistant. Answer the user's question using ONLY the provided sources.

**CRITICAL RULES:**
1. Every factual statement MUST have a citation
2. Use the format: [source_id §section] after each claim
3. If information is not in the sources, say "Not found in sources"
4. Synthesize information from multiple sources when relevant
5. Be precise and academic in tone

Question: {question}

Sources:
{sources}

Provide a comprehensive answer with inline citations."""

# ============================================================================
# RERANKING
# ============================================================================

RERANKING_PROMPT = """You are evaluating the relevance of a text passage to a query.

Query: {query}

Passage:
{passage}

Rate the relevance on a scale of 0-10:
- 0: Completely irrelevant
- 5: Somewhat relevant, mentions related concepts
- 10: Directly answers the query with specific information

Return ONLY a single number between 0 and 10, nothing else."""

# ============================================================================
# CONTRADICTION DETECTION
# ============================================================================

CONTRADICTION_DETECTION_PROMPT = """Compare the following two statements and determine if they contradict each other.

Statement 1 (from {source1}):
{statement1}

Statement 2 (from {source2}):
{statement2}

Analyze:
1. Do these statements contradict each other?
2. If yes, what is the nature of the contradiction?
3. Are there any nuances or contexts that might resolve the contradiction?

Return a JSON object:
{{
  "contradicts": true/false,
  "explanation": "detailed explanation",
  "severity": "minor/moderate/major"
}}

Only return the JSON, no other text."""

# ============================================================================
# PAPER COMPARISON
# ============================================================================

PAPER_COMPARISON_PROMPT = """Compare the following two papers on the specified aspect.

Aspect to compare: {aspect}

Paper 1 ({source1}):
{content1}

Paper 2 ({source2}):
{content2}

Provide a structured comparison:
1. **Similarities**: What do both papers agree on?
2. **Differences**: Where do they diverge?
3. **Strengths**: What is unique or superior in each paper?
4. **Conclusion**: Which approach is more suitable for what use case?

Use citations [source1] and [source2] throughout your comparison."""

# ============================================================================
# LITERATURE REVIEW
# ============================================================================

LITERATURE_REVIEW_PROMPT = """Generate a literature review on the topic: {topic}

Sources:
{sources}

Structure your review:
1. **Introduction**: Overview of the topic
2. **Key Themes**: Organize findings by theme
3. **Methodologies**: Common approaches used
4. **Findings**: Main results and conclusions
5. **Gaps**: What is missing or needs further research?
6. **Conclusion**: Summary and future directions

Cite all sources using [source_id §section] format."""

# ============================================================================
# QUERY ANALYSIS
# ============================================================================

QUERY_ANALYSIS_PROMPT = """Analyze the user's query to improve retrieval.

Query: {query}

Extract:
1. **Intent**: What is the user trying to find? (definition, comparison, explanation, procedure, etc.)
2. **Key Concepts**: Main topics or entities
3. **Constraints**: Any specific requirements (time period, methodology, etc.)
4. **Expanded Terms**: Synonyms or related terms that might help retrieval

Return a JSON object:
{{
  "intent": "intent type",
  "key_concepts": ["concept1", "concept2"],
  "constraints": ["constraint1"],
  "expanded_terms": ["synonym1", "related_term1"]
}}

Only return the JSON, no other text."""
