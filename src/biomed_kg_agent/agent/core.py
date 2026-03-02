"""Natural language interface for agent queries using LangChain + Anthropic.

Provides LLM-powered wrapper around structured query methods using industry-standard
tools (LangChain for tool routing, Anthropic Claude for reasoning).

Architecture:
    Layer 1 (queries.py): Structured Cypher queries (reliable, tested)
    Layer 2 (core.py): LLM wrapper for natural language (this file)

Entity Resolution:
    Uses UMLS entity linking to map natural language terms to canonical CUIs,
    enabling synonym resolution ("breast cancer" -> "Malignant neoplasm of breast").
    All responses include mandatory provenance (document IDs).
"""

import logging
from typing import Any, Optional

from langchain.tools import StructuredTool, Tool
from langchain_anthropic import ChatAnthropic  # noqa: F401
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from neo4j import Driver
from pydantic import BaseModel, Field

from biomed_kg_agent.agent.queries import (
    explain_relationship,
    query_disease_genes,
    query_entity_neighbors,
    query_gene_diseases,
    query_shared_neighbors,
)
from biomed_kg_agent.doc_ids import filter_cited_ids
from biomed_kg_agent.nlp.config import LinkerConfig
from biomed_kg_agent.nlp.entity_linking import EntityLinker

logger = logging.getLogger(__name__)


class BiomedKGAgent:
    """Natural language agent for biomedical knowledge graph queries.

    Uses LangChain + Anthropic Claude to route questions to appropriate query methods.
    All biomedical facts come from Neo4j graph, not LLM generation.

    Example:
        >>> driver = GraphDatabase.driver(uri, auth=(user, password))
        >>> agent = BiomedKGAgent(driver, model="claude-haiku-4-5-20251001")
        >>> result = agent.ask("What genes are associated with breast cancer?")
        >>> print(result["answer"])
        >>> print(result["doc_ids"])  # Always includes document IDs for provenance
    """

    def __init__(
        self,
        neo4j_driver: Driver,
        model: str = "claude-haiku-4-5-20251001",
        neo4j_database: str = "neo4j",
        verbose: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        min_evidence: int = 5,
        max_results: int = 20,
        max_doc_ids_per_result: int = 10,
        max_history_messages: int = 10,
        enable_umls_linking: bool = True,
    ):
        """Initialize agent with Neo4j driver and LLM configuration.

        Args:
            neo4j_driver: Connected Neo4j driver instance
            model: Anthropic model name (e.g., "claude-haiku-4-5-20251001",
                "claude-3-opus-20240229")
            neo4j_database: Neo4j database name
            verbose: Enable verbose logging for agent execution
            temperature: LLM temperature (0.0-1.0, lower = more deterministic)
            max_tokens: Maximum tokens in LLM response (default: 4096)
            min_evidence: Minimum co-occurrence count to consider (default: 5)
            max_results: Maximum results to return from graph queries (default: 20)
            max_doc_ids_per_result: Maximum document IDs to include per result (default: 10)
            max_history_messages: Maximum conversation history messages to retain (default: 10)
            enable_umls_linking: Enable UMLS entity linking for synonym resolution (default: True)
        """
        self.driver = neo4j_driver
        self.database = neo4j_database
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.min_evidence = min_evidence
        self.max_results = max_results
        self.max_doc_ids_per_result = max_doc_ids_per_result
        self.max_history_messages = max_history_messages

        # Track document IDs from tool executions (cleared on each ask() call)
        self._collected_doc_ids: list[str] = []

        # Initialize UMLS entity linker for synonym resolution (30-40s one-time load)
        self.entity_linker: Optional[EntityLinker] = None
        if enable_umls_linking:
            logger.info("Initializing UMLS entity linker (30-40s one-time load)...")
            try:
                linker_config = LinkerConfig(
                    enabled=True,
                    core_model="en_core_sci_sm",  # Required field for entity linking
                )
                self.entity_linker = EntityLinker(linker_config)
                # Trigger lazy loading now to avoid delay on first query
                self.entity_linker._get_umls_linker()
                logger.info("UMLS entity linker ready")
            except Exception as e:
                logger.warning(f"UMLS linking unavailable: {e}")
                logger.info("   Falling back to name-based matching only")
                self.entity_linker = None

        # Initialize LLM via LangChain Anthropic
        self.llm = ChatAnthropic(
            model=model, temperature=temperature, max_tokens=max_tokens
        )

        # Create structured query tools
        self.tools = self._create_tools()

        # System prompt to reduce verbosity
        self._system_prompt = SystemMessage(
            content="""
                You are a biomedical knowledge graph assistant.

                Available Entity Types:
                - gene: Genes and gene products (e.g., BRCA1, TP53, HER2)
                - disease: Diseases and disorders (e.g., breast cancer, diabetes)
                - chemical: Drugs and chemical compounds (e.g., metformin, aspirin)
                - biological_process: Biological processes and pathways
                  (e.g., apoptosis, cell cycle)
                - cell_type: Cell types (e.g., T cells, neurons)
                - anatomy: Anatomical structures (e.g., heart, lung tissue)
                - organism: Species and organisms (e.g., mice, humans)
                - Other types: cellular_component, sequence_feature, amino_acid,
                  substance, pathology

                Tool Selection Guide:

                DEFAULT TOOL: FindEntityNeighbors
                - Works for ANY entity type: chemicals, biological processes, cell types,
                  anatomy, genes, diseases, organisms, etc.
                - Optionally filter results by entity_type (e.g., entity_type='disease')
                - Examples:
                  - "What is metformin related to?" -> FindEntityNeighbors(entity='metformin')
                  - "What diseases involve brca1?" ->
                    FindEntityNeighbors(entity='brca1', entity_type='disease')
                  - "What genes are linked to inflammation?" ->
                    FindEntityNeighbors(entity='inflammation', entity_type='gene')

                SPECIALIZED SHORTCUTS (optimized for common patterns):
                - "genes for disease X" -> FindGenesForDisease (faster gene<-disease query)
                - "diseases for gene Y" -> FindDiseasesForGene (faster disease<-gene query)

                RELATIONSHIP DETAILS:
                - "Explain relationship between X and Y" -> ExplainRelationship
                - Get detailed evidence sentences for a specific entity pair

                SHARED NEIGHBORS (Graph Intersection):
                - "What genes/drugs/etc are linked to BOTH X and Y?" -> FindSharedNeighbors
                - Shows entities at the intersection of two entity neighborhoods
                - Useful for finding commonalities between diseases, drug targets, etc.
                - Examples:
                  - "What genes are implicated in both breast cancer and ovarian cancer?"
                    -> FindSharedNeighbors(entity_a='breast cancer', entity_b='ovarian cancer',
                                           neighbor_type='gene')
                  - "Which drugs are used for both HER2+ and triple-negative breast cancer?"
                    -> FindSharedNeighbors(entity_a='HER2-positive breast cancer',
                                           entity_b='triple-negative breast cancer',
                                           neighbor_type='chemical')
                  - "What biological processes connect diabetes and cancer?"
                    -> FindSharedNeighbors(entity_a='diabetes', entity_b='cancer',
                                           neighbor_type='biological_process')

                FOLLOW-UP QUESTIONS:
                - ALWAYS call tools with fresh queries - NEVER answer from memory alone
                - Use conversation context to interpret intent (e.g., "What about X?" means
                  "apply query to entity X from previous results")
                - Reference previous results in your BRIEF SUMMARY section
                - Always fetch fresh data even if entity was mentioned before

                OUTPUT STRUCTURE:

                Your response should have THREE sections:

                1. BRIEF SUMMARY (2-3 sentences)
                   Use ONLY these safe synthesis patterns:
                   [OK] Conversational context: "X, which appeared in the previous query, is..."
                   [OK] Ranking/strength: "primarily associated with", "most strongly linked to"
                   [OK] Aggregation: "combined evidence across N results", "total of N documents"
                   [OK] Categorization: "These fall into X categories: ..."
                   [OK] Comparison: "higher/lower evidence than", "similar pattern to"

                   FORBIDDEN in summary:
                   [X] New medical claims not in tool evidence
                   [X] Mechanisms or causality not explicitly stated in evidence
                   [X] Modifying entity names from tool output

                2. SELECTION RATIONALE
                   Explain which results you selected and why:
                   - State your threshold explicitly (e.g., "selecting drugs with >100 docs")
                   - ALWAYS explain why you excluded high-evidence results (>500 docs)
                   - If excluding top-3 results, justify each exclusion
                   - Distinguish between: specific drugs, drug classes, targets, biomarkers

                3. DETAILED RESULTS
                   Use this format for EACH result:
                   - **Entity Name** -- N documents; brief description [doc_id1, doc_id2, ...]

                   CRITICAL RULES for detailed results:

                   ONLY create bullet points for entities with
                   "ENTITY NAME (exact name from tool output): X" in tool output.

                   FORBIDDEN:
                   [X] Extracting names from Evidence text (e.g., "ADCs", "T-DM1", "TCGA")
                   [X] Creating bullet points for concepts without "ENTITY NAME: ..."
                   [X] Combining entity names from multiple results
                   [X] Modifying entity names (exemption: conversion to title case)

                   If you see interesting information in Evidence but no "ENTITY NAME: ..." for it,
                   mention it in your BRIEF SUMMARY, NOT in DETAILED RESULTS.

                   REQUIRED:
                   - Each Result is completely isolated - use information from ONE result only
                   - Use the EXACT entity name from "ENTITY NAME (use this exact name):" field
                   - Copy the EXACT document count from "Documents: N"
                   - Keep descriptions brief (2-3 sentences max)
                   - ONLY paraphrase or quote the shown evidence sentences;
                     do NOT add external knowledge

                EXAMPLE 1 (context-aware follow-up with specialized tool):

                User: "What genes are related to breast cancer?"
                Agent: [calls FindGenesForDisease, returns BRCA1, TP53, HER2...]

                User: "What about BRCA1?"
                [Agent interprets: previous query was about genes/diseases, so user
                likely wants more information about diseases for BRCA1]
                [Agent calls FindDiseasesForGene('BRCA1')]

                BRCA1, which appeared as a top gene for breast cancer in our previous
                query, is primarily associated with breast and ovarian cancers (combined
                75 documents across main results). Triple-negative breast cancer appears
                as a notable subtype.

                Selection Rationale:
                Selecting top disease associations with >5 documents. Focusing on distinct
                disease types rather than duplicate entries. Results 2, 5, 8 excluded as
                redundant variations of "breast cancer" with lower evidence.

                Diseases Associated with BRCA1:
                - **breast cancer** -- 65 documents; BRCA1 germline mutations identified in
                  hereditary breast cancer cohorts. Patients assessed for DNA repair deficiency
                  in clinical treatment studies [34607064, 34619284, 34638847]
                - **ovarian cancer** -- 10 documents; BRCA1 carriers assessed for familial
                  ovarian cancer risk in clinical studies [34726332, 35772246, 36928661]
                - **triple-negative breast cancer** -- 9 documents; BRCA1-mutant tumors
                  classified as triple-negative subtype in pathology reports
                  [35340889, 38613358, 39332782]

                EXAMPLE 2 (generic query with default tool):

                User: "What is tamoxifen related to?"
                [Agent calls FindEntityNeighbors(entity='tamoxifen')]

                Tamoxifen is primarily associated with breast cancer treatment and related
                biological entities (120+ total documents). The strongest associations are
                with breast cancer, estrogen receptors, and resistance mechanisms.

                Selection Rationale:
                Selecting top associations with >10 documents across all entity types.
                Excluded generic terms like "patients" (450 docs) and "treatment" (380 docs)
                as too broad. Focused on specific diseases, genes, and biological processes.

                Entities Related to Tamoxifen:
                - **breast cancer** -- 89 documents; tamoxifen administered in estrogen
                  receptor-positive breast cancer treatment regimens. Adjuvant tamoxifen
                  therapy outcomes evaluated in randomized clinical trials
                  [34567123, 34890234, 35012345]
                - **estrogen receptor** -- 45 documents; tamoxifen binding to estrogen
                  receptors measured in tumor tissue assays [34678901, 35123456, 35234567]
                - **drug resistance** -- 23 documents; tamoxifen resistance observed in
                  long-term adjuvant therapy cohorts [35345678, 35456789, 35567890]

                Examples of INCORRECT behavior (DO NOT DO THIS):

                [X] WRONG - Extracting from Evidence:
                Tool output: Evidence: "...HER2 is a transport gate for ADCs like T-DM1..."
                Your answer: - **ADCs** -- 62 documents; antibody-drug conjugates...
                Problem: No "ENTITY NAME: ADCs" exists - you extracted it from Evidence text

                [X] WRONG - Combining results:
                Result X: ENTITY NAME: entity-name | Documents: N
                Result Y: Evidence: "...full-name (entity-name)..."
                Your answer: - **entity-name** (full-name) -- N documents [doc_ids...]
                Problem: You combined Result X's entity name with Result Y's evidence text

                """
        )

        # Initialize LangChain agent
        self.agent = create_react_agent(
            self.llm, self.tools, messages_modifier=self._system_prompt
        )

        logger.info(
            f"Initialized BiomedKGAgent with model: {model}, "
            f"min_evidence: {min_evidence}, max_results: {max_results}, "
            f"UMLS linking: {enable_umls_linking}"
        )

    def update_temperature(self, temperature: float) -> None:
        """Swap LLM with new temperature without reloading UMLS/tools."""
        self.temperature = temperature
        self.llm = ChatAnthropic(
            model=self.model, temperature=temperature, max_tokens=self.max_tokens
        )
        self.agent = create_react_agent(
            self.llm, self.tools, messages_modifier=self._system_prompt
        )

    def update_model(self, model: str) -> None:
        """Swap LLM model without reloading UMLS/tools."""
        self.model = model
        self.llm = ChatAnthropic(
            model=model, temperature=self.temperature, max_tokens=self.max_tokens
        )
        self.agent = create_react_agent(
            self.llm, self.tools, messages_modifier=self._system_prompt
        )

    def _create_tools(self) -> list[Tool]:
        """Create LangChain tools wrapping structured query methods."""

        # Schema for FindEntityNeighbors tool
        class FindEntityNeighborsInput(BaseModel):
            entity: str = Field(
                description="Entity name to search (e.g., 'breast cancer', 'metformin')"
            )
            entity_type: Optional[str] = Field(
                default=None,
                description=(
                    "Optional filter by entity type. "
                    "Valid values: 'chemical', 'gene', 'disease', 'biological_process', "
                    "'cell_type', 'anatomy', 'organism', 'cellular_component', 'sequence_feature'"
                ),
            )

        # Schema for FindSharedNeighbors tool
        class FindSharedNeighborsInput(BaseModel):
            entity_a: str = Field(
                description="First entity name (e.g., 'breast cancer', 'BRCA1')"
            )
            entity_b: str = Field(
                description="Second entity name (e.g., 'ovarian cancer', 'TP53')"
            )
            neighbor_type: Optional[str] = Field(
                default=None,
                description=(
                    "Optional filter by entity type. "
                    "Valid values: 'chemical', 'gene', 'disease', 'biological_process', "
                    "'cell_type', 'anatomy', 'organism', 'cellular_component', 'sequence_feature'"
                ),
            )

        def find_genes_for_disease(disease: str) -> str:
            """Find genes associated with a disease."""
            logger.info(f"[Agent Tool] FindGenesForDisease(disease='{disease}')")
            if self.verbose:
                print(f"Tool: FindGenesForDisease | Input: {disease}")
            results = query_disease_genes(
                self.driver,
                disease,
                min_evidence=self.min_evidence,
                limit=self.max_results,
                database=self.database,
                entity_linker=self.entity_linker,
            )
            return self._format_tool_results(results)

        def find_diseases_for_gene(gene: str) -> str:
            """Find diseases associated with a gene."""
            logger.info(f"[Agent Tool] FindDiseasesForGene(gene='{gene}')")
            if self.verbose:
                print(f"Tool: FindDiseasesForGene | Input: {gene}")
            results = query_gene_diseases(
                self.driver,
                gene,
                min_evidence=self.min_evidence,
                limit=self.max_results,
                database=self.database,
                entity_linker=self.entity_linker,
            )
            return self._format_tool_results(results)

        def find_entity_neighbors(
            entity: str, entity_type: Optional[str] = None
        ) -> str:
            """Find all entities related to a given entity.

            Args:
                entity: Entity name to search for
                entity_type: Optional filter by type (e.g., 'chemical', 'biological_process')
            """
            logger.info(
                f"[Agent Tool] FindEntityNeighbors(entity='{entity}', "
                f"type='{entity_type or 'all'}')"
            )
            if self.verbose:
                print(
                    f"Tool: FindEntityNeighbors | Input: {entity}"
                    f", Type: {entity_type or 'all'}"
                )
            results = query_entity_neighbors(
                self.driver,
                entity,
                entity_type=entity_type,
                min_evidence=self.min_evidence,
                limit=self.max_results,
                database=self.database,
                entity_linker=self.entity_linker,
            )
            return self._format_tool_results(results)

        def explain_entity_relationship(entities: str) -> str:
            """Get detailed evidence for relationship between two entities.

            Input should be two entity names separated by comma, e.g., "HER2, breast cancer"
            """
            logger.info(f"[Agent Tool] ExplainRelationship(entities='{entities}')")
            if self.verbose:
                print(f"Tool: ExplainRelationship | Input: {entities}")
            try:
                entity_a, entity_b = [e.strip() for e in entities.split(",", 1)]
            except ValueError:
                return "Error: Please provide two entity names separated by comma"

            result = explain_relationship(
                self.driver,
                entity_a,
                entity_b,
                database=self.database,
                entity_linker=self.entity_linker,
            )
            return self._format_tool_results([result] if result.get("found") else [])

        def find_shared_neighbors(
            entity_a: str, entity_b: str, neighbor_type: Optional[str] = None
        ) -> str:
            """Find entities connected to BOTH input entities.

            Args:
                entity_a: First entity name
                entity_b: Second entity name
                neighbor_type: Optional filter by entity type
            """
            logger.info(
                f"[Agent Tool] FindSharedNeighbors(a='{entity_a}', "
                f"b='{entity_b}', type='{neighbor_type or 'all'}')"
            )
            if self.verbose:
                print(f"Tool: FindSharedNeighbors | {entity_a} & {entity_b}")
            results = query_shared_neighbors(
                self.driver,
                entity_a,
                entity_b,
                neighbor_type=neighbor_type,
                min_evidence=self.min_evidence,
                limit=self.max_results,
                database=self.database,
                entity_linker=self.entity_linker,
            )
            return self._format_tool_results(results)

        return [
            Tool(
                name="FindGenesForDisease",
                func=find_genes_for_disease,
                description=(
                    "Find genes associated with a disease. "
                    "Input: disease name (e.g., 'breast cancer', 'diabetes'). "
                    "Returns genes with evidence counts and document IDs."
                ),
            ),
            Tool(
                name="FindDiseasesForGene",
                func=find_diseases_for_gene,
                description=(
                    "Find diseases associated with a gene. "
                    "Input: gene name (e.g., 'BRCA1', 'TP53', 'p53'). "
                    "Returns diseases with evidence counts and document IDs."
                ),
            ),
            StructuredTool(
                name="FindEntityNeighbors",
                func=find_entity_neighbors,
                description=(
                    "Find entities related to a given entity. "
                    "Works for chemicals, drugs, biological processes, cell types, "
                    "and anatomy. Optionally filter by entity_type (e.g., 'chemical' "
                    "for chemicals/drugs, 'biological_process' for processes). "
                    "Returns related entities with evidence counts and doc IDs."
                ),
                args_schema=FindEntityNeighborsInput,
            ),
            Tool(
                name="ExplainRelationship",
                func=explain_entity_relationship,
                description=(
                    "Get detailed evidence and explanation for relationship "
                    "between two specific entities. "
                    "Input: two entity names separated by comma (e.g., 'HER2, breast cancer'). "
                    "Returns evidence sentences and document IDs."
                ),
            ),
            StructuredTool(
                name="FindSharedNeighbors",
                func=find_shared_neighbors,
                description=(
                    "Find entities (genes, drugs, diseases, etc.) that are connected to "
                    "BOTH input entities. Use for questions like 'What genes are implicated "
                    "in both breast cancer and ovarian cancer?' or 'Which drugs are used "
                    "for both HER2+ and triple-negative breast cancer?'. "
                    "Returns shared neighbors with combined evidence from both relationships."
                ),
                args_schema=FindSharedNeighborsInput,
            ),
        ]

    def _format_tool_results(self, results: list[dict[str, Any]]) -> str:
        """Format graph results as structured text for tool output.

        Side effect: Collects document IDs into self._collected_doc_ids for provenance tracking.

        CRITICAL: Each result is formatted as a completely isolated block to prevent
        cross-result synthesis (mixing information from different results).
        Entity names are clearly separated from evidence text.
        """
        if not results:
            return "No results found in knowledge graph."

        lines = []
        for i, result in enumerate(results, 1):
            # Extract the primary result entity name.
            # neighbor_name: query_disease_genes, query_gene_diseases,
            #   query_entity_neighbors, query_shared_neighbors
            # entity_a_name / entity_b_name: explain_relationship
            entity_name = None
            for key in [
                "neighbor_name",
                "entity_a_name",
                "entity_b_name",
            ]:
                if key in result and result[key]:
                    entity_name = result[key]
                    break

            # Format as isolated block with clear boundaries
            result_block = ["━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
            result_block.append(
                f"RESULT {i} (ISOLATED - DO NOT MIX WITH OTHER RESULTS)"
            )
            result_block.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            result_block.append("")
            result_block.append(f"ENTITY NAME (use this exact name): {entity_name}")
            result_block.append("")

            # Evidence counts
            result_block.append(f"Documents: {result['docs_count']}")
            result_block.append(f"Sentences: {result['sent_count']}")

            # Document IDs - collect for provenance and format for LLM
            if result["sample_doc_ids"]:
                doc_ids = result["sample_doc_ids"][: self.max_doc_ids_per_result]
                # Track document IDs for provenance
                self._collected_doc_ids.extend(str(d) for d in doc_ids)
                doc_ids_str = ", ".join(str(d) for d in doc_ids)
                result_block.append(f"Document IDs: [{doc_ids_str}]")

            # Evidence sentences - show all stored samples (up to 5) so the LLM
            # can produce grounded descriptions instead of extrapolating from one.
            if result["evidence_sentences"]:
                ev = result["evidence_sentences"]
                n_shown = len(ev)
                sent_total = result.get("sent_count", n_shown)
                result_block.append("")
                result_block.append(
                    f"Evidence ({n_shown} samples from {sent_total}"
                    " co-occurring sentences"
                    " - describe ONLY what these say):"
                )
                for idx, sentence in enumerate(ev, 1):
                    result_block.append(f'  {idx}. "{sentence}"')

            result_block.append("")
            result_block.append(f"END RESULT {i}")
            result_block.append("")

            lines.append("\n".join(result_block))

        formatted_output = "\n".join(lines)
        logger.info(f"Returning {len(results)} results to LLM")
        if self.verbose:
            logger.info(f"FULL tool output to LLM:\n{formatted_output}")
        return formatted_output

    def ask(
        self, question: str, conversation_history: Optional[list] = None
    ) -> dict[str, Any]:
        """Answer a natural language question using the knowledge graph.

        Routes the question to appropriate query method via LangChain agent,
        retrieves results, and formats them as natural language response with provenance.

        Args:
            question: Natural language question about biomedical relationships
            conversation_history: Optional list of previous messages in LangChain format.
                Expected format: [HumanMessage(...), AIMessage(...), ...]
                Automatically truncated based on max_history_messages setting.
                Default: None (single-turn query)

        Returns:
            Dictionary with:
                - question: Original question
                - answer: Natural language answer with provenance
                - doc_ids: List of document IDs for verification (deduplicated, from tool results)
                - messages: Complete message history including this turn's exchange
                    (for passing to next turn)
        """
        logger.info(f"Processing question: {question}")

        # Clear document IDs from previous execution
        self._collected_doc_ids = []

        try:
            # Execute agent (tools will populate self._collected_doc_ids with document IDs)
            # Build message list with optional conversation history
            messages = []
            if conversation_history:
                # Truncate based on max_history_messages setting
                messages = conversation_history[-self.max_history_messages :]
                logger.info(f"Using {len(messages)} messages from conversation history")

            # Add current question
            messages.append(HumanMessage(content=question))

            # Execute agent with full conversation context
            response = self.agent.invoke({"messages": messages})

            # LangGraph prebuilt agent returns a dict with "messages"
            # Extract the final AI message content
            answer = "No answer generated"
            if "messages" in response:
                msgs = response["messages"]
                # Find last AI message content
                for msg in reversed(msgs):
                    if (
                        isinstance(msg, AIMessage)
                        and hasattr(msg, "content")
                        and msg.content
                    ):
                        answer = msg.content
                        break

            # Use document IDs collected during tool execution (reliable)
            # Deduplicate while preserving order
            seen = set()
            doc_ids = []
            for doc_id in self._collected_doc_ids:
                if doc_id and doc_id not in seen:
                    seen.add(doc_id)
                    doc_ids.append(doc_id)

            # Filter to only IDs the LLM actually cited in its answer.
            # Falls back to the full collected set if none appear.
            doc_ids = filter_cited_ids(doc_ids, answer)

            # Log answer statistics for debugging
            bullet_count = answer.count("- **")
            logger.info(
                f"LLM generated answer with {bullet_count} bullet points"
                f", {len(doc_ids)} unique document IDs"
            )

            result: dict[str, Any] = {
                "question": question,
                "answer": answer,
                "doc_ids": doc_ids,
                "messages": response.get(
                    "messages", []
                ),  # Return full history for next turn
            }

            return result

        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                "question": question,
                "answer": f"Error processing question: {e}",
                "doc_ids": [],
                "error": str(e),
            }
