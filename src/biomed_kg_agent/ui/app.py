"""
Streamlit web interface for Biomedical Knowledge Graph Agent.

Provides natural language access to the biomedical knowledge graph with
inline document verification capabilities.
"""

import logging
from typing import Any, Optional

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from neo4j import GraphDatabase

from biomed_kg_agent.agent.core import BiomedKGAgent
from biomed_kg_agent.config import settings
from biomed_kg_agent.doc_ids import linkify_pmids
from biomed_kg_agent.nlp.persistence import get_document_by_id

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Biomedical KG Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "neo4j_connected" not in st.session_state:
    st.session_state.neo4j_connected = False
if "neo4j_driver" not in st.session_state:
    st.session_state.neo4j_driver = None


@st.cache_resource
def get_neo4j_driver() -> Optional[Any]:
    """Initialize Neo4j driver with environment variables."""
    try:
        uri = settings.NEO4J_URI
        user = settings.NEO4J_USER
        password = settings.NEO4J_PASSWORD

        if not password:
            st.error("❌ NEO4J_PASSWORD environment variable is required")
            return None

        driver = GraphDatabase.driver(uri, auth=(user, password))
        # Test connection
        with driver.session() as session:
            session.run("RETURN 1")

        st.session_state.neo4j_connected = True
        st.session_state.neo4j_driver = driver
        return driver

    except Exception as e:
        st.error(f"❌ Neo4j connection failed: {e}")
        st.session_state.neo4j_connected = False
        return None


def initialize_agent() -> Optional[BiomedKGAgent]:
    """Initialize the biomedical knowledge graph agent.

    Returns a cached agent if already created, otherwise builds one from
    current session-state settings.  Called both at startup (eager init)
    and lazily when the user sends the first chat message.
    """
    if st.session_state.agent is not None:
        return st.session_state.agent

    driver = st.session_state.get("neo4j_driver") or get_neo4j_driver()
    if not driver:
        return None

    try:
        umls_enabled = st.session_state.get("enable_umls", False)
        hint = " (UMLS linker ~30s)" if umls_enabled else ""
        with st.spinner(f"🔗 Initializing agent{hint}..."):
            agent = BiomedKGAgent(
                neo4j_driver=driver,
                model=st.session_state.get("model", settings.ANTHROPIC_MODEL),
                neo4j_database=settings.NEO4J_DATABASE,
                verbose=st.session_state.get("verbose", False),
                temperature=st.session_state.get("temperature", 0.1),
                max_tokens=settings.ANTHROPIC_MAX_TOKENS,
                min_evidence=st.session_state.get("min_evidence", 5),
                max_results=st.session_state.get("max_results", 20),
                max_history_messages=st.session_state.get("max_history", 10),
                enable_umls_linking=umls_enabled,
            )
            st.session_state.agent = agent
            return agent
    except Exception as e:
        st.error(f"Agent initialization failed: {e}")
        return None


def display_message(
    role: str, content: str, doc_ids: Optional[list[str]] = None
) -> None:
    """Display a chat message with clickable PMID links."""
    with st.chat_message(role):
        if doc_ids:
            content = linkify_pmids(content, doc_ids)
        st.markdown(content)


def display_documents(
    doc_ids: list[str],
    db_path: str,
    message_idx: int = 0,
    database_url: str | None = None,
) -> None:
    """Display cited documents with filter-first lazy loading."""
    if not doc_ids:
        return

    # Deduplicate while preserving order
    doc_ids = list(dict.fromkeys(doc_ids))

    with st.expander(f"📄 View cited documents ({len(doc_ids)} total)", expanded=False):
        # Filter box
        doc_filter = st.text_input(
            "Filter by document ID",
            placeholder="e.g., 34044732",
            help="Enter part or all of a document ID from the answer above",
            key=f"filter_{message_idx}",
        )

        if doc_filter:
            # Show filtered results
            filtered = [d for d in doc_ids if doc_filter in d]
            if filtered:
                st.caption(f"Found {len(filtered)} matching document(s)")
                for idx, doc_id in enumerate(filtered):
                    display_single_document(
                        doc_id, db_path, database_url, unique_key=f"{message_idx}_{idx}"
                    )
            else:
                st.info(f"No documents match '{doc_filter}'")
        else:
            # Show first document as example
            st.caption("Showing first cited document. Use filter above to find others.")
            display_single_document(
                doc_ids[0], db_path, database_url, unique_key=f"{message_idx}_0"
            )


def display_single_document(
    doc_id: str, db_path: str, database_url: str | None = None, unique_key: str = ""
) -> None:
    """Display a single document's metadata and abstract.

    Args:
        doc_id: Document ID to display
        db_path: Path to SQLite database
        database_url: Optional PostgreSQL database URL
        unique_key: Unique suffix for Streamlit widget keys to avoid duplicates
    """
    doc = get_document_by_id(doc_id, db_path, database_url)
    if doc:
        st.markdown(f"### Document {doc_id}: {doc.get('title', 'Untitled')}")

        # Metadata
        if doc.get("pub_year"):
            st.caption(f"Year: {doc['pub_year']}")
        if doc.get("authors"):
            st.caption(f"Authors: {doc['authors']}")
        if doc.get("journal"):
            st.caption(f"Journal: {doc['journal']}")
        if doc.get("doi"):
            st.caption(f"DOI: {doc['doi']}")

        # Abstract - use unique key to avoid Streamlit duplicate key errors
        if doc.get("text"):
            st.text_area(
                "Abstract",
                value=doc["text"],
                height=150,
                disabled=True,
                key=f"abstract_{doc_id}_{unique_key}",
            )

        # Keywords and MeSH terms
        if doc.get("keywords"):
            st.caption(f"🏷️ Keywords: {doc['keywords']}")
        if doc.get("mesh_terms"):
            st.caption(f"🏥 MeSH Terms: {doc['mesh_terms']}")

        st.divider()
    else:
        st.warning(f"Document {doc_id} not found in database")


def main() -> None:
    """Main Streamlit application."""
    st.markdown(
        "<style>[data-testid='stToolbar'], header {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )
    st.title("🧬 Biomedical Knowledge Graph Agent")
    st.markdown("Ask questions about biomedical relationships using natural language.")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Neo4j connection status
        if st.session_state.neo4j_connected:
            st.success("✅ Neo4j Connected")
        else:
            st.error("❌ Neo4j Disconnected")
            if st.button("🔄 Reconnect"):
                # Try to connect immediately
                driver = get_neo4j_driver()
                st.session_state.neo4j_driver = driver
                st.session_state.neo4j_connected = driver is not None
                st.session_state.agent = None  # Force re-initialization
                st.rerun()

        # Show current Neo4j env config for visibility
        st.caption(f"URI: {settings.NEO4J_URI}")
        st.caption(f"User: {settings.NEO4J_USER}")

        # Agent settings
        st.subheader("Agent Settings")

        # Tested models offered in the UI; custom models can be added via ANTHROPIC_MODEL env var
        available_models = [
            "claude-haiku-4-5-20251001",  # Haiku: fast/cost-effective
            "claude-sonnet-4-6",  # Sonnet: higher quality
        ]
        # Include custom model from env if not already listed
        if settings.ANTHROPIC_MODEL not in available_models:
            available_models.append(settings.ANTHROPIC_MODEL)
        default_idx = available_models.index(settings.ANTHROPIC_MODEL)
        model = st.selectbox("Model", available_models, index=default_idx)

        if (
            "model" in st.session_state
            and st.session_state.model != model
            and st.session_state.agent is not None
        ):
            st.session_state.agent.update_model(model)
            st.info(f"🔄 Model changed to {model}")
            logger.info("Model changed to %s", model)

        st.session_state.model = model

        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)

        if (
            "temperature" in st.session_state
            and st.session_state.temperature != temperature
            and st.session_state.agent is not None
        ):
            st.session_state.agent.update_temperature(temperature)
            st.info(f"🔄 Temperature updated to {temperature}")
            logger.info("Temperature changed to %s", temperature)

        st.session_state.temperature = temperature

        min_evidence = st.slider("Min Evidence", 1, 20, 5)
        st.session_state.min_evidence = min_evidence

        max_results = st.slider("Max Results", 5, 50, 20)
        st.session_state.max_results = max_results

        max_history = st.slider("Conversation Memory (messages)", 2, 30, 10, 2)
        st.session_state.max_history = max_history

        if st.session_state.agent is not None:
            if st.session_state.agent.max_history_messages != max_history:
                logger.info(
                    "max_history changed: %s -> %s",
                    st.session_state.agent.max_history_messages,
                    max_history,
                )
                st.session_state.agent.max_history_messages = max_history
            if st.session_state.agent.min_evidence != min_evidence:
                logger.info(
                    "min_evidence changed: %s -> %s",
                    st.session_state.agent.min_evidence,
                    min_evidence,
                )
                st.session_state.agent.min_evidence = min_evidence
            if st.session_state.agent.max_results != max_results:
                logger.info(
                    "max_results changed: %s -> %s",
                    st.session_state.agent.max_results,
                    max_results,
                )
                st.session_state.agent.max_results = max_results

        enable_umls = st.checkbox("Enable UMLS Linking", value=False)

        # Detect if UMLS setting changed - trigger agent re-initialization
        if (
            "enable_umls" in st.session_state
            and st.session_state.enable_umls != enable_umls
            and st.session_state.agent is not None
        ):
            umls_status = "enabled" if enable_umls else "disabled"
            st.info(f"🔄 UMLS linking {umls_status} - reinitializing agent...")
            logger.info("UMLS linking %s - reinitializing agent", umls_status)
            st.session_state.agent = None  # Force re-init with new setting

        st.session_state.enable_umls = enable_umls

        verbose = st.checkbox("Verbose Mode", value=False)
        st.session_state.verbose = verbose

        if st.session_state.agent is not None:
            if st.session_state.agent.verbose != verbose:
                logger.info(
                    "verbose changed: %s -> %s", st.session_state.agent.verbose, verbose
                )
                st.session_state.agent.verbose = verbose

        st.divider()
        st.caption(
            "ℹ️ Agent answers are grounded in up to 5 sample evidence "
            "sentences per relationship - the underlying literature is "
            "often much larger. Document IDs are also a sample; use them "
            "to spot-check claims against the original abstracts."
        )

    # Attempt auto-connect and eager agent initialization at startup
    if not st.session_state.neo4j_connected:
        with st.spinner("🔌 Connecting to Neo4j..."):
            driver = get_neo4j_driver()
            st.session_state.neo4j_driver = driver
            st.session_state.neo4j_connected = driver is not None

    # Eagerly initialize agent (loads UMLS linker) once connected
    if st.session_state.neo4j_connected:
        initialize_agent()

    # Main chat interface
    st.subheader("💬 Chat with the Agent")

    # Example queries showcasing graph capabilities
    st.markdown("**Try these example queries:**")
    col1, col2, col3 = st.columns(3)

    example_queries = [
        "What genes are implicated in both breast cancer and ovarian cancer?",
        "Explain the relationship between BRCA1 and breast cancer",
        "What chemicals are related to BRCA1?",
    ]

    with col1:
        if st.button("🔍 Example: Shared Genes", help="Graph intersection query"):
            st.session_state.example_query = example_queries[0]

    with col2:
        if st.button("📊 Example: Explain Link", help="Detailed relationship evidence"):
            st.session_state.example_query = example_queries[1]

    with col3:
        if st.button("🧪 Example: Find Chemicals", help="Type-filtered exploration"):
            st.session_state.example_query = example_queries[2]

    st.divider()

    # Display chat history
    for msg_idx, message in enumerate(st.session_state.messages):
        display_message(message["role"], message["content"], message.get("doc_ids"))

        # Display documents for agent responses
        if message["role"] == "assistant" and message.get("doc_ids"):
            display_documents(
                message["doc_ids"],
                settings.SQLITE_DB_PATH,
                msg_idx,
                settings.DATABASE_URL,
            )

    # Display persistent error (set on failure, cleared on next successful query).
    # Stored outside st.session_state.messages so it is never fed to the agent.
    if st.session_state.get("last_error"):
        st.error(st.session_state.last_error)

    # Chat input - MUST be called every run to render
    user_input = st.chat_input("Ask about biomedical relationships...")

    # Determine prompt from example query or user input
    prompt = None
    if "example_query" in st.session_state and st.session_state.example_query:
        prompt = st.session_state.example_query
        st.session_state.example_query = None  # Clear after use
    elif user_input:
        prompt = user_input

    if prompt:
        # Clear any stale error from a previous failed query.
        st.session_state.last_error = None

        # Build conversation history from previous messages BEFORE adding current prompt
        # (agent.ask() will add the current prompt itself)
        conversation_history = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                conversation_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                conversation_history.append(AIMessage(content=msg["content"]))

        # Add user message to history (for display)
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)

        # Get agent response
        agent = initialize_agent()
        if not agent:
            st.session_state.messages.pop()
            st.session_state.last_error = (
                "❌ Agent not available. Check Neo4j connection and configuration."
            )
            st.rerun()
            return

        try:
            with st.spinner("🤔 Thinking..."):
                # Pass conversation history to agent (current prompt will be added by agent.ask())
                result = agent.ask(prompt, conversation_history=conversation_history)

            # If the agent caught an API error, roll back the user message and
            # stash the error outside messages so it persists across reruns without
            # being fed to the agent as conversation history.
            if result.get("error"):
                st.session_state.messages.pop()
                st.session_state.last_error = f"❌ {result['answer']}"
                st.rerun()

            # Extract response components
            answer = result.get("answer", "No response generated")
            doc_ids = result.get("doc_ids", [])

            # Add assistant message to history
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "doc_ids": doc_ids}
            )

            # Display response
            display_message("assistant", answer, doc_ids)

            # Display cited documents
            if doc_ids:
                display_documents(
                    doc_ids,
                    settings.SQLITE_DB_PATH,
                    len(st.session_state.messages) - 1,
                    settings.DATABASE_URL,
                )

        except Exception as e:
            # Roll back the user message and stash the error outside messages so it
            # persists across reruns without being fed to the agent as history.
            st.session_state.messages.pop()
            st.session_state.last_error = f"❌ Error processing query: {e}"
            st.rerun()

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_error = None
        st.rerun()


if __name__ == "__main__":
    main()
