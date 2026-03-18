"""
RAG service module for web and CLI usage.
Extracts reusable RAG logic from rag_answer.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Import from rag_answer (use relative import)
try:
    from .rag_answer import (
        load_faiss_index,
        load_metadata,
        embed_text,
        retrieve_top_k,
        build_context_block,
        build_citations_map,
        call_llm_answer,
        call_llm_followup_explain,
        route_user_turn,
        MetaRecord,
        INDEX_DIR,
        FAISS_INDEX_PATH,
        METADATA_PATH,
        EMBED_MODEL,
        CHAT_MODEL,
        ROUTER_MODEL,
        TOP_K,
        MIN_RELEVANCE_SCORE,
        NORMALIZE_QUERY,
        STORE_RESPONSES,
        STORE_ROUTER,
        USE_CONVERSATION,
    )
except ImportError:
    # Fallback for absolute import
    from rag.rag_answer import (
        load_faiss_index,
        load_metadata,
        embed_text,
        retrieve_top_k,
        build_context_block,
        build_citations_map,
        call_llm_answer,
        call_llm_followup_explain,
        route_user_turn,
        MetaRecord,
        INDEX_DIR,
        FAISS_INDEX_PATH,
        METADATA_PATH,
        EMBED_MODEL,
        CHAT_MODEL,
        ROUTER_MODEL,
        TOP_K,
        MIN_RELEVANCE_SCORE,
        NORMALIZE_QUERY,
        STORE_RESPONSES,
        STORE_ROUTER,
        USE_CONVERSATION,
    )

# Import personalisation (use relative import)
try:
    from ..personalisation.personalisation import (
        parse_personalisation_consent,
        parse_user_profile_answer,
    )
except ImportError:
    # Fallback for absolute import
    from personalisation.personalisation import (
        parse_personalisation_consent,
        parse_user_profile_answer,
    )

# Load .env
DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)


# Global state (loaded once)
_index = None
_metadata = None
_client = None


def init_rag() -> None:
    """Initialize RAG system (load index and metadata). Call once at startup."""
    global _index, _metadata, _client
    
    if _index is not None:
        return  # Already initialized
    
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in environment")
    
    _client = OpenAI()
    
    print(f"Loading FAISS index from: {FAISS_INDEX_PATH.resolve()}")
    _index = load_faiss_index(FAISS_INDEX_PATH)
    
    print(f"Loading metadata from: {METADATA_PATH.resolve()}")
    _metadata = load_metadata(METADATA_PATH)
    
    print(f"RAG initialized: {len(_metadata)} metadata records")


def get_client() -> OpenAI:
    """Get OpenAI client (initializes if needed)."""
    global _client
    if _client is None:
        init_rag()
    return _client


def rag_handle_message(
    message: str,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Handle a user message in the RAG system.
    
    Args:
        message: User's message
        state: Current conversation state dict with keys:
            - awaiting_consent: bool
            - awaiting_role: bool
            - awaiting_familiarity: bool
            - personalisation_enabled: bool
            - user_profile: dict with role, edi_familiarity
            - conversation_id: Optional[str]
            - previous_response_id: Optional[str]
            - last_user_question: Optional[str]
            - last_answer: Optional[str]
            - last_results: List[Tuple[int, float, MetaRecord]]
    
    Returns:
        Dict with keys:
            - reply: str (assistant's reply)
            - sources: List[dict] (retrieved sources)
            - citations_map: dict (citation mapping)
            - state: dict (updated state)
            - error: Optional[str] (error message if any)
    """
    global _index, _metadata
    
    if _index is None or _metadata is None:
        init_rag()
    
    client = get_client()
    
    # Initialize state if needed
    if "awaiting_consent" not in state:
        state["awaiting_consent"] = True
        state["awaiting_role"] = False
        state["awaiting_familiarity"] = False
        state["personalisation_enabled"] = False
        state["user_profile"] = {"role": None, "edi_familiarity": None}
        state["conversation_id"] = None
        state["previous_response_id"] = None
        state["last_user_question"] = None
        state["last_answer"] = None
        state["last_results"] = []
    
    # Handle onboarding flow
    if state["awaiting_consent"]:
        decision = parse_personalisation_consent(
            client,
            message,
            router_model=ROUTER_MODEL,
            store_router=STORE_ROUTER,
        )
        
        if decision == "yes":
            state["personalisation_enabled"] = True
            state["awaiting_consent"] = False
            state["awaiting_role"] = True
            return {
                "reply": "Great. First question: what is your role? (student / teacher / generic user)",
                "sources": [],
                "citations_map": {},
                "state": state,
                "error": None,
            }
        elif decision == "no":
            state["personalisation_enabled"] = False
            state["awaiting_consent"] = False
            state["awaiting_role"] = False
            state["awaiting_familiarity"] = False
            return {
                "reply": "No problem — I'll answer generically. You can now ask your question.",
                "sources": [],
                "citations_map": {},
                "state": state,
                "error": None,
            }
        else:
            return {
                "reply": "Sorry — please reply with 'yes' or 'no'.",
                "sources": [],
                "citations_map": {},
                "state": state,
                "error": None,
            }
    
    if state["personalisation_enabled"] and state["awaiting_role"]:
        upd = parse_user_profile_answer(
            client,
            message,
            router_model=ROUTER_MODEL,
            store_router=STORE_ROUTER,
            current_profile=state["user_profile"],
        )
        if upd.get("role"):
            state["user_profile"]["role"] = upd["role"]
            state["awaiting_role"] = False
            state["awaiting_familiarity"] = True
            return {
                "reply": "Thanks. Second question: how familiar are you with EDI? (low / medium / high)",
                "sources": [],
                "citations_map": {},
                "state": state,
                "error": None,
            }
        else:
            return {
                "reply": "Please reply with one of: student, teacher, generic user",
                "sources": [],
                "citations_map": {},
                "state": state,
                "error": None,
            }
    
    if state["personalisation_enabled"] and state["awaiting_familiarity"]:
        upd = parse_user_profile_answer(
            client,
            message,
            router_model=ROUTER_MODEL,
            store_router=STORE_ROUTER,
            current_profile=state["user_profile"],
        )
        if upd.get("edi_familiarity"):
            state["user_profile"]["edi_familiarity"] = upd["edi_familiarity"]
            state["awaiting_familiarity"] = False
            return {
                "reply": f"Got it — role={state['user_profile']['role']}, EDI familiarity={state['user_profile']['edi_familiarity']}. You can now ask your question.",
                "sources": [],
                "citations_map": {},
                "state": state,
                "error": None,
            }
        else:
            return {
                "reply": "Please reply with one of: low, medium, high",
                "sources": [],
                "citations_map": {},
                "state": state,
                "error": None,
            }
    
    # Handle actual questions
    # Route the turn
    route = route_user_turn(
        client,
        message,
        last_user_question=state["last_user_question"],
        last_answer=state["last_answer"],
        last_results=state["last_results"],
    )
    
    # Handle follow-up explain (no new retrieval)
    if route["action"] == "FOLLOWUP_EXPLAIN" and state["last_user_question"] and state["last_answer"] and state["last_results"]:
        answer, response_id = call_llm_followup_explain(
            client,
            message,
            prev_user_question=state["last_user_question"],
            prev_assistant_answer=state["last_answer"],
            prev_results=state["last_results"],
            conversation_id=state["conversation_id"],
            previous_response_id=state["previous_response_id"],
        )
        
        if not USE_CONVERSATION:
            state["previous_response_id"] = response_id
        
        state["last_user_question"] = message
        state["last_answer"] = answer
        
        # Build sources list for response
        sources = []
        for rank, score, r in state["last_results"]:
            sources.append({
                "rank": rank,
                "score": float(score),
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "title": r.title,
                "source_path": r.source_path,
                "char_start": r.char_start,
                "char_end": r.char_end,
            })
        
        citations_map = build_citations_map(state["last_results"])
        
        return {
            "reply": answer,
            "sources": sources,
            "citations_map": citations_map,
            "state": state,
            "error": None,
        }
    
    # Handle new query or follow-up refine (needs retrieval)
    standalone_question = route.get("standalone_question", "") or message
    
    # Embed and retrieve
    qvec = embed_text(client, standalone_question)
    results = retrieve_top_k(_index, _metadata, qvec, top_k=TOP_K)
    
    # Check relevance gate
    best_score = results[0][1] if results else float("-inf")
    if not results or best_score < MIN_RELEVANCE_SCORE:
        answer = (
            f"I couldn't find sufficiently relevant evidence in the EDI document collection to answer that. "
            f"(best similarity score={best_score:.4f}, threshold={MIN_RELEVANCE_SCORE:.2f})\n\n"
            "Try rephrasing the question using EDI-related terms, or ask something that should be covered by the resources."
        )
        
        sources = []
        citations_map = {}
        
        return {
            "reply": answer,
            "sources": sources,
            "citations_map": citations_map,
            "state": state,
            "error": None,
        }
    
    # Build context and generate answer
    context = build_context_block(results)
    answer, response_id = call_llm_answer(
        client,
        message,  # Use original message wording
        context,
        conversation_id=state["conversation_id"],
        previous_response_id=state["previous_response_id"],
    )
    
    if not USE_CONVERSATION:
        state["previous_response_id"] = response_id
    
    # Update state
    state["last_user_question"] = message
    state["last_answer"] = answer
    state["last_results"] = results
    
    # Build sources list
    sources = []
    for rank, score, r in results:
        sources.append({
            "rank": rank,
            "score": float(score),
            "chunk_id": r.chunk_id,
            "doc_id": r.doc_id,
            "title": r.title,
            "source_path": r.source_path,
            "char_start": r.char_start,
            "char_end": r.char_end,
        })
    
    citations_map = build_citations_map(results)
    
    return {
        "reply": answer,
        "sources": sources,
        "citations_map": citations_map,
        "state": state,
        "error": None,
    }


def get_initial_message() -> str:
    """Get the initial welcome message for the Q&A page."""
    return "Welcome to the EDI assistant. I answer questions grounded in the EDI document collection.\n\nBefore we dive in, can I ask you a few quick questions to tailor my answers? (yes/no)"
