# Chroma_Lib.py
#
from __future__ import annotations

import hashlib
import re

from chromadb.errors import ChromaError, InvalidDimensionException

from tldw_chatbook.Chunking.Chunk_Lib import chunk_for_embedding
from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingFactory, EmbeddingConfigSchema
from tldw_chatbook.LLM_Calls.Summarization_General_Lib import analyze
from tldw_chatbook.config import get_chachanotes_db_path

"""Light‑weight ChromaDB helper for single‑user, local‑first apps.

Key simplifications vs. the original:
• no per‑user path hierarchy – caller passes explicit `storage_path`.
• no global metrics, Prometheus, rate‑limit or unload timers.
• model selection is explicit: every public call takes `model_id_override`; if
  absent and no default was configured, a `ValueError` is raised – no fall‑backs.
• minimal locking (only for Chroma client operations that mutate the store).
• designed to be IDE‑indexable and <500 lines so it ships nicely inside a
  Textual executable.
"""
#
# Imports
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, Sequence
from itertools import islice
import threading
#
# Third-Party Libraries
import numpy as np
import chromadb
from chromadb import Settings
from chromadb.api.models import Collection
from chromadb.api.types import QueryResult
from loguru import logger
#
# Local Imports
#
########################################################################################################################
#
# ---- tiny public type -------------------------------------------------------
ChromaIncludeLiteral = Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]


class ChromaDBManager:
    DEFAULT_COLLECTION_NAME_PREFIX = "user_embeddings_for_"
    COLLECTION_METADATA_EMBEDDING_MODEL_KEY = "embedding_model_id"
    COLLECTION_METADATA_EMBEDDING_DIM_KEY = "embedding_dimension"

    def _is_collection_not_found_error(self, e: Exception) -> bool:
        """
        Checks if an exception message indicates a collection was not found.
        NOTE: This relies on string matching and may break in future ChromaDB versions
        if their error messages change.
        """
        error_str = str(e).lower()
        # Common phrases indicating a collection is not found in ChromaDB errors
        return ("collection" in error_str and
                ("not found" in error_str or "does not exist" in error_str or f"no collection with name" in error_str or
                 "could not find collection" in error_str
                ))

    def __init__(self, user_id: str, user_embedding_config: Dict[str, Any]):
        if not user_id:
            logger.error("Initialization failed: user_id cannot be empty for ChromaDBManager.")
            raise ValueError("user_id cannot be empty for ChromaDBManager.")
        if not user_embedding_config:
            logger.error("Initialization failed: user_embedding_config cannot be empty for ChromaDBManager.")
            raise ValueError("user_embedding_config cannot be empty for ChromaDBManager.")

        self.user_id = str(user_id)
        # Store the raw config for components that might need the full structure
        self.raw_user_embedding_config = user_embedding_config
        self._lock = threading.RLock()  # Instance lock for ChromaDB operations
        self._inferred_dimensions: Dict[str, int] = {} # NEW: Cache for inferred model dimensions

        # Initialize EmbeddingFactory
        # The `user_embedding_config` should have an "embedding_config" key matching EmbeddingConfigSchema
        embeddings_config_dict = self.raw_user_embedding_config.get("embedding_config")
        if not embeddings_config_dict:
            logger.critical(
                "embedding_config not found in user_embedding_config. EmbeddingFactory cannot be initialized.")
            raise ValueError("embedding_config not configured in application settings.")

        try:
            self.embedding_factory = EmbeddingFactory(cfg=embeddings_config_dict)
            # IMPROVEMENT: Access config via a public property instead of a private attribute.
            # This assumes you add the following property to your EmbeddingFactory class:
            # @property
            # def config(self) -> EmbeddingConfigSchema:
            #     return self._cfg
            self.embedding_config_schema: EmbeddingConfigSchema = self.embedding_factory.config
        except Exception as e:
            logger.critical(f"Failed to initialize EmbeddingFactory for user '{self.user_id}': {e}", exc_info=True)
            raise RuntimeError(f"EmbeddingFactory initialization failed: {e}") from e

        # Get the base directory for all user data from the central CLI configuration.
        try:
            # This function from config.py resolves the user's data directory.
            # We take its parent to get the root folder for all user data.
            user_db_base_dir = get_chachanotes_db_path().parent
        except Exception as e:
            logger.critical(f"Failed to determine user DB base directory from config.py: {e}", exc_info=True)
            raise RuntimeError("Could not establish the base directory for user data.") from e

        # Construct the specific path for this user's ChromaDB storage.
        self.user_chroma_path: Path = (user_db_base_dir / self.user_id / "chroma_storage").resolve()
        try:
            self.user_chroma_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.critical(f"Failed to create ChromaDB path {self.user_chroma_path} for '{self.user_id}': {e}", exc_info=True)
            raise RuntimeError(f"Could not create ChromaDB storage directory: {e}") from e

        logger.info(f"ChromaDBManager for user '{self.user_id}' path: {self.user_chroma_path}")

        chroma_client_settings_config = self.raw_user_embedding_config.get("chroma_client_settings", {})
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.user_chroma_path),
                settings=Settings(
                    anonymized_telemetry=chroma_client_settings_config.get("anonymized_telemetry", False),
                    allow_reset=chroma_client_settings_config.get("allow_reset", True)
                )
            )
        except Exception as e:
            logger.critical(f"Failed to init ChromaDB Client for '{self.user_id}' at {self.user_chroma_path}: {e}", exc_info=True)
            raise RuntimeError(f"ChromaDB client initialization failed: {e}") from e

        self.default_embedding_model_id = self.embedding_config_schema.default_model_id

        if not self.default_embedding_model_id:
            logger.warning(
                f"User '{self.user_id}': No 'default_model_id' in embedding_config. "
                "Operations will require explicit 'embedding_model_id_override' or collection-defined model."
            )

        prompts_conf = self.raw_user_embedding_config.get("prompts", {})
        self.situate_context_prompt_template = prompts_conf.get(
            "situate_context_template",
            "<document>\n{doc_content}\n</document>\n\n\n\n\n"
            "Here is the chunk we want to situate within the whole document\n<chunk>\n{chunk_content}\n</chunk>\n\n"
            "Please give a short succinct context to situate this chunk within the overall document "
            "for the purposes of improving search retrieval of the chunk.\n"
            "Answer only with the succinct context and nothing else."
        )
        # Get default LLM for contextualization from the main embedding_config section,
        # not the specific model configs
        self.default_llm_for_contextualization = self.raw_user_embedding_config.get("embedding_config", {}).get(
            "default_llm_for_contextualization", "gpt-3.5-turbo"
        )

        default_model_spec = self.embedding_config_schema.models.get(
            self.default_embedding_model_id) if self.default_embedding_model_id else None
        logger.info(
            f"User '{self.user_id}' ChromaDBManager configured. "
            f"Default Embedding Model: {self.default_embedding_model_id or 'Not Set'} "
            f"(Provider: {default_model_spec.provider if default_model_spec else 'N/A'}). "
            f"Default Context LLM: {self.default_llm_for_contextualization}."
        )

    def _batched(self, iterable, n):
        it = iter(iterable)
        while True:
            batch = list(islice(it, n))
            if not batch:
                return
            yield batch

    def _clean_metadata_value(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (np.int_, np.integer)):
            return int(value)
        if isinstance(value, (np.float32, np.floating)):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        logger.warning(f"User '{self.user_id}': Converting metadata list element of type {type(value)} to string.")
        return str(value)

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        if not isinstance(metadata, dict):
            logger.warning(f"User '{self.user_id}': Non-dict metadata: {type(metadata)}. Returning empty.")
            return cleaned

        for key, value in metadata.items():
            if value is None:
                cleaned[key] = None
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, (np.int_, np.integer)):
                cleaned[key] = int(value)
            elif isinstance(value, (np.float32, np.floating)):
                cleaned[key] = float(value)
            elif isinstance(value, np.bool_):
                cleaned[key] = bool(value)
            elif isinstance(value, (list, tuple)):
                cleaned[key] = [self._clean_metadata_value(v) for v in value]
            else:
                logger.warning(f"User '{self.user_id}': Converting metadata value type {type(value)} for key '{key}' to str.")
                cleaned[key] = str(value)
        return cleaned

    def get_user_default_collection_name(self) -> str:
        # 1. Sanitize invalid characters from user_id
        sanitized_user_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(self.user_id))
        base_name = f"{self.DEFAULT_COLLECTION_NAME_PREFIX}{sanitized_user_id}"

        # 2. Enforce ChromaDB naming rules in a single, clear pass
        base_name = re.sub(r'\.\.+', '.', base_name) # Replace consecutive dots
        if len(base_name) > 63: base_name = base_name[:63] # Truncate to max length

        # Fix start/end characters if necessary
        if not base_name[0].isalnum():
            base_name = 'c' + base_name[1:]
        # Re-check length after potential modification
        if len(base_name) > 63: base_name = base_name[:63]
        if not base_name[-1].isalnum():
            base_name = base_name[:-1] + 'e'

        # Pad to min length (must be done after all other modifications)
        if len(base_name) < 3: base_name = (base_name + "___")[:3]

        # 3. Final validation check
        is_valid = (
                3 <= len(base_name) <= 63 and
                base_name[0].isalnum() and
                base_name[-1].isalnum() and
                ".." not in base_name
        )

        if is_valid:
            return base_name
        else:
            # Fallback to hash safety net
            logger.warning(
                f"User '{self.user_id}': Sanitized name '{base_name}' still invalid. Falling back to hash.")
            user_id_hash = hashlib.md5(str(self.user_id).encode()).hexdigest()
            return f"uc_{user_id_hash[:16]}"

    def _get_model_dimension(self, model_id: str) -> Optional[int]:
        """
        Helper to get embedding dimension for a given model_id.
        1. Checks pre-configured dimension in EmbeddingConfigSchema.models[model_id].dimension.
        2. If not found, infers by creating a dummy embedding (can be slow/costly).
        """
        if not model_id: return None
        # NEW: Check cache first
        if model_id in self._inferred_dimensions:
            return self._inferred_dimensions[model_id]

        model_spec = self.embedding_config_schema.models.get(model_id)
        if model_spec and model_spec.dimension is not None:  # Check added 'dimension' field
            try:
                dim = int(model_spec.dimension)
                self._inferred_dimensions[model_id] = dim # Cache configured dim as well
                return dim
            except ValueError:
                logger.error(f"User '{self.user_id}': Invalid dimension '{model_spec.dimension}' in config for model '{model_id}'.")
                return None

        # If allow_dynamic_hf is true in EmbeddingFactory, model_spec might be None here
        # if the model_id was dynamic. We still need to try embedding.
        if not model_spec and self.embedding_factory.allow_dynamic_hf:
            logger.info(f"User '{self.user_id}': Model '{model_id}' is dynamic. Attempting to infer dimension.")
        elif not model_spec:  # Not dynamic and not in config
            logger.error(f"User '{self.user_id}': Model '{model_id}' not found in configuration, cannot get dimension.")
            return None

        logger.warning(
            f"User '{self.user_id}': Dimension for model '{model_id}' not pre-configured. "
            "Attempting to infer by creating a dummy embedding (this can be slow/costly). "
            "Configure dimensions in 'embedding_config.models.[model_id].dimension' for performance."
        )
        try:
            # Use the embedding_factory to get a dummy embedding
            dummy_embedding_np: np.ndarray = self.embedding_factory.embed_one(
                text="dimension_check_string", model_id=model_id, as_list=False
            ) # type: ignore
            if dummy_embedding_np is not None and dummy_embedding_np.ndim == 1 and dummy_embedding_np.size > 0:
                dim = dummy_embedding_np.shape[0]
                logger.info(f"User '{self.user_id}': Inferred dimension for model '{model_id}' as {dim}.")
                self._inferred_dimensions[model_id] = dim # NEW: Cache the result
                return dim
            else:
                logger.error(
                    f"User '{self.user_id}': Failed to create valid dummy embedding for '{model_id}' dim inference. Got: {dummy_embedding_np}")
        except (IOError, ValueError) as e:  # Catch errors from EmbeddingFactory
            logger.error(
                f"User '{self.user_id}': Error during dummy embedding for model '{model_id}' (dim inference): {e}",
                exc_info=True)
        except Exception as e:
            logger.error(f"User '{self.user_id}': Unexpected error inferring dimension for '{model_id}': {e}",
                         exc_info=True)
        return None

    def _resolve_embedding_model_id(self,  # Largely unchanged, relies on self.default_embedding_model_id
                                    collection_name_for_context: Optional[str],
                                    embedding_model_id_override: Optional[str] = None) -> Optional[str]:
        if embedding_model_id_override:
            logger.debug(
                f"User '{self.user_id}': Using provided embedding_model_id_override: '{embedding_model_id_override}'.")
            return embedding_model_id_override

        if collection_name_for_context:
            try:
                collection = self.client.get_collection(name=collection_name_for_context)
                if collection.metadata and self.COLLECTION_METADATA_EMBEDDING_MODEL_KEY in collection.metadata:
                    coll_model_id = collection.metadata[self.COLLECTION_METADATA_EMBEDDING_MODEL_KEY]
                    logger.debug(
                        f"User '{self.user_id}': Using model_id '{coll_model_id}' from collection '{collection_name_for_context}'.")
                    return str(coll_model_id)
            except (ChromaError, ValueError) as e: # Catch both ChromaError and ValueError
                if self._is_collection_not_found_error(e):
                    logger.debug(f"User '{self.user_id}': Collection '{collection_name_for_context}' not found for model ID lookup.")
                else: # Other ChromaError or ValueError not related to "not found"
                    logger.warning(f"User '{self.user_id}': Error fetching collection '{collection_name_for_context}' for model ID: {e}.")
            except Exception as e: # Catch any other unexpected error
                 logger.warning(f"User '{self.user_id}': Unexpected error fetching coll '{collection_name_for_context}' for model ID: {e}.")
        if self.default_embedding_model_id:
            logger.debug(
                f"User '{self.user_id}': Using manager's default_embedding_model_id: '{self.default_embedding_model_id}'.")
            return self.default_embedding_model_id

        logger.warning(
            f"User '{self.user_id}': No embedding model ID resolved for collection '{collection_name_for_context}'.")
        return None

    def get_or_create_collection(self, collection_name: Optional[str] = None,  # Updated metadata logic
                                 collection_metadata: Optional[Dict[str, Any]] = None) -> Collection:
        name_to_use = collection_name or self.get_user_default_collection_name()
        with self._lock:  # Use instance lock for DB operations
            try:
                final_metadata = self._clean_metadata(collection_metadata) if collection_metadata else {}

                # If creating, try to set model and dimension from resolved or default model
                # This is for when the collection is first created. store_in_chroma handles updates.
                model_id_for_meta = final_metadata.get(self.COLLECTION_METADATA_EMBEDDING_MODEL_KEY)
                if not model_id_for_meta:
                    model_id_for_meta = self._resolve_embedding_model_id(None, None) # Use manager default if available
                if model_id_for_meta:
                    if self.COLLECTION_METADATA_EMBEDDING_MODEL_KEY not in final_metadata:
                        final_metadata[self.COLLECTION_METADATA_EMBEDDING_MODEL_KEY] = model_id_for_meta
                    if self.COLLECTION_METADATA_EMBEDDING_DIM_KEY not in final_metadata:
                        dim = self._get_model_dimension(model_id_for_meta)
                        if dim is not None:
                            final_metadata[self.COLLECTION_METADATA_EMBEDDING_DIM_KEY] = dim

                collection = self.client.get_or_create_collection(
                    name=name_to_use,
                    metadata=final_metadata if final_metadata else None  # Pass cleaned/augmented metadata
                )
                logger.info(
                    f"User '{self.user_id}': Accessed/Created collection '{name_to_use}' with metadata: {collection.metadata}.")
                return collection
            except ValueError as ve:
                logger.error(f"User '{self.user_id}': Invalid collection name '{name_to_use}' or metadata: {ve}",
                             exc_info=True)
                raise RuntimeError(
                    f"Failed to access/create collection '{name_to_use}': Invalid name/metadata.") from ve
            except ChromaError as ce:
                logger.error(f"User '{self.user_id}': ChromaDB error for collection '{name_to_use}': {ce}",
                             exc_info=True)
                raise RuntimeError(f"Failed to access/create collection '{name_to_use}': {ce}") from ce
            except Exception as e:
                logger.error(f"User '{self.user_id}': Unexpected error for collection '{name_to_use}': {e}",
                             exc_info=True)
                raise RuntimeError(f"Failed to access/create collection '{name_to_use}': {e}") from e

    # --- Async Situate Context ---
    # To make this truly async, `process_and_store_content` would need to be `async def`
    # and use `await asyncio.gather(*[self._situate_context_async(...) for chunk in chunks])`
    # For now, it's a placeholder. `analyze` also needs an async version.
    async def _situate_context_async(self, api_name_for_context: str, doc_content: str, chunk_content: str) -> str:
        # Requires `analyze` to be async or run in a thread pool executor.
        # This is a conceptual placeholder.
        # loop = asyncio.get_running_loop()
        # prompt = self.situate_context_prompt_template.format(doc_content=doc_content, chunk_content=chunk_content)
        # try:
        #     # Example: If analyze is sync, run it in a thread
        #     response = await loop.run_in_executor(
        #         None, # Default executor
        #         analyze,
        #         api_name_for_context,
        #         prompt,
        #         self.raw_user_embedding_config # Pass the full config if analyze needs it
        #     )
        #     return response.strip() if response else ""
        # except Exception as e:
        #    logger.error(f"User '{self.user_id}': Async situate_context LLM '{api_name_for_context}': {e}", exc_info=True)
        #    return ""
        logger.warning("Async situate_context called but using synchronous fallback.")
        return self.situate_context(api_name_for_context, doc_content, chunk_content)

    def situate_context(self, api_name_for_context: str, doc_content: str, chunk_content: str) -> str:
        prompt = self.situate_context_prompt_template.format(doc_content=doc_content, chunk_content=chunk_content)
        try:
            # FIXME
            response = analyze(
                api_name=api_name_for_context,
                prompt=prompt,
                user_embedding_config=self.raw_user_embedding_config  # Pass the main config
            )
            # api_name: str,
            # input_data: Any,
            # custom_prompt_arg: Optional[str],
            # api_key: Optional[str] = None,
            # system_message: Optional[str] = None,
            # temp: Optional[float] = None,
            # streaming: bool = False,
            # recursive_summarization: bool = False,
            # chunked_summarization: bool = False,  # Summarize chunks separately & combine
            # chunk_options: Optional[dict] = None
            return response.strip() if response else ""
        except Exception as e:
            logger.error(f"User '{self.user_id}': Error in situate_context with LLM '{api_name_for_context}': {e}",
                         exc_info=True)
            return ""

    # process_and_store_content (and other methods using embeddings) will now use self.embedding_factory
    # Consider making process_and_store_content async if create_contextualized is True often
    def process_and_store_content(self,  # Uses embedding_factory
                                  content: str,
                                  media_id: Union[int, str],
                                  file_name: str,
                                  collection_name: Optional[str] = None,
                                  embedding_model_id_override: Optional[str] = None,
                                  create_embeddings: bool = True,
                                  create_contextualized: bool = False,
                                  llm_model_for_context: Optional[str] = None,
                                  chunk_options: Optional[Dict] = None):
        _collection_name_for_ops = collection_name or self.get_user_default_collection_name()
        target_collection_obj = self.get_or_create_collection(_collection_name_for_ops)
        actual_collection_name = target_collection_obj.name

        current_op_embedding_model_id = self._resolve_embedding_model_id(
            actual_collection_name, embedding_model_id_override
        )

        if create_embeddings and not current_op_embedding_model_id:
            msg = "Cannot create embeddings: No embedding model ID resolved."
            logger.error(f"User '{self.user_id}': Media_id {media_id}. {msg}")
            raise ValueError(msg)

        effective_llm_model_for_context = llm_model_for_context or self.default_llm_for_contextualization

        logger.info(
            f"User '{self.user_id}': Processing media_id {media_id} for collection '{actual_collection_name}' "
            f"Embed Model: '{current_op_embedding_model_id or 'N/A'}'. Contextualize: {create_contextualized} "
            f"(LLM: '{effective_llm_model_for_context if create_contextualized else 'N/A'}')."
        )
        try:
            chunks = chunk_for_embedding(content, file_name, custom_chunk_options=chunk_options or {})
            if not chunks:
                logger.warning(f"User '{self.user_id}': No chunks for media_id {media_id}. Skipping.")
                return

            if create_embeddings:
                docs_for_chroma, texts_for_embedding_generation = [], []
                for chunk in chunks:
                    chunk_text = chunk['text']
                    docs_for_chroma.append(chunk_text)
                    if create_contextualized:
                        context_summary = self.situate_context(effective_llm_model_for_context, content, chunk_text)
                        texts_for_embedding_generation.append(f"{chunk_text}\n\nContextual Summary: {context_summary}")
                    else:
                        texts_for_embedding_generation.append(chunk_text)

                if not texts_for_embedding_generation:
                    logger.warning(f"User '{self.user_id}': No texts to embed for media_id {media_id}.")
                    return

                try:
                    embeddings_array: np.ndarray = self.embedding_factory.embed(
                        texts=texts_for_embedding_generation,
                        model_id=current_op_embedding_model_id,
                        as_list=False
                    ) # type: ignore
                except (IOError, ValueError) as e:
                    logger.error(f"User '{self.user_id}': EmbeddingFactory error for media_id {media_id}: {e}", exc_info=True)
                    raise RuntimeError(f"Failed to generate embeddings: {e}") from e

                ids = [f"{media_id}_chunk_{i}" for i in range(len(chunks))]
                metadatas = []
                for i, chunk_info in enumerate(chunks):
                    meta = {
                        "media_id": str(media_id), "chunk_index": i, "total_chunks": len(chunks),
                        "file_name": str(file_name), "contextualized": create_contextualized,
                        "original_chunk_text_ref": chunk_info['text'][:200] + "..." if len(
                            chunk_info['text']) > 200 else chunk_info['text']
                    }
                    meta.update(chunk_info.get('metadata', {}))
                    if create_contextualized:
                        context_part = texts_for_embedding_generation[i].split("\n\nContextual Summary: ", 1)
                        if len(context_part) > 1: meta["contextual_summary_ref"] = context_part[1][:500]
                    metadatas.append(meta)

                self.store_in_chroma(
                    collection_name=actual_collection_name,
                    texts=docs_for_chroma,
                    embeddings=embeddings_array,
                    ids=ids,
                    metadatas=metadatas,
                    embedding_model_id_for_dim_check=current_op_embedding_model_id
                )
            logger.info(f"User '{self.user_id}': Finished processing media_id {media_id}")

        except ValueError as ve:
            logger.error(
                f"User '{self.user_id}': Input/config error (media {media_id}, coll '{actual_collection_name}'): {ve}",
                exc_info=True)
            raise
        except RuntimeError as rte:
            logger.error(
                f"User '{self.user_id}': Runtime error (media {media_id}, coll '{actual_collection_name}'): {rte}",
                exc_info=True)
            raise
        except Exception as e:
            logger.error(
                f"User '{self.user_id}': Unexpected error (media {media_id}, coll '{actual_collection_name}'): {e}",
                exc_info=True)
            raise

    def store_in_chroma(self, collection_name: str,  # Logic for dim check and metadata updated
                        texts: List[str],
                        embeddings: Union[np.ndarray, List[List[float]]],
                        ids: List[str], metadatas: List[Dict[str, Any]],
                        embedding_model_id_for_dim_check: Optional[str] = None,
                        recreate_on_dim_mismatch: bool = False):
        if not all([texts, ids, metadatas]) or embeddings is None:
            raise ValueError("Texts, ids, metadatas, and embeddings must be provided and non-empty.")

        num_embeddings = len(embeddings) if isinstance(embeddings, list) else embeddings.shape[0]
        if not (len(texts) == num_embeddings == len(ids) == len(metadatas)):
            msg = (f"Input list length mismatch: Texts({len(texts)}), Embeddings({num_embeddings}), "
                   f"IDs({len(ids)}), Metadatas({len(metadatas)})")
            logger.error(f"User '{self.user_id}': {msg}")
            raise ValueError(msg)

        embeddings_list: List[List[float]]
        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim == 1:  # Handle single embedding case if it comes as 1D array
                embeddings_list = [embeddings.tolist()]
            elif embeddings.ndim == 2:
                embeddings_list = embeddings.tolist()
            else:
                raise ValueError(f"Embeddings numpy array has unexpected shape: {embeddings.shape}")
        elif isinstance(embeddings, list) and all(isinstance(e, list) for e in embeddings):
            embeddings_list = embeddings
        else:
            raise TypeError("Embeddings must be List[List[float]] or 2D/1D np.ndarray.")

        if not embeddings_list or not embeddings_list[0] or not isinstance(embeddings_list[0][0], (float, int)):
            raise ValueError("No valid numerical embeddings provided after conversion.")

        new_embedding_dim = len(embeddings_list[0])
        with self._lock:
            try: current_collection = self.client.get_collection(name=collection_name)
            except (ChromaError, ValueError) as e:
                if self._is_collection_not_found_error(e):
                    logger.error(f"User '{self.user_id}': Collection '{collection_name}' not found in store_in_chroma. Should be pre-created.")
                    current_collection = self.get_or_create_collection(collection_name) # Fallback
                else:
                    logger.error(f"User '{self.user_id}': Failed to get collection '{collection_name}': {e}", exc_info=True)
                    raise RuntimeError(f"Failed to get collection '{collection_name}': {e}") from e
            except Exception as e:
                logger.error(f"User '{self.user_id}': Unexpected error getting collection '{collection_name}': {e}", exc_info=True)
                raise RuntimeError(f"Unexpected error getting collection '{collection_name}': {e}") from e

            logger.info(
                f"User '{self.user_id}': Storing {len(embeddings_list)} items (dim: {new_embedding_dim}) "
                f"in Collection '{current_collection.name}'. Model for new: '{embedding_model_id_for_dim_check or 'Unknown'}'."
            )
            try:
                cleaned_metadatas = [self._clean_metadata(m) for m in metadatas]
                collection_meta = current_collection.metadata or {}

                existing_dim_from_meta_str = collection_meta.get(self.COLLECTION_METADATA_EMBEDDING_DIM_KEY)
                existing_dim_from_meta = int(
                    existing_dim_from_meta_str) if existing_dim_from_meta_str is not None else None

                recreate_collection = False
                if existing_dim_from_meta is not None and existing_dim_from_meta != new_embedding_dim:
                    logger.warning(
                        f"User '{self.user_id}': Dim mismatch for '{current_collection.name}'. Meta: {existing_dim_from_meta}, New: {new_embedding_dim}. Recreating.")
                    recreate_collection = True
                elif existing_dim_from_meta is None and current_collection.count() > 0:
                    sample = current_collection.get(limit=1, include=['embeddings'])
                    if sample and sample.get('embeddings') and sample['embeddings'][0]:
                        sampled_dim = len(sample['embeddings'][0])
                        if sampled_dim != new_embedding_dim:
                            logger.warning(
                                f"User '{self.user_id}': Dim mismatch (sampled) for '{current_collection.name}'. Sampled: {sampled_dim}, New: {new_embedding_dim}. Recreating.")
                            recreate_collection = True
                        else:  # Sampled dim matches, update metadata
                            collection_meta[self.COLLECTION_METADATA_EMBEDDING_DIM_KEY] = sampled_dim
                            if embedding_model_id_for_dim_check and self.COLLECTION_METADATA_EMBEDDING_MODEL_KEY not in collection_meta:
                                collection_meta[
                                    self.COLLECTION_METADATA_EMBEDDING_MODEL_KEY] = embedding_model_id_for_dim_check
                            current_collection.modify(metadata=self._clean_metadata(collection_meta))
                            logger.info(
                                f"User '{self.user_id}': Updated metadata for '{current_collection.name}' with inferred dim {sampled_dim}.")
            except Exception as e:
                logger.error(f"User '{self.user_id}': Error updating metadata for collection '{current_collection.name}': {e}", exc_info=True)
                raise RuntimeError(f"Error updating metadata for collection '{current_collection.name}': {e}") from e

            # --- IMPROVEMENT: SAFER COLLECTION RECREATION LOGIC ---
            if recreate_collection:
                if recreate_on_dim_mismatch:
                    logger.warning(
                        f"User '{self.user_id}': Dimension mismatch for '{current_collection.name}'. "
                        f"Existing: {existing_dim_from_meta}, New: {new_embedding_dim}. Recreating as requested."
                    )
                    preserved_meta = {k: v for k, v in collection_meta.items() if k not in [self.COLLECTION_METADATA_EMBEDDING_DIM_KEY, self.COLLECTION_METADATA_EMBEDDING_MODEL_KEY]}
                    self.client.delete_collection(name=current_collection.name)
                    new_coll_meta = preserved_meta
                    new_coll_meta[self.COLLECTION_METADATA_EMBEDDING_DIM_KEY] = new_embedding_dim
                    if embedding_model_id_for_dim_check:
                        new_coll_meta[self.COLLECTION_METADATA_EMBEDDING_MODEL_KEY] = embedding_model_id_for_dim_check
                    current_collection = self.client.create_collection(name=current_collection.name, metadata=self._clean_metadata(new_coll_meta))
                    logger.info(f"User '{self.user_id}': Collection '{current_collection.name}' recreated with new dimension {new_embedding_dim}.")
                else:
                    # Default behavior: raise an error to prevent accidental data loss.
                    msg = (f"Dimension mismatch for collection '{current_collection.name}'. "
                           f"Existing dimension is {existing_dim_from_meta}, but new data has dimension {new_embedding_dim}. "
                           f"Aborting to prevent data loss. To override, call `store_in_chroma` with `recreate_on_dim_mismatch=True`.")
                    logger.error(msg)
                    raise InvalidDimensionException(msg)
            # --- END IMPROVEMENT ---

            # Update metadata if needed
            final_collection_meta_to_set = {}
            current_coll_meta_from_db = current_collection.metadata or {}
            if current_coll_meta_from_db.get(self.COLLECTION_METADATA_EMBEDDING_DIM_KEY) != new_embedding_dim:
                final_collection_meta_to_set[self.COLLECTION_METADATA_EMBEDDING_DIM_KEY] = new_embedding_dim
            if embedding_model_id_for_dim_check and current_coll_meta_from_db.get(self.COLLECTION_METADATA_EMBEDDING_MODEL_KEY) != embedding_model_id_for_dim_check:
                final_collection_meta_to_set[self.COLLECTION_METADATA_EMBEDDING_MODEL_KEY] = embedding_model_id_for_dim_check
            if final_collection_meta_to_set:
                updated_meta = {**current_coll_meta_from_db, **final_collection_meta_to_set}
                current_collection.modify(metadata=self._clean_metadata(updated_meta))

            try:
                current_collection.upsert(documents=texts, embeddings=embeddings_list, ids=ids, metadatas=cleaned_metadatas)
                logger.info(f"User '{self.user_id}': Upserted {len(ids)} items to '{current_collection.name}'.")
            except (InvalidDimensionException, ChromaError, Exception) as e:
                logger.error(f"User '{self.user_id}': Error upserting to '{current_collection.name}': {e}", exc_info=True)
                raise RuntimeError(f"ChromaDB operation failed during upsert: {e}") from e

            except InvalidDimensionException as ide:
                logger.error(
                    f"User '{self.user_id}': ChromaDB InvalidDimensionException for '{current_collection.name}': {ide}.",
                    exc_info=True)
                raise RuntimeError(f"ChromaDB dimension error during upsert: {ide}") from ide
            except ChromaError as ce:
                logger.error(f"User '{self.user_id}': ChromaDB error storing to '{current_collection.name}': {ce}",
                             exc_info=True)
                raise RuntimeError(f"ChromaDB operation failed: {ce}") from ce
            except Exception as e:
                logger.error(f"User '{self.user_id}': Unexpected error storing to '{current_collection.name}': {e}",
                             exc_info=True)
                raise RuntimeError(f"Unexpected error during ChromaDB storage: {e}") from e
        return current_collection

    def vector_search(self, query: str, collection_name: Optional[str] = None, k: int = 10,
                      embedding_model_id_override: Optional[str] = None,
                      where_filter: Optional[Dict[str, Any]] = None,
                      include_fields: Optional[List[ChromaIncludeLiteral]] = None
                      ) -> List[Dict[str, Any]]:
        _collection_name_for_ops = collection_name or self.get_user_default_collection_name()

        query_embedding_model_id = self._resolve_embedding_model_id(_collection_name_for_ops, embedding_model_id_override)
        if not query_embedding_model_id:
            msg = "Cannot search: No embedding model ID resolved (override, collection metadata, or manager default)."
            logger.error(f"User '{self.user_id}': Collection '{_collection_name_for_ops}'. {msg}")
            raise ValueError(msg)

        effective_include_fields: List[ChromaIncludeLiteral] = include_fields or ["documents", "metadatas", "distances"]

        with self._lock:
            try:
                current_collection = self.get_or_create_collection(_collection_name_for_ops)
                logger.info(f"User '{self.user_id}': Vector search in '{current_collection.name}' for query: '{query[:50]}...' using model '{query_embedding_model_id}'.")

                query_embedding_np: np.ndarray = self.embedding_factory.embed_one(
                    text=query, model_id=query_embedding_model_id, as_list=False
                )  # type: ignore
                if query_embedding_np is None or query_embedding_np.ndim != 1 or query_embedding_np.size == 0:
                    raise ValueError(f"Failed to generate valid 1D embedding for query.")

                num_items_in_collection = current_collection.count()
                n_results_param = min(k, num_items_in_collection) if num_items_in_collection > 0 else k

                results: QueryResult = current_collection.query(
                    query_embeddings=[query_embedding_np.tolist()],
                    n_results=n_results_param,
                    where=self._clean_metadata(where_filter) if where_filter else None,
                    include=[f for f in effective_include_fields if f != "ids"]  # type: ignore
                )

                # --- CORRECTED RESULT PARSING LOGIC ---
                output: List[Dict[str, Any]] = []
                # `results['ids']` is a list containing one list of IDs (for our one query)
                ids_list = results.get("ids", [[]])[0]
                if not ids_list:
                    logger.info(f"User '{self.user_id}': No results for query in '{current_collection.name}'.")
                    return []

                num_results = len(ids_list)

                # Map Chroma's API field names to our desired output keys
                field_map = {"documents": "content", "metadatas": "metadata", "distances": "distance",
                             "embeddings": "embedding", "uris": "uri", "data": "data"}

                # Pre-process each field's data safely, handling missing fields.
                # `results.get(api_field)` returns `None` if the field was not included.
                # If it exists, we take the first element, which corresponds to our single query.
                processed_data = {
                    str(api_field): (results.get(api_field) or [[None] * num_results])[0]
                    for api_field in field_map.keys()
                }

                for i, item_id in enumerate(ids_list):
                    if item_id is None:
                        continue
                    current_item: Dict[str, Any] = {"id": item_id}
                    for api_field, output_key in field_map.items():
                        if api_field in effective_include_fields:
                            # Safely get the value for the current item
                            value = processed_data[api_field][i] if processed_data[api_field] and i < len(processed_data[api_field]) else None
                            if value is not None:
                                current_item[output_key] = value
                    output.append(current_item)

                logger.info(f"User '{self.user_id}': Found {len(output)} results for query in '{current_collection.name}'.")
                return output

            except (ValueError, RuntimeError) as e:
                logger.error(f"User '{self.user_id}': Error during vector search in '{_collection_name_for_ops}': {e}", exc_info=True)
                raise
            except (ChromaError) as e:
                if self._is_collection_not_found_error(e):
                    logger.warning(f"User '{self.user_id}': Collection '{_collection_name_for_ops}' not found during search. Returning empty.")
                    return []
                else:
                    logger.error(f"User '{self.user_id}': ChromaDB-related error during search in '{_collection_name_for_ops}': {e}", exc_info=True)
                    raise RuntimeError(f"ChromaDB operation failed during vector search: {e}") from e
            except Exception as e:
                logger.error(f"User '{self.user_id}': Unexpected error during search in '{_collection_name_for_ops}': {e}", exc_info=True)
                raise RuntimeError(f"Unexpected error during vector search: {e}") from e

    def reset_chroma_collection(self, collection_name: Optional[str] = None, new_metadata: Optional[Dict[str, Any]] = None):
        name_to_reset = collection_name or self.get_user_default_collection_name()
        cleaned_new_metadata = self._clean_metadata(new_metadata) if new_metadata else None
        with self._lock:
            try:
                logger.info(f"User '{self.user_id}': Attempting to reset collection: '{name_to_reset}'.")
                self.client.delete_collection(name=name_to_reset)
                logger.info(f"User '{self.user_id}': Deleted collection '{name_to_reset}' (or it didn't exist).")
            except (ChromaError, ValueError) as e:
                if self._is_collection_not_found_error(e):
                    logger.info(f"User '{self.user_id}': Collection '{name_to_reset}' did not exist for deletion.")
                else:  # Other ChromaError or ValueError
                    logger.error(
                        f"User '{self.user_id}': ChromaDB error deleting '{name_to_reset}' for reset: {e}. Will attempt creation.",
                        exc_info=True)
            except Exception as e:
                logger.error(
                    f"User '{self.user_id}': Unexpected error deleting '{name_to_reset}' for reset: {e}. Will attempt creation.",
                    exc_info=True)
            try:
                self.get_or_create_collection(name_to_reset, collection_metadata=cleaned_new_metadata)
                logger.info(f"User '{self.user_id}': Successfully (re)created collection: '{name_to_reset}'.")
            except Exception as ice:
                logger.error(f"User '{self.user_id}': Failed to create collection '{name_to_reset}' after reset: {ice}",
                             exc_info=True)
                raise RuntimeError(f"Failed to finalize reset for collection '{name_to_reset}': {ice}") from ice

    def delete_from_collection(self, ids: List[str], collection_name: Optional[str] = None):
        if not ids:
            logger.warning(f"User '{self.user_id}': No IDs provided for deletion. Skipping.")
            return

        _collection_name_for_ops = collection_name or self.get_user_default_collection_name()
        with self._lock:  # Use instance lock
            try:
                target_collection = self.client.get_collection(name=_collection_name_for_ops)
                target_collection.delete(ids=ids)
                logger.info(f"User '{self.user_id}': Attempted deletion of IDs {ids} from '{target_collection.name}'.")
            except (ChromaError, ValueError) as e:
                if self._is_collection_not_found_error(e):
                    logger.warning(f"User '{self.user_id}': Collection '{_collection_name_for_ops}' not found for deletion of IDs {ids}.")
                else: # Other ChromaError or ValueError
                    logger.error(f"User '{self.user_id}': ChromaDB error deleting from '{_collection_name_for_ops}': {e}", exc_info=True)
                    raise RuntimeError(f"ChromaDB deletion failed: PLACEHOLDER") from e # Original code had 'ce', assuming e
            except Exception as e:
                logger.error(f"User '{self.user_id}': Unexpected error deleting from '{_collection_name_for_ops}': {e}",
                             exc_info=True)
                raise RuntimeError(f"Unexpected error during deletion: {e}") from e

    def query_collection_with_precomputed_embeddings(
            self, query_embeddings: List[List[float]], n_results: int = 5,
            where_clause: Optional[Dict[str, Any]] = None, collection_name: Optional[str] = None,
            include_fields: Optional[List[ChromaIncludeLiteral]] = None
    ) -> QueryResult:
        _collection_name_for_ops = collection_name or self.get_user_default_collection_name()
        target_collection = self.get_or_create_collection(_collection_name_for_ops)
        effective_include_fields: List[ChromaIncludeLiteral] = include_fields or ["documents", "metadatas", "distances"]
        with self._lock:
            try:
                target_collection = self.get_or_create_collection(_collection_name_for_ops)
                if not query_embeddings or not all(isinstance(e, list) and e for e in query_embeddings):
                    raise ValueError("Query embeddings must be non-empty list of non-empty embedding vectors.")
                num_items = target_collection.count()
                n_results_param = min(n_results, num_items) if num_items > 0 else n_results
                if n_results_param == 0 and n_results > 0 : n_results_param = n_results
                return target_collection.query(
                    query_embeddings=query_embeddings, n_results=n_results_param,
                    where=self._clean_metadata(where_clause) if where_clause else None, include=effective_include_fields
                )
            except ValueError as ve:
                logger.error(f"User '{self.user_id}': Input validation error: {ve}", exc_info=True); raise
            except (ChromaError, ValueError) as e:
                if self._is_collection_not_found_error(e):
                    logger.warning(f"User '{self.user_id}': Collection '{_collection_name_for_ops}' not found for precomputed query.")
                    return QueryResult(ids=[[]], embeddings=None, documents=None, metadatas=None, distances=None, uris=None, data=None) # type: ignore
                else:
                    logger.error(f"User '{self.user_id}': ChromaDB error querying '{_collection_name_for_ops}': {e}", exc_info=True)
                    raise RuntimeError(f"ChromaDB query with precomputed embeddings failed: {e}") from e
            except Exception as e:
                logger.error(f"User '{self.user_id}': Unexpected error querying '{_collection_name_for_ops}': {e}", exc_info=True)
                raise RuntimeError(f"Unexpected error during query: {e}") from e

    def count_items_in_collection(self, collection_name: Optional[str] = None) -> int:
        _collection_name_for_ops = collection_name or self.get_user_default_collection_name()
        with self._lock:  # Use instance lock
            try:
                target_collection = self.client.get_collection(name=_collection_name_for_ops)
                return target_collection.count()
            except (ChromaError, ValueError) as e:
                if self._is_collection_not_found_error(e):
                    logger.info(f"User '{self.user_id}': Collection '{_collection_name_for_ops}' not found for count. Returning 0.")
                    return 0
                else: # Other ChromaError or ValueError
                    logger.error(f"User '{self.user_id}': Error counting items in '{_collection_name_for_ops}': {e}", exc_info=True)
                    raise RuntimeError(f"Failed to count items in collection: {e}") from e
            except Exception as e:
                logger.error(f"User '{self.user_id}': Error counting items in '{_collection_name_for_ops}': {e}",
                             exc_info=True)
                raise RuntimeError(f"Failed to count items in collection: {e}") from e

    def list_collections(self) -> Sequence[Collection]:
        """Lists all collections in the database."""
        with self._lock:  # Use instance lock
            try:
                return self.client.list_collections()
            except Exception as e:
                logger.error(f"User '{self.user_id}': Error listing collections: {e}", exc_info=True)
                raise RuntimeError(f"Failed to list collections: {e}") from e

    def delete_collection(self, collection_name: str):
        if not collection_name: raise ValueError("collection_name must be provided.")
        with self._lock:  # Use instance lock
            try:
                self.client.delete_collection(name=collection_name)
                logger.info(f"User '{self.user_id}': Successfully deleted collection '{collection_name}'.")
            except (ChromaError, ValueError) as e:
                if self._is_collection_not_found_error(e):
                     logger.warning(f"User '{self.user_id}': Collection '{collection_name}' not found for deletion.")
                else: # Other ChromaError or ValueError
                    logger.error(f"User '{self.user_id}': ChromaDB error deleting collection '{collection_name}': {e}", exc_info=True)
                    raise RuntimeError(f"ChromaDB failed to delete collection '{collection_name}': {e}") from e
            except Exception as e:
                logger.error(f"User '{self.user_id}': Unexpected error deleting collection '{collection_name}': {e}",
                             exc_info=True)
                raise RuntimeError(f"Unexpected error deleting collection '{collection_name}': {e}") from e

    def close(self):
        """Closes the ChromaDB client and the underlying EmbeddingFactory."""
        logger.info(f"User '{self.user_id}': Closing ChromaDBManager.")
        # EmbeddingFactory has its own __exit__ and close method for resource cleanup.
        if hasattr(self, 'embedding_factory') and self.embedding_factory:
            try:
                self.embedding_factory.close()
                logger.info(f"User '{self.user_id}': EmbeddingFactory closed.")
            except Exception as e:
                logger.error(f"User '{self.user_id}': Error closing EmbeddingFactory: {e}", exc_info=True)

        # ChromaDB PersistentClient doesn't have an explicit close method in the same way.
        # Resources are typically managed by its lifecycle (e.g., when the object is garbage collected).
        # If there were specific cleanup needed for self.client, it would go here.
        # For PersistentClient, ensuring data is flushed is usually implicit.
        logger.info(f"User '{self.user_id}': ChromaDB client resources will be released upon garbage collection.")

# Example usage would need to be updated with the new config structure for EmbeddingFactory.
# Example config for user_embedding_config:
# mock_config = {
#     "USER_DB_BASE_DIR": "./user_data_test_chroma",
#     "embedding_config": { # This part now directly matches EmbeddingConfigSchema
#         "default_model_id": "e5-small-v2",
#         "models": {
#             "e5-small-v2": {
#                 "provider": "huggingface",
#                 "model_name_or_path": "intfloat/e5-small-v2",
#                 "dimension": 384 # Crucial: Pre-configure dimensions
#             },
#             "openai-ada": {
#                 "provider": "openai",
#                 "model_name_or_path": "text-embedding-ada-002",
#                 "api_key": "YOUR_OPENAI_KEY_OR_SET_ENV_VAR",
#                 "dimension": 1536
#             }
#         },
#         "default_llm_for_contextualization": "mock-llm-for-context" # Can be any identifier analyze uses
#         # Optional EmbeddingFactory settings (max_cached, idle_seconds, allow_dynamic_hf)
#         # "max_cached": 3,
#         # "idle_seconds": 600,
#         # "allow_dynamic_hf": False
#     },
#     "prompts": {"situate_context_template": "Context for {chunk_content} in {doc_content}: ..."},
#     "chroma_client_settings": {"anonymized_telemetry": False}
# }

# if __name__ == "__main__":
#     # This simple test won't run chunking/LLM calls correctly without mocks
#     # It's more for seeing if ChromaDBManager initializes with the new EmbeddingFactory
#     import logging.config
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#     test_user = "test_user_chroma_ef"
#     try:
#         print(f"Initializing ChromaDBManager for {test_user}...")
#         manager = ChromaDBManager(user_id=test_user, user_embedding_config=mock_config)
#         print(f"Manager initialized. Default collection name: {manager.get_user_default_collection_name()}")

#         # Example: Count items (should be 0 or error if collection doesn't exist yet)
#         # count = manager.count_items_in_collection()
#         # print(f"Items in default collection: {count}")

#         # Example: Search (will fail if no model/collection and try to create embeddings)
#         # try:
#         #     results = manager.vector_search(query="hello world")
#         #     print(f"Search results: {results}")
#         # except Exception as e:
#         #     print(f"Search failed as expected (no data/model setup): {e}")

#     except Exception as e:
#         print(f"Error in main example: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         if 'manager' in locals() and manager:
#             manager.close()
#             print("ChromaDBManager closed.")
#         # Clean up test directory (optional)
#         # import shutil
#         # test_chroma_path = Path(mock_config["USER_DB_BASE_DIR"]) / test_user
#         # if test_chroma_path.exists():
#         #     shutil.rmtree(test_chroma_path)
#         #     print(f"Cleaned up test directory: {test_chroma_path}")

#
# End of Chroma_Lib.py
########################################################################################################################
