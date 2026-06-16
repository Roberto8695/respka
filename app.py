"""
TranscripKa — estudio basado en transcripciones.

Carga archivos TXT, fragmenta, indexa con embeddings localmente
(Sentence Transformers + ChromaDB) y responde preguntas usando
solo el contexto recuperado.

Opcional: usa DeepSeek (vía API compatible OpenAI) para generar
respuestas razonadas a partir del contexto. Sin API key,
funciona en modo solo-contexto.

Uso:
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions

# ponytail: openai se importa solo si hay API key, así no se paga
# el import si no se usa. dotenv se carga silenciosamente.

from dotenv import load_dotenv

load_dotenv()

# ── configuración ──────────────────────────────────────────────

CHROMA_DIR = Path(tempfile.gettempdir()) / "transcripka_chroma"
CHUNK_SIZE = 500      # caracteres por fragmento
CHUNK_OVERLAP = 50    # solapamiento entre fragmentos
COLLECTION_NAME = "transcripciones"
MODEL_NAME = "all-MiniLM-L6-v2"  # ~80 MB, buen balance velocidad/calidad

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-chat"  # modelo más capaz de DeepSeek

# ── helpers ────────────────────────────────────────────────────


def _chunk_text(text: str) -> list[str]:
    """Divide texto en fragmentos de ~CHUNK_SIZE chars con solapamiento."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _load_docs_from_dir(dir_path: str) -> list[tuple[str, str]]:
    """Retorna lista de (filename, content) para cada .txt en dir_path."""
    docs: list[tuple[str, str]] = []
    for fname in sorted(os.listdir(dir_path)):
        if fname.lower().endswith(".txt"):
            path = os.path.join(dir_path, fname)
            with open(path, encoding="utf-8") as f:
                docs.append((fname, f.read()))
    return docs


# ponytail: ChromaDB ya maneja el batching y persistencia por defecto.
# Si en el futuro se necesitara un modelo más grande (p.ej. para otro
# idioma), cambiar MODEL_NAME. El upgrade es transparente.

@st.cache_resource
def _get_collection():
    """Obtiene (o crea) la colección ChromaDB persistente."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )


def _index_docs(docs: list[tuple[str, str]]):
    """Borra el índice y lo reconstruye con los documentos dados."""
    collection = _get_collection()
    # borrado masivo
    try:
        collection.delete(where={"source": {"$ne": ""}})
    except Exception:
        pass  # colección vacía o recién creada -> no hay nada que borrar

    ids: list[str] = []
    metadatas: list[dict] = []
    texts: list[str] = []

    for fname, content in docs:
        chunks = _chunk_text(content)
        for i, chunk in enumerate(chunks):
            ids.append(f"{fname}___{i:05d}")
            metadatas.append({"source": fname, "chunk": i})
            texts.append(chunk)

    if ids:
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
    return len(ids)


def _query(query_text: str, n_results: int = 5):
    """Recupera los fragmentos más relevantes."""
    collection = _get_collection()
    return collection.query(
        query_texts=[query_text],
        n_results=n_results,
    )


def _answer_with_deepseek(query: str, context: str) -> str | None:
    """Usa DeepSeek para responder basándose en el contexto."""
    # ponytail: import lazy para no pagar coste de import si no se usa
    from openai import OpenAI

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )

    system_prompt = (
        "Eres un asistente de estudio. Responde la pregunta del usuario "
        "basándote ÚNICAMENTE en el contexto proporcionado abajo. "
        "IMPORTANTE: No empieces con frases como 'Según el contexto', "
        "'Basado en el contexto', 'El contexto indica' ni repitas la pregunta. "
        "Ve directo al punto, máximo 2-3 oraciones. "
        "Si el contexto no contiene suficiente información, responde "
        "'No hay información suficiente en las transcripciones.' "
        "No inventes información. Responde en el mismo idioma de la pregunta."
    )

    user_prompt = f"""Contexto:
---inicio contexto---
{context}
---fin contexto---

Pregunta: {query}

Respuesta (basada solo en el contexto anterior):"""

    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # baja para ser preciso con el contexto
            max_tokens=1024,
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Error al consultar DeepSeek: {e}")
        return None


# ── Auto-carga de transcripciones ──────────────────────────────

# ponytail: Si existe la carpeta transcripciones/ y la colección está
# vacía, se indexan automáticamente al arrancar.
TRANSCRIPCIONES_DIR = Path("transcripciones")


def _auto_index_if_empty():
    """Indexa transcripciones/ si existe y la colección está vacía."""
    if not TRANSCRIPCIONES_DIR.is_dir():
        return
    collection = _get_collection()
    try:
        count = collection.count()
    except Exception:
        count = 0
    if count > 0:
        return  # ya hay datos

    docs = _load_docs_from_dir(str(TRANSCRIPCIONES_DIR))
    if not docs:
        return
    n_chunks = _index_docs(docs)
    st.session_state["docs_count"] = len(docs)
    st.session_state["chunks_count"] = n_chunks


# ── UI ─────────────────────────────────────────────────────────

st.set_page_config(page_title="TranscripKa", page_icon="📝")
st.title("📝 TranscripKa")
st.caption("Estudio basado en transcripciones — con DeepSeek opcional.")

# Auto-cargar transcripciones al arrancar (solo primera vez)
_auto_index_if_empty()

# ── Sidebar: carga ──

with st.sidebar:
    st.header("1. Cargar documentos")
    uploaded_files = st.file_uploader(
        "Selecciona archivos .txt",
        type="txt",
        accept_multiple_files=True,
    )

    if st.button("Procesar documentos", type="primary"):
        if not uploaded_files:
            st.error("Selecciona al menos un archivo .txt.")
            st.stop()

        # Guardar a disco temporal para _load_docs_from_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            for uf in uploaded_files:
                path = os.path.join(tmpdir, uf.name)
                with open(path, "wb") as f:
                    f.write(uf.getbuffer())

            docs = _load_docs_from_dir(tmpdir)

        if not docs:
            st.error("No se encontraron archivos .txt válidos.")
            st.stop()

        with st.spinner("Indexando documentos…"):
            n_chunks = _index_docs(docs)
        st.success(
            f"✅ {len(docs)} archivo(s) indexados "
            f"({n_chunks} fragmentos)."
        )
        # guardamos para mostrar en sidebar
        st.session_state["docs_count"] = len(docs)
        st.session_state["chunks_count"] = n_chunks

    if "docs_count" in st.session_state:
        st.markdown(
            f"**Archivos indexados:** {st.session_state.docs_count}"
        )
        st.markdown(
            f"**Fragmentos:** {st.session_state.chunks_count}"
        )

    if st.button("🗑️ Borrar y reconstruir índice"):
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            client.delete_collection(COLLECTION_NAME)
            st.session_state.clear()
            st.success("Índice borrado. Vuelve a cargar documentos.")
            st.rerun()
        except Exception:
            st.info("No había índice que borrar.")

    # ── Configuración DeepSeek ──
    st.divider()
    st.header("⚙️ DeepSeek")

    llm_available = bool(DEEPSEEK_API_KEY)
    use_llm = st.checkbox(
        "Usar DeepSeek para respuestas",
        value=llm_available,
        disabled=not llm_available,
        help=(
            "Requiere DEEPSEEK_API_KEY en el archivo .env. "
            "Sin ella, se usa el modo solo-contexto."
        ),
    )
    if not llm_available:
        st.info(
            "💡 Crea un archivo `.env` con:\n"
            "`DEEPSEEK_API_KEY=sk-tu-key-aqui`\n"
            "y reinicia la app para activar DeepSeek."
        )

# ── Main: consulta ──

st.header("2. Preguntar")

query = st.text_input(
    "¿Qué quieres saber?",
    placeholder="Ej: ¿Qué es la fotosíntesis?",
)

if st.button("Consultar", type="primary"):
    if not query.strip():
        st.warning("Escribe una pregunta.")
        st.stop()

    with st.spinner("Buscando en las transcripciones…"):
        result = _query(query)

    # ── Resultados ──
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    if not documents:
        st.info("No se encontraron fragmentos relevantes.")
        st.stop()

    # armar contexto para LLM (texto + fuente)
    context_parts: list[str] = []
    seen_sources: set[str] = set()
    for doc, meta in zip(documents, metadatas):
        src = meta.get("source", "desconocido")
        seen_sources.add(src)
        context_parts.append(f"[Fuente: {src}]\n{doc}")

    full_context = "\n\n".join(context_parts)

    # ── Respuesta ──
    st.subheader("Respuesta")

    if use_llm and llm_available:
        with st.spinner("Generando respuesta con DeepSeek…"):
            answer = _answer_with_deepseek(query, full_context)
        if answer:
            st.markdown(answer)
        else:
            st.warning(
                "DeepSeek falló. Mostrando contexto directamente."
            )
            st.markdown(full_context)
    else:
        # modo solo-contexto (fallback / sin key)
        for doc, meta, dist in zip(
            documents[:3], metadatas[:3], distances[:3]
        ):
            src = meta.get("source", "desconocido")
            st.markdown(f"> {doc}")
            st.caption(f"— _{src}_ (distancia: {dist:.3f})")

    # ── Fuentes ──
    st.subheader("Fuentes utilizadas")
    for src in sorted(seen_sources):
        st.markdown(f"- `{src}`")

    # ── Contexto completo ──
    with st.expander("Ver todo el contexto encontrado"):
        for doc, meta, dist in zip(documents, metadatas, distances):
            src = meta.get("source", "desconocido")
            chunk_n = meta.get("chunk", "?")
            st.markdown(
                f"**{src}** — fragmento {chunk_n} "
                f"(distancia: {dist:.4f})"
            )
            st.text(doc)
            st.divider()