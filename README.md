# TranscripKa 📝

Aplicación local para estudio basado en transcripciones. Carga archivos TXT, los indexa con embeddings (Sentence Transformers + ChromaDB) y responde preguntas usando el contexto recuperado.

## Requisitos

- Python 3.9+
- pip

## Instalación

```bash
# 1. Clonar o copiar los archivos
# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar
streamlit run app.py
```

## Uso

1. Abrir http://localhost:8501
2. Subir archivos `.txt` con transcripciones
3. Click **"Procesar documentos"**
4. Escribir una pregunta y click **"Consultar"**
5. Ver respuesta, fuentes y contexto completo
6. Botón **"🗑️ Borrar y reconstruir índice"** en la barra lateral para reiniciar

## Stack

- [Streamlit](https://streamlit.io) — UI
- [ChromaDB](https://www.trychroma.com) — vector store persistente
- [Sentence Transformers](https://www.sbert.net) — embeddings locales (modelo all-MiniLM-L6-v2)
- Sin LLM externo — respuestas construidas con el contexto recuperado