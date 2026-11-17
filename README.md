# Práctica PLN mediante Deep Learning 26/27

PAN 2026 — Multi-Author Writing Style Analysis: detectar cambios de autor entre frases consecutivas.

- Notebook: `practica/practica_nlp.ipynb`
- Enunciado: `practica/Enunciado_práctica_NLP_2026-2027.pdf`
- Reglas y protocolo: `CLAUDE.md` · Checklist: `practica/CHECKLIST_ENTREGA.md`

## Entorno

```bash
uv sync
uv run python -m ipykernel install --user --name pln --display-name "pln (3.13)"
```

Abrir el notebook con el kernel `pln (3.13)` y ejecutarlo desde `practica/`.
El dataset va en `practica/full_dataset/{easy,medium,hard}/{train,validation,test}` (no está en el repo).
