# Práctica PLN mediante Deep Learning 26/27

Tarea elegida: **PAN 2026 – Multi-Author Writing Style Analysis** (detección de cambio de autor entre frases consecutivas).
Enunciado: `practica/Enunciado_práctica_NLP_2026-2027.pdf` (el nombre del fichero está en Unicode NFD: ábrelo con un glob `practica/Enunciado*.pdf`).
Entregable: **un único notebook** `practica/practica_nlp.ipynb`. **Fecha límite: 30/11/2026, 23:55.**
Checklist de entrega: `practica/CHECKLIST_ENTREGA.md`.

## Reglas del enunciado (obligatorias)

1. **Preparación:** cargar y preprocesar el dataset.
2. **Entrenamiento:** diseñar, **justificar** e implementar un modelo. Como mínimo **un fine-tuning de un Transformer**. El enfoque debe ser **original** (no solo replicar un baseline).
3. **Evaluación:** sobre validation/test; **comparar con modelos base y con propuestas publicadas** de PAN para la misma tarea (citando la fuente).
4. **Formato del notebook:**
   - Primera celda: integrantes de la pareja + **tabla resumen** de experimentos (resultado y conclusión de cada uno) + **metodología experimental**.
   - Todas las celdas **ejecutadas**, con salidas visibles.
   - Celdas markdown que expliquen cada paso.
   - **Conservar todos los experimentos, también los que no funcionaron**, explicando por qué se probó y qué se concluyó. Nunca borrar un experimento fallido: documentarlo.
5. **Entrega:** zip `PRÁCTICA_DL_NLP_26_27_GRUPO_XX.zip` (XX = nº de grupo en Moodle).

## Protocolo experimental (no romperlo)

- Tres niveles independientes: `easy`, `medium`, `hard`, cada uno con `train` / `validation` / `test` (10500 / 2250 / 2250 docs). Verificado: sin documentos duplicados ni solapamientos entre splits.
- Lectura: `open(..., newline="")` y `text.rstrip("\n").split("\n")`. **Nunca `splitlines()`** (rompe frases con ` `/`\x85`).
- **Métrica: macro F1** sobre todos los pares concatenados del split (como el evaluador oficial). Accuracy no sirve (desbalance: 30 % / 4 % / 21 % de cambios).
- Hiperparámetros, checkpoint y **umbral de decisión se eligen en validation**. **Test solo para el número final**: nunca para decidir nada.
- Todo resultado se registra con `log_result(exp, nivel, split, f1, notas)` → `artifacts/results.csv`. La tabla resumen de la celda 0 se rellena desde ese CSV (`show_results("test")`).
- IDs de experimento: `B0*` triviales, `B1*` estilometría, `B2*` embeddings, `T1*` fine-tuning estándar, `P1*` propuesta original (ablaciones como `P1-sin-X`).
- Semilla fija `SEED = 42`. Cachear en `artifacts/` lo caro (features, probabilidades, modelos) y respetar `RUN_TRAINING`.

## Estilo: lo más parecido a lo visto en la asignatura

Material de referencia en `/home/milu/NLP` (notebooks de clase). Seguir sus patrones:

- **Hugging Face `transformers`:** `AutoTokenizer`, `AutoModel`, `AutoModelForSequenceClassification`, `Trainer` + `TrainingArguments` + `compute_metrics` (Tarea 2.5), `datasets.Dataset` + `.map(batched=True)`, `DataCollatorWithPadding`.
- **Bucle PyTorch propio** con `torch.utils.data.Dataset`/`DataLoader` y cabeza sobre `AutoModel` cuando haga falta una arquitectura a medida (patrón de `transformers_multi_label_classification.ipynb` y `transformers_question_answering.ipynb`).
- **PEFT/LoRA** (`peft.LoraConfig`, `get_peft_model`) si se usa un modelo grande (patrón de `supporting_material_demo(1).ipynb`).
- **Baselines clásicos** con scikit-learn (`LogisticRegression`, `DummyClassifier`, `f1_score`, `confusion_matrix`), como en la Tarea 2.5.
- Análisis de tokenizadores (Tarea 2.4) para justificar la elección del modelo.
- Visualización con matplotlib/seaborn. Datos con pandas/numpy.
- No introducir frameworks ajenos al curso (Lightning, Keras, sentence-transformers como librería, etc.) salvo justificación explícita. Modelos preentrenados se cargan con `transformers`.
- Markdown y comentarios **en español**, frases claras, sin relleno. Código simple y legible; preferir funciones pequeñas reutilizables ya definidas en el notebook (`flatten`, `macro_f1`, `best_threshold`, `log_result`, `run_classic`, `pairs_dataset`, `predict_probs`).

## Restricciones de hardware

- GPU local: **RTX 3050 Ti Laptop, 4 GB VRAM** (bf16 disponible). El notebook asumía una 3070 de 8 GB: usar `T1_BATCH = 8` y `T1_GRAD_ACCUM = 2` (mismo batch efectivo 16).
- Modelos tipo `roberta-base` caben con `max_length=128`, bf16 y batch 8. Para modelos más grandes: LoRA, gradient checkpointing o menos longitud.
- **RAM: 15 GB en total, pero solo hay ~6 GB libres** (navegador, etc.). El kernel ya ocupa bastante con `data` y las features de B1/B2 en memoria. Ya hubo un OOM en B2 (*medium*): no duplicar matrices grandes. Construirlas bajo demanda, en float32/float16, con `del` + `gc.collect()` entre experimentos. Para T1/P1, liberar `b1`/`b2` si no se usan.
- Entrenar y predecir con todos los pares es caro (medium tiene ~279 k pares en validation y test). Estimar el tiempo antes de lanzar y submuestrear train si hace falta (documentándolo).

## Entorno

- `uv` con el `pyproject.toml` del repo (`package = false`, proyecto solo de notebooks). `uv sync` crea `.venv`.
- Kernel de Jupyter: **`pln (3.13)`** (apunta a `/home/milu/PLN/.venv`).
- El notebook se ejecuta **desde `practica/`** (rutas relativas `full_dataset/` y `artifacts/`).
- `practica/full_dataset/` y `practica/artifacts/` están en `.gitignore` (pesan ~2 GB). `full_dataset/medium/_validation_report/` es un informe de los organizadores sin utilidad: ignorarlo.

## Forma de trabajar

- Antes de un entrenamiento largo, probar con un subconjunto pequeño (smoke test) y estimar el tiempo.
- Cada experimento nuevo en el notebook: markdown con **hipótesis y diseño** → código → resultados → markdown de **conclusiones** (rellenar los "completar tras ejecutar").
- No sobrescribir resultados ni cachés de otros experimentos sin avisar.
- Al final: reiniciar el kernel y ejecutar todo en orden (con `RUN_TRAINING = False` para cargar cachés) para que los `execution_count` sean consecutivos.
