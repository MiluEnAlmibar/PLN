# Checklist de entrega — PRÁCTICA_DL_NLP_26_27_GRUPO_XX.zip (fecha límite 30/11/2026, 23:55)

## Contenido del notebook (`practica_nlp.ipynb`)

### Celda 0
- [ ] Integrantes de la pareja (falta el segundo integrante).
- [ ] Tabla resumen con **todos** los experimentos: ID, descripción, macro F1 easy/medium/hard (test) y conclusión individual.
- [ ] Explicación de la metodología experimental.

### a) Preparación
- [x] Carga del dataset con verificación (`assert` de `len(changes) == n_frases - 1`).
- [x] Análisis exploratorio: tamaños, desbalance, autores, longitudes de frase y de tramo.
- [x] Representación en pares y utilidades de evaluación.

### b) Entrenamiento
- [x] B0 / B0r: baselines triviales.
- [x] B1-LR / B1-HGB: estilometría + similitudes TF-IDF, con ablación.
- [ ] B2: ejecución completa (se cortó en *medium*; faltan B1+B2 en medium y todo hard) y conclusiones.
- [ ] T1: fine-tuning del cross-encoder `roberta-base` (ajustar batch a la GPU de 4 GB), curvas y conclusiones.
- [ ] P1: propuesta **original** justificada, con ablaciones y análisis de errores.
- [ ] Experimentos fallidos conservados y explicados.

### c) Evaluación
- [ ] Macro F1 en test con umbral elegido en validation para todos los experimentos.
- [ ] Comparación con baselines (B0–B2, T1).
- [ ] Comparación con propuestas publicadas de PAN (overview y notebook papers), citando las fuentes e indicando diferencias de datos o de protocolo.
- [ ] Conclusiones finales.

## Antes de comprimir
- [ ] Reiniciar el kernel y ejecutar todo en orden (`RUN_TRAINING = False` con cachés) → `execution_count` consecutivos y sin errores.
- [ ] Ningún "TODO" ni "completar tras ejecutar" pendiente.
- [ ] Las salidas son legibles (no imprimir logs enormes; las barras de progreso no deben inundar el notebook).
- [ ] Zip con el nombre exacto `PRÁCTICA_DL_NLP_26_27_GRUPO_XX.zip` y el número de grupo real.
