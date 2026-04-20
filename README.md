# Historical Ecuador Validator

Aplicación web en Streamlit para evaluar salidas generadas por IA mediante métricas automáticas reales. El foco principal del proyecto es comparar textos generados contra una fuente de referencia, usando medidas que capturan tanto coincidencia superficial como cercanía semántica.

Aunque la app también incluye un módulo de imágenes, la finalidad central del proyecto es servir como una herramienta práctica para revisar calidad textual en flujos de generación asistida por IA.

## Objetivo del proyecto

Este proyecto nace para responder una necesidad muy concreta: no evaluar textos de IA solo “a ojo”, sino con métricas reproducibles que permitan comparar varias salidas frente a una misma fuente.

En su estado actual, la app permite:

- ingresar una fuente textual de referencia
- ingresar tres textos generados por IA
- calcular métricas para cada texto contra la fuente
- visualizar los resultados de forma clara dentro de una interfaz web sencilla

Esto lo vuelve útil para tareas como:

- comparar tres respuestas de distintos modelos
- comparar tres versiones de un mismo prompt
- revisar si un texto generado conserva contenido clave de una fuente histórica
- medir si una reescritura mantiene significado aunque cambie la forma

## Qué evalúa exactamente

### Módulo principal: evaluación de texto

El flujo textual está pensado para comparar:

- una `Fuente`
- `Texto 1`
- `Texto 2`
- `Texto 3`

Cada uno de los tres textos se evalúa por separado contra la fuente utilizando:

- `BLEU`
- `BERTScore Precision`
- `BERTScore Recall`
- `BERTScore F1`

### Por qué estas métricas

`BLEU` mide superposición de n-gramas. Es útil cuando interesa verificar cuánto se parece un texto generado a la redacción o estructura superficial de la fuente.

`BERTScore` usa embeddings contextuales. Es útil cuando el texto generado cambia palabras o reformula frases, pero conserva significado.

La combinación de ambas métricas permite una lectura más realista:

- BLEU alto: el texto se parece mucho en forma
- BERTScore alto: el texto se parece mucho en contenido o sentido
- BLEU bajo + BERTScore alto: probablemente hubo parafraseo correcto
- BLEU alto + BERTScore bajo: puede haber coincidencia local sin buena coherencia global

## Análisis del proyecto

La arquitectura actual es deliberadamente simple: un solo archivo `app.py` concentra la interfaz y la lógica de cálculo, lo que facilita entender, ejecutar y extender el proyecto rápidamente.

Puntos fuertes del enfoque actual:

- interfaz accesible para usuarios no técnicos
- métricas reales y ampliamente usadas en NLP
- validaciones de entrada y manejo de errores con Streamlit
- compatibilidad CPU por defecto
- estructura suficientemente clara para evolucionar a módulos separados más adelante

Limitaciones importantes:

- BLEU no captura bien equivalencia semántica profunda por sí solo
- BERTScore depende de un modelo preentrenado y puede variar según idioma, dominio y estilo
- las métricas automáticas no reemplazan evaluación humana
- el módulo de imágenes es exploratorio, especialmente en FID con solo tres imágenes

En otras palabras, la app es muy útil como sistema de apoyo a la evaluación, no como juez absoluto.

## Módulo secundario: evaluación de imágenes

La aplicación también incluye un segundo módulo multimodal para experimentación con imágenes.

Actualmente calcula:

- `FID` en los cruces `1 vs (2,3)`, `2 vs (1,3)` y `3 vs (1,2)`
- `CLIPScore` de cada imagen contra un texto

Este módulo complementa el proyecto, pero no cambia el enfoque principal del repositorio: evaluar producción generada por IA con métricas objetivas.

## Tecnologías utilizadas

- Python 3.11+
- Streamlit
- sacrebleu
- bert-score
- torch
- torchvision
- torchmetrics
- Pillow

Dependencias complementarias relevantes:

- `transformers` para cargar modelos usados por BERTScore y CLIPScore
- `torch-fidelity` para soporte de FID dentro de `torchmetrics`

## Estructura del proyecto

```text
historical_evaluator/
├── app.py
├── requirements.txt
└── README.md
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Consideraciones de uso

- La primera ejecución puede tardar más porque algunas librerías descargan modelos.
- Si se trabaja solo en CPU, BERTScore y CLIPScore pueden demorar en entradas grandes.
- Para análisis serios de textos generados por IA, conviene combinar estas métricas con revisión humana experta.

## Proyección del proyecto

Este proyecto ya es funcional para comparar tres textos generados por IA contra una fuente. Las extensiones más naturales serían:

- exportar resultados a CSV o JSON
- agregar rankings automáticos entre textos
- incorporar ROUGE u otras métricas de resumen
- permitir lotes de evaluación
- separar lógica y UI en módulos más pequeños

## Autoría y propósito

Este repositorio está pensado como una base práctica para validación de contenido generado por IA, especialmente en contextos donde importa contrastar fidelidad semántica frente a una fuente original.

Si el caso de uso principal es histórico, educativo o documental, esta herramienta puede servir como primera capa de verificación cuantitativa antes de una revisión editorial o académica.
