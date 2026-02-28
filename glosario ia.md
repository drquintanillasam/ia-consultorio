# Glosario de Conceptos sobre IA
## Kit de Herramientas de IA para el Consultorio Médico

**Autor:** SAM  
**Fecha:** Marzo 2026  
**Tipo de Documento:** Glosario Técnico para Ponencia (45 min)  
**Público Objetivo:** Médicos con conocimiento del entorno clínico, enfoque en conceptos de IA

---

## Tabla de Contenidos

1. [Fundamentos de IA y Machine Learning](#1-fundamentos-de-ia-y-machine-learning)
2. [Modelos de Lenguaje y Arquitecturas de Generación](#2-modelos-de-lenguaje-y-arquitecturas-de-generación)
3. [Datos en Salud, Interoperabilidad y Privacidad](#3-datos-en-salud-interoperabilidad-y-privacidad)
4. [Implementación Clínica, Regulación y Ética](#4-implementación-clínica-regulación-y-ética)
5. [Explicabilidad, Validación y Evaluación Crítica](#5-explicabilidad-validación-y-evaluación-crítica)
6. [Referencias Bibliográficas](#6-referencias-bibliográficas)

---

## 1. FUNDAMENTOS DE IA Y MACHINE LEARNING

### 1.1 Conceptos Básicos

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **Inteligencia Artificial (IA)** | Sistemas de ordenador diseñados para realizar tareas que normalmente requieren inteligencia humana, como razonamiento, percepción y toma de decisiones. | Proporciona las bases teóricas para todas las aplicaciones de IA en salud, desde diagnóstico hasta administración hospitalaria. | [29] |
| **Aprendizaje Automático (ML)** | Conjunto de técnicas que permiten a los algoritmos de IA mejorar su rendimiento en una tarea a partir de datos, sin ser programados explícitamente. | Tecnología subyacente que impulsa la mayoría de las herramientas de IA clínica, como modelos de predicción de riesgo o diagnóstico. | [9] |
| **Aprendizaje Profundo (Deep Learning)** | Subcampo del ML que utiliza redes neuronales artificiales con múltiples capas para aprender representaciones complejas de datos. | Fundamental para análisis de imágenes médicas (radiología, patología, dermatología). | [9] |
| **Red Neuronal Artificial** | Modelo computacional inspirado en el cerebro humano, compuesto por nodos interconectados que procesan información. | Base arquitectónica de la mayoría de los modelos modernos de IA en salud. | [9] |

### 1.2 Componentes del Modelo

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **Modelo** | Construcción matemática que genera una inferencia o predicción a partir de datos de entrada. Resultado del entrenamiento de un algoritmo de ML. | Representa el núcleo de la tecnología de IA. Ejemplo: modelo que predice probabilidad de readmisión hospitalaria. | [9] |
| **Algoritmo** | Conjunto de reglas o instrucciones que un modelo de IA sigue para aprender de los datos y hacer predicciones. | Determina cómo el sistema procesa la información clínica para generar resultados. | [9] |
| **Parámetros** | Variables internas del modelo que se ajustan durante el entrenamiento para minimizar errores de predicción. | Definen el comportamiento específico del modelo una vez entrenado. | [9] |
| **Hiperparámetros** | Configuraciones externas del modelo que se establecen antes del entrenamiento (ej. tasa de aprendizaje, número de capas). | Controlan el proceso de aprendizaje y afectan directamente el rendimiento final del modelo. | [9] |

### 1.3 Conjuntos de Datos

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **Datos de Entrenamiento** | Conjunto de datos utilizado para entrenar un modelo de IA, enseñándole a identificar patrones y relaciones. | La calidad y representatividad determinan directamente la precisión y equidad del modelo clínico final. | [9] |
| **Datos de Validación** | Subconjunto de datos utilizado durante el entrenamiento para ajustar hiperparámetros y evitar sobreajuste. | Permite optimizar el modelo sin contaminar la evaluación final con datos de prueba. | [9] |
| **Datos de Prueba** | Conjunto de datos completamente independiente, utilizado para evaluar el rendimiento final del modelo. | Proporciona estimación objetiva de cómo se comportará el modelo en pacientes nuevos antes de uso clínico real. | [9] |
| **Datos de Desarrollo** | Datos utilizados en fases tempranas para prototipado y pruebas preliminares del modelo. | Permiten iteraciones rápidas antes de la validación formal. | [9] |

### 1.4 Problemas Comunes en Modelado

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **Sobreajuste (Overfitting)** | Fenómeno donde un modelo aprende el conjunto de entrenamiento con demasiada precisión, incluyendo ruido, perdiendo capacidad de generalización. | Modelo puede parecer excelente en pruebas internas pero fallar en práctica clínica con pacientes diversos. | [9] |
| **Subajuste (Underfitting)** | Situación donde un modelo es demasiado simple para capturar patrones subyacentes, resultando en mal rendimiento general. | Indica que el modelo no ha aprendido suficientemente y será poco fiable en cualquier aplicación clínica. | [9] |
| **Deriva del Modelo (Model Drift)** | Deterioro del rendimiento del modelo con el tiempo debido a cambios en los datos de entrada o en el entorno clínico. | Requiere monitoreo continuo y posible reentrenamiento para mantener precisión clínica. | [9] |
| **Sesgo de Selección** | Error sistemático que ocurre cuando la muestra de datos no es representativa de la población objetivo. | Puede llevar a modelos que funcionan bien en algunos grupos demográficos pero fallan en otros. | [60,62] |

### 1.5 Tipos de Aprendizaje

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **Aprendizaje Supervisado** | Tipo de ML donde el modelo se entrena con datos etiquetados (entrada + salida correcta conocida). | Común en diagnóstico asistido donde hay etiquetas claras (ej. tumor maligno/benigno). | [9] |
| **Aprendizaje No Supervisado** | Tipo de ML donde el modelo encuentra patrones en datos sin etiquetas previas. | Útil para descubrimiento de subtipos de enfermedades o agrupación de pacientes similares. | [9] |
| **Aprendizaje por Refuerzo** | Tipo de ML donde el modelo aprende mediante recompensas/castigos por sus acciones en un entorno. | Aplicaciones emergentes en optimización de tratamientos y dosificación de medicamentos. | [9] |
| **Aprendizaje Continuo** | Capacidad de un modelo de adaptar su rendimiento al incorporar nuevos datos a lo largo del tiempo. | Permite que herramientas de IA se actualicen automáticamente con nueva evidencia científica. | [9] |
| **Modelo Bloqueado** | Modelo cuya configuración y parámetros no cambian una vez entrenado. | Ofrece consistencia y previsibilidad, ventajoso para regulación, pero limita mejora continua. | [9] |

---

## 2. MODELOS DE LENGUAJE Y ARQUITECTURAS DE GENERACIÓN

### 2.1 Fundamentos de LLM

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **Grandes Modelos de Lenguaje (LLM)** | Tipos de IA que operan prediciendo y ensamblando secuencias de palabras basadas en entradas de texto. | Base de herramientas como ChatGPT, con aplicaciones en resumen de historias clínicas y educación médica. | [6] |
| **Modelo de Fundación (Foundation Model)** | Modelo de IA entrenado en datos masivos y diversos que puede adaptarse a múltiples tareas downstream. | Base sobre la cual se construyen aplicaciones médicas específicas mediante fine-tuning o RAG. | [12] |
| **Token** | Unidad básica de texto que un LLM procesa (puede ser una palabra, parte de palabra o carácter). | Determina el costo computacional y los límites de contexto de las consultas. | [6] |
| **Ventana de Contexto** | Cantidad máxima de tokens que un LLM puede procesar en una sola interacción. | Limita la cantidad de información clínica que se puede proporcionar en una consulta. | [6] |
| **Temperatura** | Parámetro que controla la aleatoriedad en la generación de texto de un LLM. | Valores bajos producen respuestas más deterministas; valores altos más creativas pero menos predecibles. | [6] |

### 2.2 Problemas y Limitaciones

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **Alucinación** | Generación de información falsa, incorrecta o no respaldada por evidencia por parte de un modelo de IA. | Riesgo significativo en práctica clínica; puede llevar a diagnósticos o tratamientos incorrectos si no se valida. | [1,97] |
| **Sesgo Algorítmico** | Tendencia sistemática de un modelo a producir resultados prejudiciales para ciertos grupos poblacionales. | Puede exacerbar disparidades en salud si los datos de entrenamiento no son diversos y representativos. | [60,62] |
| **Contaminación de Datos** | Cuando datos de prueba se filtran inadvertidamente en el conjunto de entrenamiento. | Infla artificialmente las métricas de rendimiento, dando falsa confianza en el modelo. | [9] |
| **Prompt Injection** | Técnica donde un usuario manipula la entrada para hacer que el LLM ignore sus instrucciones originales. | Riesgo de seguridad en sistemas que procesan información sensible de pacientes. | [2] |

### 2.3 Técnicas de Adaptación

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **Fine-Tuning (FT)** | Proceso de reentrenar un modelo preexistente con un conjunto de datos específico para adaptarlo a una tarea particular. | Útil para crear modelos especializados en áreas como oncología o dermatología, o para seguir protocolos clínicos. | [6,18] |
| **Full Fine-Tuning** | Todos los parámetros del modelo se actualizan durante el entrenamiento especializado. | Requiere más recursos computacionales pero puede lograr mejor adaptación al dominio médico. | [18] |
| **LoRA (Low-Rank Adaptation)** | Técnica de fine-tuning eficiente que actualiza solo un subconjunto pequeño de parámetros. | Reduce costos computacionales manteniendo buen rendimiento en tareas médicas específicas. | [18] |
| **Prompt Engineering** | Diseño estratégico de instrucciones de entrada para optimizar las respuestas de un LLM. | Habilidad esencial para médicos que usan LLMs; afecta directamente la calidad de las respuestas clínicas. | [6] |
| **Few-Shot Learning** | Técnica donde se proporcionan algunos ejemplos en el prompt para guiar la respuesta del modelo. | Permite adaptar el comportamiento del LLM sin necesidad de reentrenamiento. | [6] |
| **Zero-Shot Learning** | Capacidad del modelo de realizar tareas sin ejemplos previos, solo con instrucciones. | Útil para tareas médicas generales donde no hay ejemplos específicos disponibles. | [12] |

### 2.4 Arquitectura RAG

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **RAG (Generación Aumentada por Recuperación)** | Marco que mejora la precisión de un LLM conectándolo a una fuente de conocimiento externa para recuperar información relevante antes de generar respuesta. | Ayuda a mitigar alucinaciones y mantiene conocimiento actualizado con guías clínicas sin reentrenar. | [19,21] |
| **Arquitectura Naïve RAG** | Pipeline simple de dos etapas con recuperador fijo y LLM congelado. Estático y sin retroalimentación. | Ilustra enfoque básico de RAG, menos flexible que arquitecturas avanzadas con reranking. | [7] |
| **Modular RAG** | Descompone RAG en componentes independientes y reemplazables (recuperador y generador separados). | Aumenta flexibilidad y facilidad de mantenimiento, permitiendo optimizar cada componente. | [7] |
| **Embedding** | Representación vectorial numérica de texto que captura significado semántico para búsqueda. | Permite búsqueda semántica en bases de conocimiento médicas, no solo por palabras clave. | [7] |
| **Base de Datos Vectorial** | Sistema de almacenamiento optimizado para búsquedas de similitud en embeddings. | Almacena guías clínicas, literatura médica y protocolos para recuperación rápida por RAG. | [7] |
| **Reranking** | Proceso de reordenar resultados de recuperación por relevancia antes de enviarlos al LLM. | Mejora la calidad de la información contextual proporcionada al modelo para generación. | [7] |

### 2.5 IA Médica Avanzada

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **IA Médica Generalista (GMAI)** | Paradigma donde modelos pueden realizar amplia gama de tareas médicas usando pocos datos, interpretando múltiples modalidades. | Representa futuro de IA clínica con modelos capaces de razonamiento avanzado y flexibilidad. | [12] |
| **Modelo Multimodal** | Modelo de IA que puede procesar y relacionar diferentes tipos de datos (texto, imagen, audio, vídeo). | Permite integración de radiografías, notas clínicas y datos de laboratorio en un solo análisis. | [12] |
| **Chain-of-Thought (CoT)** | Técnica donde el modelo muestra su razonamiento paso a paso antes de dar una respuesta final. | Aumenta transparencia y permite al médico verificar la lógica detrás de recomendaciones. | [10] |
| **Agente de IA** | Sistema de IA que puede tomar acciones autónomas en un entorno para lograr objetivos específicos. | Futuras aplicaciones en gestión de citas, seguimiento de pacientes y coordinación de cuidados. | [12] |

---

## 3. DATOS EN SALUD, INTEROPERABILIDAD Y PRIVACIDAD

### 3.1 Tipos de Datos Clínicos

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **Datos Estructurados** | Información clínica organizada en formatos predefinidos y tabulares (resultados de laboratorio, mediciones vitales, códigos). | Fáciles de procesar por máquinas; base de muchas herramientas de análisis predictivo en salud. | [137,142] |
| **Datos No Estructurados** | Información clínica sin formato tabular (notas clínicas, informes de radiología, registros de conversaciones). | Constituyen mayor parte de información del paciente; análisis por IA clave para extraer conocimiento profundo. | [79,136] |
| **Datos Semiestructurados** | Datos con alguna organización pero sin esquema rígido (XML, JSON, formularios parcialmente completados). | Comunes en intercambio de información entre sistemas de salud. | [137] |
| **Datos Longitudinales** | Datos recopilados del mismo paciente a lo largo del tiempo. | Esenciales para modelos predictivos de progresión de enfermedades y respuesta a tratamientos. | [9] |
| **Datos en Tiempo Real** | Información generada y disponible inmediatamente (monitoreo continuo, dispositivos wearables). | Habilita intervenciones tempranas y medicina preventiva basada en alertas automáticas. | [9] |

### 3.2 Interoperabilidad

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **Interoperabilidad** | Capacidad de aplicaciones de software para comunicarse, intercambiar datos y usar información compartida. | Esencial para construir sistemas de IA que accedan a datos de múltiples HCE de manera consistente. | [114,152] |
| **HL7 FHIR** | Estándar moderno para intercambio de información de salud mediante API RESTful y recursos definidos. | Estándar de facto para lograr interoperabilidad en sector salud; permite acceso escalable a datos. | [152] |
| **API (Interface de Programación)** | Conjunto de definiciones y protocolos para construir e integrar software. | Permite que sistemas de IA se conecten de forma segura y estandarizada a HCE y otras fuentes. | [114] |
| **DICOM** | Estándar para almacenamiento y transmisión de imágenes médicas. | Fundamental para sistemas de IA en radiología, patología y otras especialidades con imágenes. | [114] |
| **SNOMED CT** | Terminología clínica estandarizada para codificación de conceptos médicos. | Facilita el procesamiento automático de diagnósticos y procedimientos por sistemas de IA. | [114] |
| **LOINC** | Estándar para identificación de pruebas de laboratorio y observaciones clínicas. | Permite interoperabilidad semántica de resultados de laboratorio entre diferentes sistemas. | [114] |

### 3.3 Privacidad y Seguridad

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **PHI (Información de Salud Protegida)** | Cualquier información de salud relacionada con un individuo identificable, protegida bajo HIPAA. | Manipulación indebida es ilegal y éticamente inaceptable. Gestión es piedra angular de seguridad de IA médica. | [80,103] |
| **HIPAA** | Ley estadounidense que establece estándares para protección de información de salud. | Marco regulatorio clave para cualquier sistema de IA que procese datos de pacientes en EE.UU. | [45] |
| **GDPR** | Reglamento General de Protección de Datos de la Unión Europea. | Aplica a sistemas de IA que procesan datos de ciudadanos europeos; requiere consentimiento explícito. | [80] |
| **Desidentificación** | Proceso de identificar y eliminar todos los identificadores de PHI de un conjunto de datos. | Requisito previo para compartir y utilizar datos clínicos a gran escala para entrenamiento de IA. | [134,135] |
| **Anonimización** | Técnica que elimina permanentemente la posibilidad de identificar a un individuo de los datos. | Datos verdaderamente anonimizados quedan fuera del alcance de regulaciones como GDPR. | [158] |
| **Pseudonimización** | Reemplazo de identificadores directos con códigos; reversible con clave adecuada. | Sigue estando sujeta a regulaciones de protección de datos; requiere salvaguardas adicionales. | [158] |
| **Cifrado (AES-256)** | Proceso de convertir datos en código para protegerlos de acceso no autorizado. | Medida fundamental para proteger PHI tanto en tránsito como en reposo. | [43,45] |
| **BAA (Business Associate Agreement)** | Contrato legal que establece responsabilidades de un tercero para proteger PHI. | Hospitales y proveedores de IA deben firmar BAAs para garantizar cumplimiento de HIPAA. | [124] |

### 3.4 Técnicas de Privacidad Avanzada

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **Diferencial Privado** | Metodología que añade ruido controlado a datos o resultados para garantizar que participación individual no pueda inferirse. | Proporciona garantía matemática de privacidad en análisis de datos poblacionales. | [158] |
| **Aprendizaje Federado** | Técnica donde modelos se entrenan de forma distribuida en múltiples instituciones sin compartir datos crudos. | Permite colaboración multicéntrica manteniendo datos en cada institución; preserva privacidad. | [9] |
| **Cómputo en Entorno Confidencial** | Ejecución de procesamiento de datos en entornos hardware seguros y aislados. | Protege datos incluso durante el procesamiento por sistemas de IA. | [45] |
| **Homomórfico** | Tipo de cifrado que permite realizar cálculos sobre datos cifrados sin descifrarlos. | Emergente para análisis de datos sensibles sin exponer información del paciente. | [45] |

---

## 4. IMPLEMENTACIÓN CLÍNICA, REGULACIÓN Y ÉTICA

### 4.1 Implementación

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **Implementación Clínica** | Proceso de integrar una nueva tecnología de IA en el flujo de trabajo diario de consultorio u hospital. | Determina si IA realmente mejora eficiencia o se convierte en distracción. Infraestructura y capacitación cruciales. | [56,87] |
| **Integración HCE** | Conexión de herramientas de IA con sistemas de Historia Clínica Electrónica existentes. | Esencial para adopción exitosa; evita duplicación de trabajo y flujos paralelos. | [70,91] |
| **Asistente Ambiental** | Tecnología de IA que transcribe conversación médico-paciente para generar borradores de notas clínicas. | Potencial para reducir agotamiento del personal; se necesitan más estudios rigurosos de impacto a largo plazo. | [89,156] |
| **Flujo de Trabajo Clínico** | Secuencia de tareas y procesos que conforman la atención al paciente. | La IA debe integrarse sin interrumpir ni complicar el flujo existente para ser adoptada. | [87] |
| **Adopción Tecnológica** | Proceso por el cual usuarios aceptan y utilizan regularmente una nueva tecnología. | Depende de utilidad percibida, facilidad de uso, y apoyo institucional. | [56] |

### 4.2 Regulación

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **FDA (Food and Drug Administration)** | Agencia reguladora estadounidense que aprueba dispositivos médicos, incluidos los basados en IA. | Más de 1000 dispositivos con IA/ML aprobados desde 2014; marco regulatorio de referencia mundial. | [14,125] |
| **SaMD (Software como Dispositivo Médico)** | Software destinado a ser utilizado para fines médicos sin ser parte de un dispositivo médico hardware. | Categoría regulatoria bajo la cual caen la mayoría de las herramientas de IA clínica. | [50] |
| **MDR 2017/745** | Regulación de la UE que establece requisitos para seguridad y rendimiento de dispositivos médicos. | Cumplimiento es requisito legal para vender dispositivo médico en mercado único europeo. | [82,126] |
| **Acta de IA de la UE** | Legislación que clasifica sistemas de IA por niveles de riesgo y establece requisitos regulatorios. | Define marco de gobernanza; impone obligaciones estrictas a desarrolladores de IA de alto riesgo. | [107,108] |
| **Alto Riesgo (IA)** | Sistemas de IA utilizados como componente de seguridad de dispositivo médico o como producto médico. | Sujetos a documentación técnica detallada, evaluación de impacto y altos estándares de rendimiento. | [108] |
| **CE Mark** | Marcado de conformidad que indica cumplimiento con requisitos esenciales de seguridad de la UE. | Requisito indispensable para poner dispositivo médico en mercado europeo. | [123] |
| **510(k)** | Vía de autorización de la FDA para dispositivos que demuestran equivalencia sustancial con dispositivo ya comercializado. | Ruta común para aprobación de herramientas de IA médica en Estados Unidos. | [14] |
| **De Novo** | Vía de autorización de la FDA para dispositivos de riesgo bajo-moderado sin predicado existente. | Utilizada para tecnologías de IA innovadoras sin precedentes regulatorios. | [14] |
| **PCCP (Plan de Control de Cambios Predefinidos)** | Documento presentado a FDA detallando cambios preaprobados que fabricante realizará en algoritmo de IA/ML. | Permite que dispositivos de IA se actualicen y mejoren continuamente de forma segura sin revisiones constantes. | [49,149,170] |

### 4.3 Ética y Gobernanza

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **Gobernanza de IA** | Políticas, procedimientos y prácticas para asegurar que sistemas de IA se desarrollen y utilicen de manera segura y ética. | Fundamental para navegar riesgos de IA (sesgo, transparencia) y garantizar que beneficios superen riesgos. | [58] |
| **Responsabilidad** | Principio ético que se refiere a capacidad de asignar causa y atribuir culpa por acciones y resultados de IA. | En caso de error, determinar si responsabilidad recae en desarrollador, hospital o médico es desafío complejo. | [11] |
| **Transparencia** | Principio que requiere que sistemas de IA sean abiertos sobre su funcionamiento, limitaciones y fuentes de datos. | Esencial para construir confianza del profesional sanitario y del paciente. | [10] |
| **Equidad** | Principio que requiere que sistemas de IA no discriminen ni exacerben disparidades en salud existentes. | Requiere datos de entrenamiento diversos y evaluación continua de rendimiento por subgrupos poblacionales. | [60,62] |
| **Consentimiento Informado** | Proceso por el cual pacientes autorizan el uso de sus datos o la aplicación de IA en su cuidado. | Aspecto ético y legal crítico; pacientes deben comprender cómo se usa IA en su atención. | [54] |
| **Supervisión Humana** | Requisito de que decisiones clínicas finales permanezcan bajo control de profesionales sanitarios. | La IA es herramienta de apoyo; responsabilidad última siempre recae en el médico. | [31,85] |

---

## 5. EXPLICABILIDAD, VALIDACIÓN Y EVALUACIÓN CRÍTICA

### 5.1 Explicabilidad (XAI)

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **IA Explicable (XAI)** | Conjunto de métodos para explicar decisiones de modelo de IA, aumentando transparencia y confianza. | Fundamental para que médicos comprendan el "porqué" detrás de recomendación de IA antes de actuar. | [10] |
| **Caja Negra** | Modelo cuyo proceso interno de toma de decisiones es opaco o difícil de interpretar para humanos. | Limita adopción clínica; los médicos necesitan entender el razonamiento para confiar en recomendaciones. | [10] |
| **Caja Blanca** | Modelo cuyo funcionamiento interno es transparente e interpretable por diseño. | Preferible en contextos clínicos críticos, aunque puede tener menor rendimiento predictivo. | [10] |
| **Explicación Local** | Explicación centrada en por qué modelo tomó decisión específica para una instancia única. | Más útil para decisión clínica individual; responde "¿por qué para ESTE paciente?". | [10] |
| **Explicación Global** | Explicación que describe comportamiento general del modelo en toda la gama de datos de entrada. | Ayuda a validar modelo en su conjunto; más útil para desarrolladores y reguladores. | [10] |
| **Método Ante-hoc** | Técnicas de explicabilidad incorporadas en el diseño del modelo desde el inicio. | Modelos interpretables por diseño (árboles de decisión, regresiones); pueden tener menor rendimiento. | [10] |
| **Método Post-hoc** | Técnicas aplicadas a modelo ya entrenado para generar explicaciones a posteriori. | Más comunes en práctica clínica; permiten explicar modelos complejos existentes. | [10] |

### 5.2 Técnicas de XAI

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **LIME** | Técnica post-hoc que explica predicciones aproximando modelo localmente con modelo lineal interpretable. | Proporciona lista de características más influyentes para decisión específica; ayuda verificar lógica de sugerencia. | [10] |
| **SHAP** | Técnica post-hoc basada en teoría de juegos que asigna valor de contribución a cada característica para predicción. | Ofrece medición cuantitativa de cómo cada dato de entrada afectó salida; justificación más robusta que LIME. | [10] |
| **Mapas de Activación** | Visualizaciones que muestran qué regiones de una imagen contribuyeron más a la predicción del modelo. | Esencial en radiología y patología; permite al médico verificar si modelo mira áreas correctas. | [10] |
| **Contrafactuales** | Explicaciones que muestran qué cambios mínimos en la entrada producirían diferente salida del modelo. | Útil para entender umbrales de decisión y qué factores podrían cambiar una recomendación. | [10] |

### 5.3 Validación y Evaluación

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **Calibración** | Proceso de ajustar probabilidades predichas por modelo para que coincidan con frecuencias reales de eventos observados. | Modelo bien calibrado proporciona estimaciones de riesgo realistas; crucial para decisiones basadas en probabilidades. | [9] |
| **Validación Externa** | Evaluación del modelo en poblaciones y entornos diferentes a los de entrenamiento. | Esencial para demostrar generalizabilidad; muchos modelos fallan en validación externa. | [42] |
| **Validación Clínica Rigurosa** | Evaluación de dispositivo de IA en entorno clínico real para demostrar seguridad, eficacia y beneficio para pacientes. | Muchas herramientas carecen de esta validación; clínicos deben exigir evidencia de estudios clínicos sólidos. | [42] |
| **Ensayo Clínico Aleatorizado (ECA)** | Estudio donde participantes son asignados aleatoriamente a grupo de intervención (IA) o control. | Estándar de oro para demostrar beneficio clínico real de herramientas de IA. | [89] |
| **Métricas de Rendimiento** | Medidas cuantitativas de precisión del modelo (sensibilidad, especificidad, AUC, F1-score, etc.). | Clínicos deben entender limitaciones de cada métrica; alta precisión no siempre significa utilidad clínica. | [9] |
| **Curva ROC/AUC** | Representación gráfica del rendimiento de modelo de clasificación; AUC mide área bajo la curva. | Métrica común para evaluar capacidad discriminatoria; AUC >0.8 generalmente considerado bueno. | [9] |

### 5.4 Evaluación Crítica para Clínicos

| Término | Definición | Relevancia Clínica | Fuente |
|---------|------------|-------------------|--------|
| **Competencia del Experto** | Nivel de habilidad donde profesional puede evaluar críticamente herramientas de IA sin ser experto técnico. | Nivel deseable para mayoría de clínicos; permite participación activa en implementación segura de IA. | [11] |
| **Alfabetización en IA** | Conjunto de conocimientos y habilidades para comprender, usar y evaluar críticamente tecnologías de IA. | Componente cada vez más importante de formación médica; programas en universidades de todo el mundo. | [11] |
| **Checklist de Verificación** | Herramientas diseñadas para guiar a clínicos en evaluación crítica de investigación y aplicaciones de IA. | Proporcionan marco estructurado para evaluar objetivamente calidad, fiabilidad y potencial sesgo de herramienta. | [143,144] |
| **TRIPOD-AI** | Guía de reporte para estudios de predicción que utilizan IA. | Estándar para evaluar calidad metodológica de estudios de IA en medicina. | [143] |
| **CONSORT-AI** | Extensión de CONSORT para ensayos clínicos que evalúan intervenciones con IA. | Ayuda a clínicos a evaluar rigor de ensayos clínicos de herramientas de IA. | [144] |
| **Evidencia de Nivel 1** | Evidencia proveniente de múltiples ECA o meta-análisis de alta calidad. | Nivel más alto de evidencia; pocas herramientas de IA han alcanzado este nivel actualmente. | [42] |

---

## 6. REFERENCIAS BIBLIOGRÁFICAS

### Fuentes Primarias

| ID | Referencia |
|----|------------|
| [1] | Bang Y. et al. (2023). A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity. arXiv. |
| [2] | Liu Y. et al. (2023). Prompt Injection Attacks on Large Language Models. IEEE Security & Privacy. |
| [6] | OpenAI (2023). GPT-4 Technical Report. OpenAI Research. |
| [7] | Gao Y. et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997. |
| [8] | Lewis P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS. |
| [9] | Topol E.J. (2019). High-Performance Medicine: The Convergence of Human and Artificial Intelligence. Nature Medicine, 25:44-56. |
| [10] | Arrieta A.B. et al. (2020). Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges. Information Fusion, 58:82-115. |
| [11] | Char D.S. et al. (2018). Implementing Machine Learning in Health Care—Addressing Ethical Challenges. NEJM, 378:981-983. |
| [12] | Moor M. et al. (2023). Foundation Models for Generalist Medical Artificial Intelligence. Nature, 616:259-265. |
| [13] | FDA (2023). Artificial Intelligence and Machine Learning in Software as a Medical Device. FDA Guidance. |
| [14] | FDA (2021). Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) Action Plan. |
| [18] | Hu E.J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR. |
| [19] | RAG Survey (2023). Retrieval-Augmented Generation: A Comprehensive Review. arXiv. |
| [21] | Guu K. et al. (2020). Retrieval Augmented Language Model Pre-Training. ICML. |
| [22] | Zhang Y. et al. (2023). Siren's Song in the AI Ocean: A Survey on Hallucination in LLMs. arXiv. |
| [27] | Davenport T. & Kalakota R. (2019). The Potential for Artificial Intelligence in Healthcare. Future Healthcare Journal, 6(2):94-98. |
| [28] | Reddy S. et al. (2020). Artificial Intelligence in Healthcare: Past, Present and Future. BMJ Innovations. |
| [29] | Russell S. & Norvig P. (2020). Artificial Intelligence: A Modern Approach. 4th Edition. Pearson. |
| [31] | Obermeyer Z. et al. (2019). Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations. Science, 366:447-453. |
| [32] | Rajkomar A. et al. (2019). Machine Learning in Medicine. NEJM, 380:1347-1358. |
| [34] | Samek W. et al. (2017). Explainable Artificial Intelligence: Understanding, Visualizing and Interpreting Deep Learning Models. ITU Journal. |
| [39] | FDA (2019). Proposed Regulatory Framework for Modifications to Artificial Intelligence/Machine Learning-Based Software as a Medical Device. |
| [42] | Vasey B. et al. (2021). Reporting Guideline for the Early-Stage Clinical Evaluation of Decision Support Systems Driven by Artificial Intelligence. Nature Medicine. |
| [43] | NIST (2020). Recommendation for Key Management. NIST Special Publication 800-57. |
| [45] | HIPAA Journal (2023). HIPAA Compliance for AI Systems. |
| [49] | FDA (2023). Predetermined Change Control Plan for AI/ML-Enabled Device Software Functions. Draft Guidance. |
| [50] | IMDRF (2019). Software as a Medical Device: Possible Framework for Risk Categorization. |
| [54] | Gerke S. et al. (2020). The Ethical and Legal Implications of Using AI in Healthcare. Cambridge Quarterly of Healthcare Ethics. |
| [56] | Greenhalgh T. et al. (2017). Beyond Adoption: A New Framework for Theorizing and Evaluating Nonadoption, Abandonment, and Challenges to Scale-Up. Journal of Medical Internet Research. |
| [58] | WHO (2021). Ethics and Governance of Artificial Intelligence for Health. WHO Guidance. |
| [60] | Pierson E. et al. (2021). An Algorithmic Approach to Reducing Unexplained Pain Disparities. Nature Medicine. |
| [61] | Norman D. (2013). The Design of Everyday Things. Revised Edition. Basic Books. |
| [62] | Seyyed-Kalantari L. et al. (2021). Underdiagnosis Bias of Artificial Intelligence Algorithms Applied to Chest Radiographs. Nature Medicine. |
| [63] | Lin S.C. et al. (2021). Association of Electronic Health Record Design and Use Factors with Clinician Stress and Burnout. JAMA Network Open. |
| [65] | Bates D.W. et al. (2014). Big Data in Health Care: Using Analytics to Identify and Manage High-Risk and High-Cost Patients. Health Affairs. |
| [67] | Hiernaux M. et al. (2023). Large Language Models in Dermatology: Opportunities and Challenges. JAMA Dermatology. |
| [69] | Thirunavukarasu A.J. et al. (2023). Large Language Models in Medicine. Nature Medicine, 29:1930-1940. |
| [70] | Kvedar J. et al. (2020). Artificial Intelligence in Healthcare: Transforming the Practice of Medicine. The Health IT Book. |
| [79] | Nadkarni P.M. et al. (2011). Natural Language Processing: An Introduction. JAMIA, 18(5):544-551. |
| [80] | HHS (2023). HIPAA Privacy Rule Summary. U.S. Department of Health and Human Services. |
| [82] | European Commission (2017). Regulation (EU) 2017/745 on Medical Devices. |
| [84] | Bodenheimer T. & Sinsky C. (2014). From Triple to Quadruple Aim: Care of the Patient Requires Care of the Provider. Annals of Family Medicine. |
| [85] | Topol E.J. (2019). Deep Medicine: How Artificial Intelligence Can Make Healthcare Human Again. Basic Books. |
| [86] | Sallam M. (2023). ChatGPT Utility in Healthcare Education, Research, and Practice: Systematic Review. Healthcare, 11(6):887. |
| [87] | Matheny M.E. et al. (2019). Bringing AI to BI: The Future of Artificial Intelligence in Health Care. NEJM Catalyst. |
| [89] | Yang J. et al. (2023). Ambient Artificial Intelligence Scribes to Alleviate the Burden of Clinical Documentation. NEJM AI. |
| [91] | Mandl K.D. et al. (2019). Interoperability: The Path to Realizing the Promise of Digital Health. NEJM. |
| [97] | Ji Z. et al. (2023). Survey of Hallucination in Natural Language Generation. ACM Computing Surveys. |
| [103] | OCR (2023). Protected Health Information. HHS Office for Civil Rights. |
| [105] | WHO (2023). Regulatory Considerations on Artificial Intelligence for Health. |
| [106] | FUTURE-AI Consortium (2023). Consensus Guidelines for Trustworthy AI in Healthcare. |
| [107] | European Parliament (2024). Artificial Intelligence Act. |
| [108] | European Commission (2021). Proposal for a Regulation on Artificial Intelligence. |
| [110] | Jiang F. et al. (2017). Artificial Intelligence in Healthcare: Past, Present and Future. Stroke and Vascular Neurology. |
| [114] | HL7 International (2023). FHIR Release 4 Specification. |
| [117] | NIST (2020). Security and Privacy Controls for Information Systems. NIST SP 800-53. |
| [123] | European Commission (2023). CE Marking for Medical Devices. |
| [124] | HHS (2023). Business Associate Contracts. HIPAA Guidance. |
| [125] | FDA (2023). AI/ML-Enabled Medical Devices Database. |
| [126] | European Commission (2023). Medical Device Regulation Implementation Guide. |
| [129] | BSI (2023). MDR and AI: Compliance Requirements. |
| [133] | Hripcsak G. & Albers D.J. (2013). Next-Generation Phenotyping of Electronic Health Records. JAMIA. |
| [134] | El Emam K. (2013). Guide to the De-Identification of Personal Health Information. CRC Press. |
| [135] | HHS (2012). HIPAA De-Identification Guidance. |
| [136] | Meystre S.M. et al. (2014). Extracting Data from Electronic Medical Records: Validation of a NLP System. JAMIA. |
| [137] | Esteva A. et al. (2019). A Guide to Deep Learning in Healthcare. Nature Medicine. |
| [142] | Rieke N. et al. (2020). The Future of Digital Health with Federated Learning. NPJ Digital Medicine. |
| [143] | Collins G.S. et al. (2023). TRIPOD+AI Statement: Updated Guidance for Reporting Prediction Models Developed Using AI. BMJ. |
| [144] | Liu X. et al. (2020). CONSORT-AI Extension for Clinical Trials Evaluating AI Interventions. Nature Medicine. |
| [147] | Sarker A. et al. (2021). Natural Language Processing for Electronic Health Records. Annual Review of Biomedical Data Science. |
| [148] | HIPAA Journal (2023). BAA Requirements for AI Vendors. |
| [149] | FDA (2024). PCCP Approval Database. |
| [151] | Chen I.Y. et al. (2023). Automated De-identification of Clinical Notes Using LLMs. JAMIA Open. |
| [152] | HL7 (2023). FHIR for AI/ML Integration. |
| [153] | Miotto R. et al. (2018). Deep Patient: An Unsupervised Representation of EHR Data. Scientific Reports. |
| [154] | European Commission (2024). AI Act Implementation Timeline. |
| [155] | FDA (2023). Continuous Learning in AI/ML Devices. |
| [156] | Abridge AI (2023). Ambient Clinical Intelligence Platform. |
| [158] | Dwork C. (2006). Differential Privacy. ICALP. |
| [159] | FDA (2024). PCCP Technical Considerations. |
| [164] | NEJM AI (2024). Ambient AI Documentation: Early Results. |
| [170] | FDA (2024). PCCP Final Guidance. |

---

## APÉNDICE: ACRÓNIMOS Y ABREVIATURAS

| Acrónimo | Significado Completo |
|----------|---------------------|
| AI/ML | Artificial Intelligence / Machine Learning |
| AUC | Area Under the Curve |
| BAA | Business Associate Agreement |
| CE | Conformité Européenne |
| CoT | Chain-of-Thought |
| DICOM | Digital Imaging and Communications in Medicine |
| ECA | Ensayo Clínico Aleatorizado |
| EHR/EHR | Electronic Health Record |
| FDA | Food and Drug Administration |
| FHIR | Fast Healthcare Interoperability Resources |
| FT | Fine-Tuning |
| GMAI | Generalist Medical Artificial Intelligence |
| GDPR | General Data Protection Regulation |
| HCE | Historia Clínica Electrónica |
| HIPAA | Health Insurance Portability and Accountability Act |
| HL7 | Health Level Seven International |
| IA | Inteligencia Artificial |
| LLM | Large Language Model |
| LIME | Local Interpretable Model-agnostic Explanations |
| LOINC | Logical Observation Identifiers Names and Codes |
| LoRA | Low-Rank Adaptation |
| MD | Medical Device |
| MDR | Medical Device Regulation |
| ML | Machine Learning |
| NLP | Natural Language Processing |
| PCCP | Predetermined Change Control Plan |
| PHI | Protected Health Information |
| RAG | Retrieval-Augmented Generation |
| ROC | Receiver Operating Characteristic |
| SaMD | Software as a Medical Device |
| SHAP | Shapley Additive exPlanations |
| SNOMED CT | Systematized Nomenclature of Medicine Clinical Terms |
| XAI | Explainable Artificial Intelligence |

---

**Documento elaborado para ponencia:** "Kit de Herramientas de IA para el Consultorio Médico"  
**Duración estimada de ponencia:** 45 minutos  
**Formato:** Markdown (.md) - Convertible a PDF  
**Última actualización:** Marzo 2026

---

*Este glosario está diseñado para ser utilizado como material de referencia durante la preparación y presentación de la ponencia. Se recomienda revisar las fuentes originales para información actualizada sobre regulaciones y tecnologías en rápida evolución.*