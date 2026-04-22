# Discurso: Investigacion de Datasets Relacionados

**Para:** Presentacion ante el asesor de tesis
**Contexto:** Explicar los datasets existentes de anti-spoofing y como se comparan con HABLA 2.0

---

## Guion

Buenas [dia/tarde], profesor. Quiero presentarle la investigacion que hice sobre los datasets existentes de anti-spoofing de voz, especificamente los que incluyen espanol, y como se posiciona nuestro trabajo frente a ellos.

### LRLSpoof

El primer dataset que investigue es **LRLSpoof**, presentado en Interspeech 2026. Es un corpus masivo de 2,732 horas de audio sintetico en 66 idiomas, generado con 24 sistemas TTS diferentes. Para espanol, tiene 23 horas con 11 sistemas TTS. Lo interesante es que tres de nuestros sistemas de ataque coinciden con los de ellos: Fish Speech, Chatterbox y MeloTTS, que es el motor base de OpenVoice. Sin embargo, LRLSpoof no tiene Qwen3-TTS, que es uno de nuestros sistemas mas peligrosos con la similitud de hablante mas alta (0.720). Ademas, LRLSpoof no incluye audio bonafide — solo sintetico — y no diferencia entre acentos. Todo el espanol se trata como un solo idioma, sin distinguir entre mexicano, colombiano o argentino.

### SpeechFake-MD

El segundo dataset es **SpeechFake-MD**, publicado en ACL 2025. Tiene mas de 3 millones de muestras de deepfake en 46 idiomas con 40 herramientas TTS. Es el mas grande en escala. Para espanol reportan un EER de 0.12% a 0.42%, lo cual suena excelente, pero hay un problema: utilizan audio bonafide de Mozilla Common Voice, que es el mismo corpus de donde nosotros sacamos los textos para nuestros prompts. Sin embargo, como LRLSpoof, tampoco diferencian entre acentos del espanol y no tienen validacion de calidad por muestra — solo reportan EER global.

### HISPASpoof

El tercer dataset, y el mas relevante para nosotros, es **HISPASpoof**, del laboratorio VIPER de Purdue University. Este si es especifico para espanol: tiene 535,687 senales de voz en 6 acentos (peninsular, argentino, colombiano, mexicano, chileno y peruano) con 6 sistemas TTS (ElevenLabs, F5-TTS, FishSpeech, XTTS-v1.1, XTTS-v2 y YourTTS). Tienen 24 hablantes, 4 por acento.

El hallazgo mas critico de HISPASpoof es este: **los detectores entrenados en ingles fallan catastroficamente en espanol**. El mejor detector en ingles, Wav2Vec2-AASIST, tiene 0.27% de EER en ASVspoof, pero salta a 19.92% cuando se evalua en HISPASpoof. Esto valida completamente la premisa de nuestra tesis — la investigacion de anti-spoofing especifica por idioma es esencial.

Cuando entrenan Spec-ResNet directamente en datos de espanol, baja a 0.72% de EER. Esto demuestra que el problema no es que los detectores no funcionen, sino que necesitan datos en el idioma correcto.

### Como nos posicionamos

Ahora, donde entra nuestro trabajo. HABLA 2.0 se diferencia en varios aspectos clave:

**Primero, escala de hablantes.** Nosotros tenemos 1,567 hablantes contra los 24 de HISPASpoof. Eso es 65 veces mas, lo que nos da mucho mas poder estadistico por acento.

**Segundo, sistemas TTS novedosos.** Qwen3-TTS y OuteTTS no estan ni en HISPASpoof, ni en LRLSpoof, ni en SpeechFake-MD. Son arquitecturas completamente nuevas — Qwen usa un codec Dual-Track y OuteTTS usa un LLM con codec DAC. Representan vectores de ataque que no han sido estudiados.

**Tercero, partial spoof.** Ninguno de los tres datasets tiene audio parcialmente falsificado. Nosotros estamos construyendo el primer dataset de partial spoof para espanol latinoamericano, donde reemplazamos 1, 2 o 3 palabras individuales con versiones clonadas. Esto es una contribucion completamente nueva.

**Cuarto, validacion por muestra.** Nosotros validamos cada muestra generada con WER, CER, NISQA MOS y similitud de hablante ECAPA-TDNN. Los otros datasets solo reportan metricas globales como EER.

**Quinto, acentos latinoamericanos.** Incluimos venezolano y puertorriqueno, que HISPASpoof no tiene. Ellos incluyen peninsular (espanol europeo), que nosotros no tenemos. Son complementarios.

### Estado actual

En terminos de progreso, ya completamos las corridas de produccion de tres pipelines: FishGram con 95.2% de aprobacion, Qwen3-TTS con 87.9%, y OpenVoice con 83.4%. Cada uno genero aproximadamente 35,000 muestras sobre los 1,567 hablantes. Chatterbox esta corriendo ahora mismo y OuteTTS esta en cola. El pipeline de partial spoof esta en fase de prueba con la estrategia de Qwen.

El siguiente paso es comenzar a escribir el paper con toda esta informacion.

---

## Puntos clave si preguntan

- **Por que Qwen es el mas peligroso?** Tiene la similitud de hablante mas alta (0.720) con WER bajo (1.46%). Un detector basado en embeddings tendria mas dificultad detectandolo.
- **Por que OpenVoice tiene la tasa de aprobacion mas baja?** MeloTTS divide las oraciones en fragmentos, lo que causa artefactos de duracion. Ademas, la conversion de color tonal solo transfiere parcialmente la identidad del hablante (0.394 similitud).
- **Por que se descarto CosyVoice?** Solo soporta chino, ingles, japones, cantones y coreano. Genera audio que suena a chino cuando recibe texto en espanol. Limitacion fundamental del modelo.
- **Que es partial spoof?** Se toma un audio real, se clona la misma oracion completa con TTS, y luego se extraen 1-3 palabras del clon via alineamiento forzado y se insertan en el audio real con crossfade de 5ms. El resultado es un audio que es 90%+ real pero tiene palabras sinteticas.
