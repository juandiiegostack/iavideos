# ClipHunter

ClipHunter descarga un VOD (Twitch, YouTube o Kick), graba un directo, o analiza un
archivo local, detecta los mejores momentos usando **chat**, **audio** y
**transcripción**, y exporta clips listos para publicar (horizontal o vertical 9:16,
con subtítulos opcionales estilo TikTok).

```bash
python cliphunter.py "https://twitch.tv/videos/123456789" --clips 10
```

El resultado es una carpeta con los `.mp4` y un `clips.json` que explica por qué se
eligió cada momento (score, señales, motivos, título y transcripción).

---

## Índice

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación paso a paso](#instalación-paso-a-paso)
- [Uso rápido](#uso-rápido)
- [Cómo funciona](#cómo-funciona)
- [Señales y puntuación](#señales-y-puntuación)
- [Opciones de la CLI](#opciones-de-la-cli)
- [Configuración](#configuración)
- [Variables de entorno](#variables-de-entorno)
- [Formato de clips.json](#formato-de-clipsjson)
- [Caché](#caché)
- [Modo directo](#modo-directo)
- [LLM opcional](#llm-opcional)
- [Subtítulos y formato vertical](#subtítulos-y-formato-vertical)
- [Pruebas y demo offline](#pruebas-y-demo-offline)
- [Solución de problemas](#solución-de-problemas)
- [Limitaciones](#limitaciones)
- [Estructura del proyecto](#estructura-del-proyecto)

---

## Características

- **Fuentes**: VOD de Twitch, YouTube y Kick; directos de Twitch/YouTube/Kick;
  archivos locales (`--file`); URLs directas `.mp4`/`.m3u8`.
- **Señales independientes y desactivables**:
  - **Chat**: ritmo de mensajes, picos respecto a la media, densidad de emotes y
    palabras de hype (ES/EN). Usa `chat-downloader`; en Twitch puede usar
    `TwitchDownloaderCLI` como respaldo.
  - **Audio**: RMS, subidas bruscas de volumen, gritos (banda 2.5–7.5 kHz) y risas
    (modulación de 3–9 Hz). Decodifica con ffmpeg + NumPy, sin librosa.
  - **Transcripción**: `faster-whisper` con timestamps por palabra, palabras clave
    ES/EN y cambios de velocidad del habla. CUDA automática si está disponible.
- **Selección**: ventana deslizante, normalización robusta, picos locales,
  supresión de no-máximos, clips no solapados, límites de duración, pre/post roll y
  ajuste a silencios.
- **Exportación**: recorte preciso con ffmpeg, H.264/AAC, `+faststart`.
  - Vertical 9:16 (1080×1920) con fondo desenfocado o recorte centrado.
  - Subtítulos ASS quemados, palabra a palabra o por frases.
- **LLM opcional** (cualquier API compatible con OpenAI) para reordenar y titular.
- **Caché** de descargas, audio, chat, transcripción y señales.
- **Errores accionables** y barras de progreso.
- **Sin secretos hardcodeados**: todo va por `.env` o variables de entorno.

---

## Requisitos

| Componente | Obligatorio | Notas |
|---|---|---|
| Python 3.10+ | Sí | Probado con 3.13 en Windows. |
| ffmpeg + ffprobe | Sí | Recorte, audio, vertical y subtítulos. |
| `yt-dlp` | Sí | Descarga de VODs y metadatos. |
| `streamlink` | Recomendado | Grabación de directos (si falta, usa yt-dlp). |
| `faster-whisper` | Sí | Transcripción y subtítulos. |
| GPU NVIDIA + CUDA | No | Acelera Whisper; en CPU funciona con `int8`. |
| `TwitchDownloaderCLI` | No | Respaldo para el chat de Twitch. |
| Endpoint LLM | No | Solo si usas `--llm`. |

---

## Instalación paso a paso

### 1. Instalar Python 3.10 o superior

Comprueba la versión:

```bash
python --version
```

Si es menor que 3.10, instala una versión reciente desde
<https://www.python.org/downloads/>. En Windows marca **"Add Python to PATH"**.

### 2. Instalar ffmpeg (y ffprobe)

**Windows (PowerShell):**

```powershell
winget install Gyan.FFmpeg
```

Alternativas: `choco install ffmpeg` o descargar los binarios de
<https://www.gyan.dev/ffmpeg/builds/> y añadir la carpeta `bin` al `PATH`.
Cierra y vuelve a abrir la terminal después de instalar.

**macOS:**

```bash
brew install ffmpeg
```

**Linux (Debian/Ubuntu):**

```bash
sudo apt update && sudo apt install ffmpeg
```

En Fedora: `sudo dnf install ffmpeg`; en Arch: `sudo pacman -S ffmpeg`.

Verifica que ambos están disponibles:

```bash
ffmpeg -version
ffprobe -version
```

### 3. Crear un entorno virtual e instalar dependencias

Desde la carpeta del proyecto:

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Si PowerShell bloquea la activación, ejecuta una vez:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 4. (Opcional) Instalar el comando `cliphunter`

```bash
pip install -e .
```

Después podrás usar `cliphunter ...` en lugar de `python cliphunter.py ...`.
También funciona `python -m cliphunter ...` sin instalar nada.

### 5. (Opcional) Configurar el LLM y rutas

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Edita `.env` y rellena solo lo que necesites (clave del LLM, rutas de binarios,
modelo de Whisper...). Nunca subas `.env` a un repositorio: ya está en `.gitignore`.

### 6. Comprobar la instalación

```bash
python cliphunter.py --version
python cliphunter.py --help
```

La primera vez que uses Whisper se descargará el modelo (por defecto `small`) desde
Hugging Face. Necesita conexión a Internet esa primera vez.

---

## Uso rápido

### VOD de Twitch

```bash
python cliphunter.py "https://twitch.tv/videos/123456789" --clips 10
```

### VOD de YouTube

```bash
python cliphunter.py "https://www.youtube.com/watch?v=XXXXXXXXXXX" --clips 5 --vertical --subtitles
```

### VOD de Kick

```bash
python cliphunter.py "https://kick.com/video/12345678-abcd-1234-abcd-1234567890ab" --clips 3
```

### Directo (grabar y analizar)

```bash
python cliphunter.py "https://twitch.tv/tucanal" --live --max-duration 20 --clips 5
```

Graba 20 minutos y, al terminar, analiza lo grabado. Con `--max-duration 0` graba
hasta que pulses `Ctrl+C`.

### Archivo local + chat exportado

```bash
python cliphunter.py --file partida.mp4 --chat-file chat.json --clips 8
```

### Solo analizar, sin exportar

```bash
python cliphunter.py "https://www.youtube.com/watch?v=XXXXXXXXXXX" --dry-run
```

Escribe `analysis.json` con las señales y los momentos candidatos, pero no genera
vídeos.

### Clips verticales con subtítulos y LLM

```bash
python cliphunter.py "https://twitch.tv/videos/123456789" \
  --clips 10 --vertical --vertical-mode blur --subtitles --llm \
  --min-dur 15 --max-dur 45 --out clips_finales
```

### Desactivar señales

```bash
python cliphunter.py "https://www.youtube.com/watch?v=XXXXXXXXXXX" --no-chat
python cliphunter.py "https://www.youtube.com/watch?v=XXXXXXXXXXX" --no-transcript
python cliphunter.py --file video.mp4 --no-chat --no-audio --no-transcript --dry-run
```

### VOD privado o solo para suscriptores

```bash
python cliphunter.py "https://twitch.tv/videos/123456789" --cookies-from-browser chrome
# o con un cookies.txt exportado:
python cliphunter.py "https://twitch.tv/videos/123456789" --cookies cookies.txt
```

---

## Cómo funciona

1. **Resolución de la fuente**
   - Clasifica la URL (Twitch/YouTube/Kick/directa) sin acceso a red.
   - Obtiene título, duración y estado con `yt-dlp`; si es un directo, graba con
     `streamlink` (o `yt-dlp`) a segmentos MPEG-TS y los une en MP4.
2. **Adquisición de señales**
   - Descarga el video al caché (si no estaba).
   - Descarga el chat salvo que uses `--no-chat` o una plataforma sin API pública.
   - Extrae audio mono 16 kHz con ffmpeg.
   - Transcribe con Whisper si la señal de transcripción o los subtítulos están
     activos.
3. **Análisis**
   - Cada señal se calcula en una rejilla temporal común (`window=10 s`,
     `hop=2 s`), se suaviza y se normaliza por percentiles.
   - Se combinan con pesos (`chat 0.40`, `audio 0.35`, `transcript 0.25`).
4. **Selección**
   - Picos locales por encima del umbral; si faltan candidatos se usa un umbral
     relativo al mejor pico.
   - Se fusionan picos cercanos, se descartan solapamientos y se respetan
     `--min-dur`/`--max-dur`.
   - Los cortes se ajustan a puntos silenciosos cercanos cuando es posible.
5. **Exportación**
   - Se recorta con re-encode (preciso, no keyframe) y se compone el vertical.
   - Se generan subtítulos ASS con la palabra activa resaltada.
   - Se escribe `clips.json` con el desglose de señales y motivos.

---

## Señales y puntuación

### Chat

- Ritmo de mensajes por ventana y **pico relativo** a una media móvil.
- Densidad de emotes (Twitch/YouTube) y exceso respecto a la línea base.
- Palabras de hype en español e inglés (`clutch`, `no way`, `que locura`, `gg`...).
- Si hay menos de `chat.min_messages` (20 por defecto), la señal se marca como no
  disponible y el resto sigue funcionando.

### Audio

- **RMS**: volumen medio por ventana.
- **Novelty**: subida brusca respecto a la media local (gritos, golpes, risas).
- **Scream**: energía en 2.5–7.5 kHz ponderada por volumen.
- **Laughter**: modulación de amplitud de 3–9 Hz (risas).
- Se guarda una envolvente de volumen para ajustar los cortes a silencios.

### Transcripción

- `faster-whisper` con `word_timestamps=True`.
- Palabras clave ES/EN configurables, con pesos mayores para frases de varias
  palabras.
- Picos de velocidad del habla (habla acelerada = emoción).
- El texto del clip se guarda en `clips.json` y se usa para títulos y subtítulos.

### Pesos

En `config.yaml`:

```yaml
weights:
  chat: 0.40
  audio: 0.35
  transcript: 0.25
```

Si una señal no está disponible, su peso se reparte automáticamente entre las
demás. Un peso a `0` desactiva esa señal en la combinación (aunque se siga
calculando para los motivos).

---

## Opciones de la CLI

```text
python cliphunter.py [URL | --file ARCHIVO] [opciones]
```

### Fuente

| Opción | Defecto | Descripción |
|---|---|---|
| `url` | — | URL de VOD o directo (Twitch, YouTube, Kick o directa). |
| `--file RUTA` | — | Archivo local de video. Excluyente con `url`. |
| `--live` | off | Graba un directo antes de analizar. |
| `--max-duration MIN` | `5` | Minutos a grabar en modo directo (`0` = hasta Ctrl+C). |
| `--chat-file JSON` | — | Chat local (JSON/JSONL o formato TwitchDownloader). |
| `--cookies-from-browser NAV` | — | Cookies de `chrome`, `firefox`, `edge`... para VODs privados. |
| `--cookies ARCHIVO` | — | `cookies.txt` para yt-dlp. |

### Selección

| Opción | Defecto | Descripción |
|---|---|---|
| `--clips N` | `10` | Número de clips a exportar. |
| `--min-dur SEG` | `15` | Duración mínima del clip. |
| `--max-dur SEG` | `60` | Duración máxima del clip. |
| `--pre-roll SEG` | `8` | Segundos antes del pico. |
| `--post-roll SEG` | `4` | Segundos después del pico. |
| `--window SEG` | `10` | Ventana de análisis. |
| `--hop SEG` | `2` | Paso entre ventanas. |

### Señales

| Opción | Defecto | Descripción |
|---|---|---|
| `--chat` / `--no-chat` | activo | Activa/desactiva la señal de chat. |
| `--audio` / `--no-audio` | activo | Activa/desactiva la señal de audio. |
| `--transcript` / `--no-transcript` | activo | Activa/desactiva Whisper. |
| `--model NOMBRE` | `small` | Modelo Whisper (`tiny`, `base`, `small`, `medium`, `large-v3`...). |
| `--device auto\|cpu\|cuda` | `auto` | Dispositivo para Whisper. |
| `--language es\|en\|...` | autodetectar | Idioma para Whisper. |

> Los subtítulos implican transcripción aunque uses `--no-transcript`: si pides
> `--subtitles`, Whisper se ejecuta igualmente.

### Formato de salida

| Opción | Defecto | Descripción |
|---|---|---|
| `--vertical` / `--no-vertical` | horizontal | Exporta 1080×1920 (9:16). |
| `--vertical-mode blur\|crop` | `blur` | Fondo desenfocado o recorte centrado. |
| `--subtitles` / `--no-subtitles` | off | Quema subtítulos ASS. |
| `--subtitle-mode word\|chunk` | `word` | Palabra a palabra o por frases. |
| `--llm` / `--no-llm` | off | Reordena y titula con un LLM. |
| `--workers N` | `2` | Exportaciones ffmpeg en paralelo. |
| `--out DIR` | `clips` | Carpeta de salida. |
| `--dry-run` | off | Analiza y escribe `analysis.json` sin exportar. |

### Caché y diagnóstico

| Opción | Descripción |
|---|---|
| `--cache-dir DIR` | Carpeta de caché alternativa. |
| `--no-cache` | No reutiliza ni conserva caché (usa un temporal). |
| `--clear-cache` | Borra la caché de esta fuente antes de empezar. |
| `--force` | Re-descarga, recalcula y sobrescribe todo. |
| `--config YAML` | Archivo de configuración alternativo. |
| `--verbose` | Logs detallados y trazas completas. |
| `--version` | Muestra la versión. |

La precedencia de configuración es: **CLI > variables de entorno > YAML > defectos**.

---

## Configuración

El archivo base es `cliphunter/config.yaml`. Para personalizar sin tocar el original,
copia y edita un YAML propio:

```bash
cp cliphunter/config.yaml mi_config.yaml
python cliphunter.py "https://twitch.tv/videos/123" --config mi_config.yaml
```

Secciones principales:

```yaml
analysis:
  window: 10.0
  hop: 2.0
  signals: { chat: true, audio: true, transcript: true }

weights: { chat: 0.40, audio: 0.35, transcript: 0.25 }

scoring:
  peak_threshold: 0.42
  relative_threshold: 0.35
  min_peak_distance: 12.0
  merge_distance: 8.0
  max_overlap: 0.0

selection:
  clips: 10
  min_duration: 15.0
  max_duration: 60.0
  pre_roll: 8.0
  post_roll: 4.0
  silence_snap: 3.0

export:
  vertical: false
  vertical_mode: blur
  vertical_blur: 28
  crf: 20
  preset: veryfast
  workers: 2

subtitles:
  enabled: false
  mode: word
  font: Arial Black
  font_size_horizontal: 56
  font_size_vertical: 104
  highlight_color: "&H00FFFF&"
```

Los valores del YAML se validan al arrancar (duraciones, pesos, umbrales...). Si algo
es incoherente, verás un error claro antes de descargar nada.

---

## Variables de entorno

Puedes definirlas en `.env` (en la carpeta desde la que ejecutas) o en el sistema.

| Variable | Uso |
|---|---|
| `CLIPHUNTER_LLM_BASE_URL` | Base de la API compatible con OpenAI. |
| `CLIPHUNTER_LLM_API_KEY` | Clave del LLM. |
| `CLIPHUNTER_LLM_MODEL` | Modelo del LLM. |
| `CLIPHUNTER_CACHE_DIR` | Carpeta de caché. |
| `CLIPHUNTER_WHISPER_MODEL` | Modelo Whisper por defecto. |
| `CLIPHUNTER_FFMPEG` | Ruta explícita a `ffmpeg`. |
| `CLIPHUNTER_FFPROBE` | Ruta explícita a `ffprobe`. |
| `CLIPHUNTER_STREAMLINK` | Ruta explícita a `streamlink`. |
| `CLIPHUNTER_TWITCHDOWNLOADER` | Ruta a `TwitchDownloaderCLI`. |

Las variables de entorno **no** pisan los argumentos explícitos de la CLI.

---

## Formato de clips.json

Ejemplo simplificado:

```json
{
  "tool": "ClipHunter",
  "version": "0.1.0",
  "created_at": "2026-11-01T22:00:00+00:00",
  "source": {
    "file": "C:\\cache\\twitch_123456789\\source.mp4",
    "duration": 5400.2,
    "width": 1920,
    "height": 1080,
    "fps": 60.0,
    "has_audio": true,
    "title": "Ranked con viewers",
    "source_url": "https://twitch.tv/videos/123456789",
    "video_id": "123456789",
    "platform": "twitch"
  },
  "settings": {
    "clips_requested": 10,
    "window": 10.0,
    "hop": 2.0,
    "min_duration": 15.0,
    "max_duration": 60.0,
    "pre_roll": 8.0,
    "post_roll": 4.0,
    "vertical": true,
    "vertical_mode": "blur",
    "subtitles": true,
    "llm": false,
    "signals": { "chat": true, "audio": true, "transcript": true },
    "weights": { "chat": 0.4, "audio": 0.35, "transcript": 0.25 }
  },
  "clips": [
    {
      "index": 1,
      "file": "clip_001_no-puede-ser.mp4",
      "subtitle_file": "clip_001_no-puede-ser.ass",
      "start": 1234.5,
      "end": 1249.5,
      "duration": 15.0,
      "peak": 1240.0,
      "score": 0.93,
      "signals": { "chat": 0.98, "audio": 0.84, "transcript": 0.71 },
      "reasons": [
        "chat: 18.2× el promedio (72 msg/10s)",
        "chat: pico de emotes",
        "audio: grito / pico agudo",
        "transcripción: «no puede ser», «qué locura»"
      ],
      "title": "No puede ser",
      "transcript": "no puede ser, qué locura, increíble...",
      "llm_score": null,
      "llm_reason": ""
    }
  ],
  "errors": []
}
```

- `start`, `end` y `peak` están en segundos desde el inicio de la fuente.
- `signals` contiene el valor normalizado (0–1) de cada señal dentro del clip.
- `errors` lista los clips que fallaron al exportar, sin abortar el resto.
- Con `--dry-run` se escribe `analysis.json` con `times`, `scores` y los candidatos.

---

## Caché

Por defecto:

| Sistema | Ruta |
|---|---|
| Windows | `%LOCALAPPDATA%\ClipHunter\cache` |
| macOS | `~/Library/Caches/ClipHunter` |
| Linux | `$XDG_CACHE_HOME/cliphunter` o `~/.cache/cliphunter` |

Se guardan por fuente:

- `source.mp4` (o `.mkv`/`.webm`) descargado.
- `audio_16000.wav`.
- `chat.json` y `chat_unavailable.json`.
- `transcript_<modelo>_<hash>.json` (hash de idioma/beam/VAD).
- `signals_<hash>.json` (hash de la configuración de análisis).

Los clips **siempre se vuelven a exportar** al repetir una ejecución, para que una
carpeta de salida reutilizada no muestre clips antiguos. Para forzar también la
descarga y el análisis usa `--force`; para no conservar nada usa `--no-cache`.

---

## Modo directo

```bash
python cliphunter.py "https://twitch.tv/canal" --live --max-duration 10
```

- `--max-duration` está en **minutos**; `0` graba hasta `Ctrl+C`.
- `streamlink` escribe MPEG-TS por stdout y ffmpeg lo segmenta cada 30 s. Al parar,
  los segmentos se unen en `source.mp4` y se analiza todo lo grabado.
- El chat se captura en paralelo (Twitch/YouTube) y se alinea con la grabación.
- Un primer `Ctrl+C` detiene la grabación y continúa con el análisis; un segundo
  `Ctrl+C` cancela todo.
- Si el canal está offline, verás un error claro en lugar de un archivo vacío.

---

## LLM opcional

ClipHunter habla con cualquier endpoint compatible con
`POST {base_url}/chat/completions` (OpenAI, Groq, OpenRouter, Ollama, LM Studio...).

```bash
# .env
CLIPHUNTER_LLM_BASE_URL=https://api.openai.com/v1
CLIPHUNTER_LLM_API_KEY=sk-...
CLIPHUNTER_LLM_MODEL=gpt-4o-mini
```

```bash
python cliphunter.py "https://twitch.tv/videos/123456789" --clips 10 --llm
```

- El LLM recibe los candidatos con sus señales y transcripción, devuelve un score
  de 1 a 10, un título corto y un motivo.
- El score final se mezcla con el heurístico según `llm.weight` (0.30 por defecto).
- Si el LLM falla (red, cuota, JSON inválido), se avisa y se continúa con el ranking
  heurístico: nunca se pierde la ejecución.
- No se envía el video ni el audio, solo texto (títulos, motivos y transcripción).

---

## Subtítulos y formato vertical

```bash
python cliphunter.py "https://www.youtube.com/watch?v=XXXXXXXXXXX" \
  --clips 5 --vertical --subtitles
```

- **Vertical** (`--vertical`): 1080×1920.
  - `--vertical-mode blur` (defecto): fondo ampliado y desenfocado + video centrado.
  - `--vertical-mode crop`: recorte centrado que llena el encuadre.
- **Subtítulos**: se genera un `.ass` por clip y se quema con libass.
  - `--subtitle-mode word`: muestra la frase y resalta la palabra activa.
  - `--subtitle-mode chunk`: muestra frases completas.
  - La fuente, tamaño, colores y márgenes se configuran en `subtitles` del YAML.
  - Si un clip no tiene palabras transcritas, se exporta sin subtítulos y se avisa
    (no falla la ejecución).

---

## Pruebas y demo offline

### Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

La suite incluye pruebas unitarias de configuración, caché, chat, audio,
transcripción, scorer, subtítulos, LLM y descarga, además de pruebas de integración
con ffmpeg (se omiten si ffmpeg no está instalado).

### Demo sin descargar nada

```bash
python scripts/make_demo_media.py
python cliphunter.py --file demo/demo_stream.mp4 --chat-file demo/demo_chat.json \
  --clips 3 --vertical --no-transcript --out demo/clips
```

Genera un video sintético de 120 s con tres momentos "hype" y un chat alineado.
Es la forma más rápida de comprobar la instalación sin depender de una URL pública.

---

## Solución de problemas

### "No se encontró ffmpeg"

Instálalo (ver [instalación](#instalación-paso-a-paso)) y comprueba `ffmpeg -version`.
Si está en una ruta no estándar, define `CLIPHUNTER_FFMPEG` y `CLIPHUNTER_FFPROBE`.

### Fallo al descargar un VOD

- Actualiza el extractor: `pip install -U yt-dlp`.
- Comprueba que el video es público.
- Para contenido privado/solo suscriptores usa `--cookies-from-browser chrome`
  (o `--cookies cookies.txt`).

### El chat no se descarga

- Twitch: `chat-downloader` suele bastar. Como respaldo, instala
  `TwitchDownloaderCLI` y define `CLIPHUNTER_TWITCHDOWNLOADER` con su ruta.
- YouTube: el chat de VODs antiguos puede no estar disponible.
- Kick: no hay API pública de chat para VODs; la señal se omite.
- También puedes exportar el chat y pasarlo con `--chat-file`.

### No se selecciona ningún clip

- Baja `scoring.peak_threshold` (por ejemplo a `0.30`) o
  `scoring.relative_threshold` en tu YAML.
- Revisa que el video dure al menos `selection.min_duration`.
- Prueba `--dry-run` y mira `analysis.json` para ver si las señales tienen
  variación. Un video con audio constante y sin chat puede no tener picos.
- Desactiva señales ruidosas: `--no-transcript`, `--no-chat`.

### Whisper tarda mucho

- En CPU usa un modelo más pequeño: `--model base` o `--model tiny`.
- Desactiva la transcripción si no necesitas subtítulos: `--no-transcript`.
- Con NVIDIA, comprueba que `ctranslate2` detecta la GPU (`--device cuda`).
  El primer uso descarga el modelo y tarda más.

### Los subtítulos no aparecen

- Comprueba que ffmpeg tiene libass: `ffmpeg -filters` debe listar `ass`.
- En Linux puede faltar la fuente `Arial Black`; instala
  `ttf-mscorefonts-installer` o cambia `subtitles.font` a una fuente instalada.
- Verifica que el clip tenga palabras transcritas; si no, se omite el subtítulo.

### Acentos raros en la consola de Windows

Define `PYTHONIOENCODING=utf-8` antes de ejecutar:

```powershell
$env:PYTHONIOENCODING='utf-8'
```

ClipHunter ya intenta reconfigurar stdout/stderr a UTF-8, pero algunos terminales
antiguos lo ignoran.

### El directo termina antes de tiempo

- `streamlink` puede cerrar si el stream se corta; revisa el log con `--verbose`.
- `--max-duration 0` graba hasta `Ctrl+C`; un valor mayor que 0 es un límite en
  minutos, no en segundos.

---

## Limitaciones

- **Kick**: descarga de VOD y detección por audio/transcripción; no hay chat público
  de VOD, así que esa señal no está disponible.
- **Directos**: solo funcionan mientras el canal está en vivo. No se pueden analizar
  directos ya finalizados salvo que estén publicados como VOD.
- **Whisper**: la calidad depende del modelo, el idioma y la mezcla de audio (música,
  gritos, varios hablantes). En CPU puede ser lento para VODs muy largos.
- **Detección heurística**: los scores son una ayuda, no una verdad absoluta.
  Ajusta umbrales, ventanas y pesos a tu tipo de contenido.
- **Vertical**: el modo `blur` rellena con el fondo desenfocado; el modo `crop`
  pierde los laterales. Revisa los clips antes de publicar.
- **Derechos**: descarga y publica solo contenido que tengas permiso para usar.
  Respeta los términos de servicio de Twitch, YouTube y Kick.

---

## Estructura del proyecto

```text
cliphunter.py                 # entrada: python cliphunter.py ...
cliphunter/
  __main__.py                 # python -m cliphunter
  cli.py                      # argparse y validaciones
  config.py / config.yaml     # configuración, entorno y validación
  pipeline.py                 # orquestación completa
  downloader.py               # URLs, yt-dlp, chat, grabación en directo
  scorer.py                   # señales -> picos -> clips
  exporter.py                 # ffmpeg: recorte, vertical, subtítulos
  subtitles.py                # generación de ASS
  llm.py                      # OpenAI-compatible opcional
  cache.py / models.py / utils.py / exceptions.py
  signals/
    chat.py                   # ritmo, emotes, hype
    audio.py                  # RMS, gritos, risas
    transcript.py             # Whisper + palabras clave
scripts/make_demo_media.py    # demo sintética offline
tests/                        # pytest (unitarias + integración ffmpeg)
requirements.txt
requirements-dev.txt
.env.example
pyproject.toml
```

---

Hecho para trabajar en CPU sin dependencias pesadas más allá de Whisper, y para
aprovechar CUDA automáticamente cuando está disponible.
