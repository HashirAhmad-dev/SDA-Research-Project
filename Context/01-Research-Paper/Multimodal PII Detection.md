# Multimodal PII Detection

Intercepted payloads pass through a three-branch pipeline ordered by cost:

1. **Deterministic NLP:** Native text and text-selectable PDFs go to **Microsoft Presidio**, running regex and NER pattern matching under 150 ms.
2. **High-confidence OCR:** Image payloads go to **PaddleOCR**. If extraction confidence exceeds 0.85, the transcript is handed to Presidio.
3. **VLM Fallback:** When OCR confidence is low (e.g., cursive handwriting, dense layouts), the raw image is routed asynchronously to a fine-tuned Vision-Language Model (**PaliGemma-3B, INT8 quantized**) that infers entities directly from pixels.

## Why this structure?
The expensive VLM only sees payloads the cheaper branches cannot handle, keeping average latency low. INT8 quantization ensures it fits on commodity hardware. Asynchronous hand-off prevents the proxy request path from stalling.

Each branch emits a set of detected entities, reduced to a scalar sensitivity score `S` (0-100).
