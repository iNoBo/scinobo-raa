from transformers import T5ForConditionalGeneration
import os

CACHE_DIR = "/app/src/raa/models_cache"

model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-base",
    cache_dir=CACHE_DIR,
    torch_dtype="auto",
    device_map="auto"
)