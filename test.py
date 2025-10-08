from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("~/Coding/mlm_model")
model = AutoModelForMaskedLM.from_pretrained("~/Coding/mlm_model")

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
fill_mask("The quick brown [MASK] jumps over the lazy dog.")
