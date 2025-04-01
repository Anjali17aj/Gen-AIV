import sys
sys.path.append("IndicTrans2")  # Add IndicTrans2 to Python path

from inference.engine import Model  # Now import should work

# Load the model
model = Model(ckpt_dir="IndicTrans2/model_configs", device="cpu")

# Test translation
output_text = model.ctranslate2_translate_lines(["Hello, how are you?"])
print(output_text)
