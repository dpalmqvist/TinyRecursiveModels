# fix_mps.py
import sys

# Read the pretrain.py file
with open('pretrain.py', 'r') as f:
    content = f.read()

# Add device detection at the top (after imports)
device_detection = """
# Device detection for Mac/CUDA/CPU
import torch
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    USE_COMPILE = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    USE_COMPILE = False  # Disable compilation for MPS due to compatibility issues
    print("Warning: Using MPS device. torch.compile disabled due to MPS limitations.")
else:
    DEVICE = torch.device("cpu")
    USE_COMPILE = False
print(f"Using device: {DEVICE}")
"""

# Find where to insert (after the imports, look for the first function or class definition)
import_end = content.find('\ndef ')
if import_end == -1:
    import_end = content.find('\nclass ')

if import_end > 0:
    content = content[:import_end] + '\n' + device_detection + '\n' + content[import_end:]

# Replace .cuda() with .to(DEVICE)
content = content.replace('.cuda()', '.to(DEVICE)')

# Find and modify torch.compile calls
# Look for model = torch.compile(model)
content = content.replace(
    'model = torch.compile(model)',
    'model = torch.compile(model) if USE_COMPILE else model'
)

# Save the modified file
with open('pretrain.py', 'w') as f:
    f.write(content)

print("pretrain.py has been modified for MPS compatibility")