# ==============================================================================
# Makefile for Training Traffic Sign Recognition Models
#
# Usage:
#   make all          		- Train all available models (original versions).
#   make all-refactored     - Train all refactored models.
#   make train-v1-basic     - Train the baseline model (saves as model_v1.h5).
# 	make train-v2-advanced 	- Train the advanced model (saves as model_v2.h5)
#   make train-v3-vgg 		- Train the VGG-style model (saves as model_v3.h5).
#   make train-v4-gln 		- Train the GoogLeNet-style model (saves as model_v4.h5).
#   make train-v5-res 		- Train the ResNet-style model (saves as model_v5.h5).
#   make train-v6-vit 		- Train the Vision Transformer model (saves as model_v6.h5).
#
# Refactored versions (cleaner code, same functionality):
#   make refactored-v1      - Train refactored v1 (saves as model_v1_refactored.h5).
#   make refactored-v2      - Train refactored v2 (saves as model_v2_refactored.h5).
#   make refactored-v3      - Train refactored v3 (saves as model_v3_refactored.h5).
#   make refactored-v4      - Train refactored v4 (saves as model_v4_refactored.h5).
#   make refactored-v5      - Train refactored v5 (saves as model_v5_refactored.h5).
#   make refactored-v6      - Train refactored v6 (saves as model_v6_refactored.h5).
#
#   make clean        		- Remove all generated model files and cache folders.
#   make clean-refactored   - Remove only refactored model files.
# ==============================================================================

# --- Variables ---
PYTHON = python3
DATA_DIR = gtsrb

# Original model scripts and their corresponding output file names
V1_SCRIPT = traffic/traffic_v1_basic.py
V2_SCRIPT = traffic/traffic_v2_advanced.py
V3_SCRIPT = traffic/traffic_v3_vgg.py
V4_SCRIPT = traffic/traffic_v4_googlenet.py
V5_SCRIPT = traffic/traffic_v5_resnet.py
V6_SCRIPT = traffic/traffic_v6_vit.py

V1_MODEL = model_v1.h5
V2_MODEL = model_v2.h5
V3_MODEL = model_v3.h5
V4_MODEL = model_v4.h5
V5_MODEL = model_v5.h5
V6_MODEL = model_v6.h5

# Refactored model scripts and their corresponding output file names
V1_REFACTORED_SCRIPT = refactored/traffic_v1_basic_refactored.py
V2_REFACTORED_SCRIPT = refactored/traffic_v2_advanced_refactored.py
V3_REFACTORED_SCRIPT = refactored/traffic_v3_vgg_refactored.py
V4_REFACTORED_SCRIPT = refactored/traffic_v4_googlenet_refactored.py
V5_REFACTORED_SCRIPT = refactored/traffic_v5_resnet_refactored.py
V6_REFACTORED_SCRIPT = refactored/traffic_v6_vit_refactored.py

V1_REFACTORED_MODEL = model_v1_refactored.h5
V2_REFACTORED_MODEL = model_v2_refactored.h5
V3_REFACTORED_MODEL = model_v3_refactored.h5
V4_REFACTORED_MODEL = model_v4_refactored.h5
V5_REFACTORED_MODEL = model_v5_refactored.h5
V6_REFACTORED_MODEL = model_v6_refactored.h5

# --- Phony Targets (commands that don't produce a file with the same name) ---
.PHONY: all all-refactored clean clean-refactored \
        train-v1-basic train-v2-advanced train-v3-vgg train-v4-gln train-v5-res train-v6-vit \
        refactored-v1 refactored-v2 refactored-v3 refactored-v4 refactored-v5 refactored-v6

# --- Main Targets ---

# Default target: train all original models
all: $(V1_MODEL) $(V2_MODEL) $(V3_MODEL) $(V4_MODEL) $(V5_MODEL) $(V6_MODEL)

# Train all refactored models
all-refactored: $(V1_REFACTORED_MODEL) $(V2_REFACTORED_MODEL) $(V3_REFACTORED_MODEL) $(V4_REFACTORED_MODEL) $(V5_REFACTORED_MODEL) $(V6_REFACTORED_MODEL)

# ==================================================================================
# --- Model Training Rules ---

# These rules use the model file name as the target.
# Makefile will only run the command if the target file doesn't already exist.
# The `$@` variable automatically uses the target name as the output file argument.

$(V1_MODEL): $(V1_SCRIPT)
	@echo "--- Training Baseline Model (v1) -> $(V1_MODEL) ---"
	$(PYTHON) $(V1_SCRIPT) $(DATA_DIR) $@

$(V2_MODEL): $(V2_SCRIPT)
	@echo "--- Training Advanced Model (v2) -> $(V2_MODEL) ---"
	$(PYTHON) $(V2_SCRIPT) $(DATA_DIR) $@

$(V3_MODEL): $(V3_SCRIPT)
	@echo "--- Training VGG-Style Model (v3) -> $(V3_MODEL) ---"
	$(PYTHON) $(V3_SCRIPT) $(DATA_DIR) $@

$(V4_MODEL): $(V4_SCRIPT)
	@echo "--- Training GoogLeNet-Style Model (v4) -> $(V4_MODEL) ---"
	$(PYTHON) $(V4_SCRIPT) $(DATA_DIR) $@

$(V5_MODEL): $(V5_SCRIPT)
	@echo "--- Training ResNet-Style Model (v5) -> $(V5_MODEL) ---"
	$(PYTHON) $(V5_SCRIPT) $(DATA_DIR) $@

$(V6_MODEL): $(V6_SCRIPT)
	@echo "--- Training Vision Transformer Model (v6) -> $(V6_MODEL) ---"
	$(PYTHON) $(V6_SCRIPT) $(DATA_DIR) $@

# ==================================================================================
# --- Refactored Model Training Rules ---

$(V1_REFACTORED_MODEL): $(V1_REFACTORED_SCRIPT)
	@echo "--- Training Refactored Baseline Model (v1) -> $(V1_REFACTORED_MODEL) ---"
	$(PYTHON) $(V1_REFACTORED_SCRIPT) $(DATA_DIR) $@

$(V2_REFACTORED_MODEL): $(V2_REFACTORED_SCRIPT)
	@echo "--- Training Refactored Advanced Model (v2) -> $(V2_REFACTORED_MODEL) ---"
	$(PYTHON) $(V2_REFACTORED_SCRIPT) $(DATA_DIR) $@

$(V3_REFACTORED_MODEL): $(V3_REFACTORED_SCRIPT)
	@echo "--- Training Refactored VGG-Style Model (v3) -> $(V3_REFACTORED_MODEL) ---"
	$(PYTHON) $(V3_REFACTORED_SCRIPT) $(DATA_DIR) $@

$(V4_REFACTORED_MODEL): $(V4_REFACTORED_SCRIPT)
	@echo "--- Training Refactored GoogLeNet-Style Model (v4) -> $(V4_REFACTORED_MODEL) ---"
	$(PYTHON) $(V4_REFACTORED_SCRIPT) $(DATA_DIR) $@

$(V5_REFACTORED_MODEL): $(V5_REFACTORED_SCRIPT)
	@echo "--- Training Refactored ResNet-Style Model (v5) -> $(V5_REFACTORED_MODEL) ---"
	$(PYTHON) $(V5_REFACTORED_SCRIPT) $(DATA_DIR) $@

$(V6_REFACTORED_MODEL): $(V6_REFACTORED_SCRIPT)
	@echo "--- Training Refactored Vision Transformer Model (v6) -> $(V6_REFACTORED_MODEL) ---"
	$(PYTHON) $(V6_REFACTORED_SCRIPT) $(DATA_DIR) $@
# ==================================================================================

# ==================================================================================
# --- Convenience Training Targets ---

# Original versions
train-v1-basic: $(V1_MODEL)
train-v2-advanced: $(V2_MODEL)
train-v3-vgg: $(V3_MODEL)
train-v4-gln: $(V4_MODEL)
train-v5-res: $(V5_MODEL)
train-v6-vit: $(V6_MODEL)

# Refactored versions
refactored-v1: $(V1_REFACTORED_MODEL)
refactored-v2: $(V2_REFACTORED_MODEL)
refactored-v3: $(V3_REFACTORED_MODEL)
refactored-v4: $(V4_REFACTORED_MODEL)
refactored-v5: $(V5_REFACTORED_MODEL)
refactored-v6: $(V6_REFACTORED_MODEL)
# ==================================================================================


# ==================================================================================
# --- Housekeeping ---
clean:
	@echo "Cleaning up all generated files..."
	@rm -f $(V1_MODEL) $(V2_MODEL) $(V3_MODEL) $(V4_MODEL) $(V5_MODEL) $(V6_MODEL)
	@rm -f $(V1_REFACTORED_MODEL) $(V2_REFACTORED_MODEL) $(V3_REFACTORED_MODEL) $(V4_REFACTORED_MODEL) $(V5_REFACTORED_MODEL) $(V6_REFACTORED_MODEL)
	@find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	@echo "Cleanup complete."

clean-refactored:
	@echo "Cleaning up refactored model files..."
	@rm -f $(V1_REFACTORED_MODEL) $(V2_REFACTORED_MODEL) $(V3_REFACTORED_MODEL) $(V4_REFACTORED_MODEL) $(V5_REFACTORED_MODEL) $(V6_REFACTORED_MODEL)
	@echo "Refactored models cleaned."
# ==================================================================================