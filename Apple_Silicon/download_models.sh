#!/bin/bash
# Download pretrained models for Stage 1 inference

echo "Downloading Stage 1 pretrained models..."

cd ..

# Download the Global checkpoint (contains VAE models, mapping models, and detection model)
if [ ! -d "Global/checkpoints/restoration/VAE_A_quality" ] || [ ! -f "Global/checkpoints/detection/FT_Epoch_latest.pt" ]; then
    echo "Downloading checkpoints from GitHub releases..."
    
    cd Global/checkpoints
    wget -q --show-progress https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip
    
    echo "Extracting checkpoints..."
    unzip -o global_checkpoints.zip
    
    rm -f global_checkpoints.zip
    echo "Download complete!"
    cd ../..
else
    echo "Models already exist. Skipping download."
fi

# Verify the models are in place
echo ""
echo "Verifying model files..."
echo "Restoration models:"
for model in VAE_A_quality VAE_B_quality mapping_quality; do
    if [ -d "Global/checkpoints/restoration/$model" ]; then
        echo "  [OK] $model"
    else
        echo "  [MISSING] $model"
    fi
done

echo "Detection model:"
if [ -f "Global/checkpoints/detection/FT_Epoch_latest.pt" ]; then
    echo "  [OK] detection/FT_Epoch_latest.pt"
else
    echo "  [MISSING] detection/FT_Epoch_latest.pt"
fi

echo ""
echo "Download complete! You can now run Stage 1 inference."
echo ""
echo "Usage:"
echo "  # For photos without scratches (quality restoration):"
echo "  cd Apple_Silicon"
echo "  python stage1_inference.py --test_input /path/to/photos --outputs_dir /path/to/output --Quality_restore"
echo ""
echo "  # For photos with scratches (automatic mask detection):"
echo "  python stage1_inference.py --test_input /path/to/photos --outputs_dir /path/to/output --Scratch_and_Quality_restore"
echo ""
echo "  # For photos with scratches (manual mask):"
echo "  python stage1_inference.py --test_input /path/to/photos --test_mask /path/to/masks --outputs_dir /path/to/output --Scratch_and_Quality_restore"
