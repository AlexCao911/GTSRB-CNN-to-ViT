#!/bin/bash
set -e

echo "======================================================================"
echo "Generate Professional Architecture Diagrams with PlotNeuralNet"
echo "======================================================================"

# 1. 克隆 PlotNeuralNet（如果不存在）
if [ ! -d "plotneuralnet" ]; then
    echo "Cloning PlotNeuralNet..."
    git clone https://github.com/HarisIqbal88/PlotNeuralNet.git plotneuralnet
fi

# 2. 安装 LaTeX 包（如果需要）
echo ""
echo "Checking LaTeX packages..."
if ! kpsewhich standalone.cls > /dev/null 2>&1; then
    echo "Installing required LaTeX packages..."
    sudo tlmgr update --self
    sudo tlmgr install standalone adjustbox collectbox varwidth import
else
    echo "✓ LaTeX packages already installed"
fi

# 3. 创建示例输入图片（如果不存在）
if [ ! -f "plotneuralnet/examples/fcn8s/cats.jpg" ]; then
    echo "Note: Input image not found, will use placeholder"
fi

# 4. 创建专业的模型脚本
echo ""
echo "Creating professional model scripts..."
mkdir -p plotneuralnet/pyexamples

# V1: Basic CNN - 更专业的版本
cat > plotneuralnet/pyexamples/gtsrb_v1_lenet.py << 'EOF'
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # Input layer with image
    to_input('../examples/fcn8s/cats.jpg', to='(-3,0,0)', width=2, height=2),
    
    # Conv Block 1
    to_ConvConvRelu(name='ccr_b1', s_filer=32, n_filer=32, offset="(0,0,0)", to="(0,0,0)", 
                    width=(2,2), height=28, depth=28, caption="Conv 32"),
    to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", 
            width=1, height=14, depth=14, opacity=0.5),
    
    # Flatten representation
    to_ConvRes(name='flatten', s_filer=128, n_filer=128, offset="(2,0,0)", to="(pool_b1-east)",
               width=1, height=40, depth=1, caption="Flatten"),
    
    # Dense layers
    to_ConvRes(name='fc1', s_filer=128, n_filer=128, offset="(1.5,0,0)", to="(flatten-east)",
               width=3, height=30, depth=1, caption="Dense 128"),
    
    # Output
    to_SoftMax(name="soft1", s_filer=43, offset="(2,0,0)", to="(fc1-east)", 
               width=1.5, height=25, depth=25, caption="Softmax 43"),
    
    # Connections
    to_connection("ccr_b1", "pool_b1"),
    to_connection("pool_b1", "flatten"),
    to_connection("flatten", "fc1"),
    to_connection("fc1", "soft1"),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
EOF

# V2: Improved CNN - 更详细的版本
cat > plotneuralnet/pyexamples/gtsrb_v2_alexnet.py << 'EOF'
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # Input
    to_input('../examples/fcn8s/cats.jpg', to='(-3,0,0)', width=2, height=2),
    
    # Block 1: Conv x2 + Pool
    to_ConvConvRelu(name='ccr_b1', s_filer=32, n_filer=32, offset="(0,0,0)", to="(0,0,0)", 
                    width=(2,2), height=28, depth=28, caption="Conv 32x2"),
    to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", 
            width=1, height=13, depth=13, opacity=0.5),
    
    # Block 2: Conv x2 + Pool
    to_ConvConvRelu(name='ccr_b2', s_filer=64, n_filer=64, offset="(1,0,0)", to="(pool_b1-east)", 
                    width=(4,4), height=11, depth=11, caption="Conv 64x2"),
    to_Pool(name="pool_b2", offset="(0,0,0)", to="(ccr_b2-east)", 
            width=1, height=4, depth=4, opacity=0.5),
    
    # Flatten
    to_ConvRes(name='flatten', s_filer=512, n_filer=512, offset="(2,0,0)", to="(pool_b2-east)",
               width=1, height=35, depth=1, caption="Flatten"),
    
    # Dense
    to_ConvRes(name='fc1', s_filer=512, n_filer=512, offset="(1.5,0,0)", to="(flatten-east)",
               width=5, height=40, depth=1, caption="Dense 512"),
    
    # Output
    to_SoftMax(name="soft1", s_filer=43, offset="(2,0,0)", to="(fc1-east)", 
               width=1.5, height=25, depth=25, caption="43 classes"),
    
    # Connections
    to_connection("ccr_b1", "pool_b1"),
    to_connection("pool_b1", "ccr_b2"),
    to_connection("ccr_b2", "pool_b2"),
    to_connection("pool_b2", "flatten"),
    to_connection("flatten", "fc1"),
    to_connection("fc1", "soft1"),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
EOF

# V3: VGG - 深度网络
cat > plotneuralnet/pyexamples/gtsrb_v3_vgg.py << 'EOF'
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # Input
    to_input('../examples/fcn8s/cats.jpg', to='(-3,0,0)', width=2, height=2),
    
    # Block 1
    to_ConvConvRelu(name='ccr_b1', s_filer=32, n_filer=32, offset="(0,0,0)", to="(0,0,0)", 
                    width=(2,2), height=30, depth=30, caption="32"),
    to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", 
            width=1, height=15, depth=15, opacity=0.5),
    
    # Block 2
    to_ConvConvRelu(name='ccr_b2', s_filer=64, n_filer=64, offset="(1,0,0)", to="(pool_b1-east)", 
                    width=(4,4), height=15, depth=15, caption="64"),
    to_Pool(name="pool_b2", offset="(0,0,0)", to="(ccr_b2-east)", 
            width=1, height=7, depth=7, opacity=0.5),
    
    # Block 3
    to_ConvConvRelu(name='ccr_b3', s_filer=128, n_filer=128, offset="(1,0,0)", to="(pool_b2-east)", 
                    width=(6,6), height=7, depth=7, caption="128"),
    to_Pool(name="pool_b3", offset="(0,0,0)", to="(ccr_b3-east)", 
            width=1, height=3, depth=3, opacity=0.5),
    
    # Dense layers
    to_ConvRes(name='fc1', s_filer=512, n_filer=512, offset="(2,0,0)", to="(pool_b3-east)",
               width=5, height=40, depth=1, caption="FC 512"),
    to_ConvRes(name='fc2', s_filer=256, n_filer=256, offset="(1,0,0)", to="(fc1-east)",
               width=3, height=30, depth=1, caption="FC 256"),
    
    # Output
    to_SoftMax(name="soft1", s_filer=43, offset="(2,0,0)", to="(fc2-east)", 
               width=1.5, height=25, depth=25, caption="43"),
    
    # Connections
    to_connection("ccr_b1", "pool_b1"),
    to_connection("pool_b1", "ccr_b2"),
    to_connection("ccr_b2", "pool_b2"),
    to_connection("pool_b2", "ccr_b3"),
    to_connection("ccr_b3", "pool_b3"),
    to_connection("pool_b3", "fc1"),
    to_connection("fc1", "fc2"),
    to_connection("fc2", "soft1"),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
EOF

# V4: GoogLeNet - Inception模块
cat > plotneuralnet/pyexamples/gtsrb_v4_googlenet.py << 'EOF'
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # Input
    to_input('../examples/fcn8s/cats.jpg', to='(-3,0,0)', width=2, height=2),
    
    # Initial conv
    to_Conv(name='conv1', s_filer=32, n_filer=32, offset="(0,0,0)", to="(0,0,0)", 
            width=2, height=30, depth=30, caption="Conv 32"),
    to_Pool(name="pool1", offset="(0,0,0)", to="(conv1-east)", 
            width=1, height=15, depth=15, opacity=0.5),
    
    # Inception 1 (simplified)
    to_ConvConvRelu(name='inc1', s_filer=72, n_filer=72, offset="(2,0,0)", to="(pool1-east)", 
                    width=(4,4), height=15, depth=15, caption="Inception 72"),
    to_Pool(name="pool2", offset="(1,0,0)", to="(inc1-east)", 
            width=1, height=7, depth=7, opacity=0.5),
    
    # Inception 2
    to_ConvConvRelu(name='inc2', s_filer=144, n_filer=144, offset="(2,0,0)", to="(pool2-east)", 
                    width=(6,6), height=7, depth=7, caption="Inception 144"),
    
    # GAP
    to_Pool(name="gap", offset="(2,0,0)", to="(inc2-east)", 
            width=6, height=1, depth=1, opacity=0.5, caption="GAP"),
    
    # Dense
    to_ConvRes(name='fc1', s_filer=512, n_filer=512, offset="(2,0,0)", to="(gap-east)",
               width=5, height=40, depth=1, caption="FC 512"),
    
    # Output
    to_SoftMax(name="soft1", s_filer=43, offset="(2,0,0)", to="(fc1-east)", 
               width=1.5, height=25, depth=25, caption="43"),
    
    # Connections
    to_connection("conv1", "pool1"),
    to_connection("pool1", "inc1"),
    to_connection("inc1", "pool2"),
    to_connection("pool2", "inc2"),
    to_connection("inc2", "gap"),
    to_connection("gap", "fc1"),
    to_connection("fc1", "soft1"),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
EOF

# V5: ResNet - 残差连接
cat > plotneuralnet/pyexamples/gtsrb_v5_resnet.py << 'EOF'
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # Input
    to_input('../examples/fcn8s/cats.jpg', to='(-3,0,0)', width=2, height=2),
    
    # Initial conv
    to_Conv(name='conv_init', s_filer=32, n_filer=32, offset="(0,0,0)", to="(0,0,0)", 
            width=2, height=30, depth=30, caption="Conv 32"),
    
    # ResBlock 1
    to_ConvConvRelu(name='res1', s_filer=32, n_filer=32, offset="(2,0,0)", to="(conv_init-east)", 
                    width=(2,2), height=30, depth=30, caption="ResBlock 32"),
    to_skip(of='conv_init', to='res1', pos=1.25),
    
    # ResBlock 2
    to_ConvConvRelu(name='res2', s_filer=64, n_filer=64, offset="(2,0,0)", to="(res1-east)", 
                    width=(4,4), height=15, depth=15, caption="ResBlock 64"),
    to_skip(of='res1', to='res2', pos=1.25),
    
    # ResBlock 3
    to_ConvConvRelu(name='res3', s_filer=128, n_filer=128, offset="(2,0,0)", to="(res2-east)", 
                    width=(6,6), height=7, depth=7, caption="ResBlock 128"),
    to_skip(of='res2', to='res3', pos=1.25),
    
    # GAP
    to_Pool(name="gap", offset="(2,0,0)", to="(res3-east)", 
            width=6, height=1, depth=1, opacity=0.5, caption="GAP"),
    
    # Dense
    to_ConvRes(name='fc1', s_filer=256, n_filer=256, offset="(2,0,0)", to="(gap-east)",
               width=3, height=30, depth=1, caption="FC 256"),
    
    # Output
    to_SoftMax(name="soft1", s_filer=43, offset="(2,0,0)", to="(fc1-east)", 
               width=1.5, height=25, depth=25, caption="43"),
    
    # Connections
    to_connection("conv_init", "res1"),
    to_connection("res1", "res2"),
    to_connection("res2", "res3"),
    to_connection("res3", "gap"),
    to_connection("gap", "fc1"),
    to_connection("fc1", "soft1"),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
EOF

# V6: Vision Transformer
cat > plotneuralnet/pyexamples/gtsrb_v6_vit.py << 'EOF'
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # Input
    to_input('../examples/fcn8s/cats.jpg', to='(-3,0,0)', width=2, height=2),
    
    # Patch embedding
    to_Conv(name='patch', s_filer=64, n_filer=64, offset="(0,0,0)", to="(0,0,0)", 
            width=1, height=30, depth=30, caption="Patch Embed"),
    
    # Transformer blocks
    to_ConvRes(name='trans1', s_filer=64, n_filer=64, offset="(2,0,0)", to="(patch-east)",
               width=3, height=30, depth=30, caption="Transformer 1"),
    to_ConvRes(name='trans2', s_filer=64, n_filer=64, offset="(1,0,0)", to="(trans1-east)",
               width=3, height=30, depth=30, caption="Transformer 2"),
    to_ConvRes(name='trans3', s_filer=64, n_filer=64, offset="(1,0,0)", to="(trans2-east)",
               width=3, height=30, depth=30, caption="Transformer 3"),
    
    # Classification head
    to_ConvRes(name='cls', s_filer=64, n_filer=64, offset="(2,0,0)", to="(trans3-east)",
               width=3, height=15, depth=1, caption="[CLS] Token"),
    
    # Output
    to_SoftMax(name="soft1", s_filer=43, offset="(2,0,0)", to="(cls-east)", 
               width=1.5, height=25, depth=25, caption="43"),
    
    # Connections
    to_connection("patch", "trans1"),
    to_connection("trans1", "trans2"),
    to_connection("trans2", "trans3"),
    to_connection("trans3", "cls"),
    to_connection("cls", "soft1"),
    
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
EOF

echo "✓ Professional scripts created"

# 5. 编译生成 PDF
echo ""
echo "======================================================================"
echo "Compiling diagrams..."
echo "======================================================================"

for script in gtsrb_v1_lenet gtsrb_v2_alexnet gtsrb_v3_vgg gtsrb_v4_googlenet gtsrb_v5_resnet gtsrb_v6_vit; do
    echo ""
    echo "Processing $script..."
    (cd plotneuralnet/pyexamples && python3 "${script}.py" && pdflatex -interaction=nonstopmode "${script}.tex" > /dev/null 2>&1)
    if [ -f "plotneuralnet/pyexamples/${script}.pdf" ]; then
        echo "✓ Generated ${script}.pdf"
    else
        echo "✗ Failed to generate ${script}.pdf"
    fi
done

# 6. 复制结果
mkdir -p architecture_diagrams_latex
cp plotneuralnet/pyexamples/gtsrb_*.pdf architecture_diagrams_latex/ 2>/dev/null || true

# 7. 转换为 PNG（如果有 ImageMagick）
if command -v convert &> /dev/null; then
    echo ""
    echo "Converting to PNG..."
    for pdf in architecture_diagrams_latex/*.pdf; do
        [ -f "$pdf" ] && convert -density 300 "$pdf" -quality 90 "${pdf%.pdf}.png"
    done
fi

echo ""
echo "======================================================================"
echo "Complete! Diagrams saved in: architecture_diagrams_latex/"
echo "======================================================================"
ls -lh architecture_diagrams_latex/ 2>/dev/null || echo "No files generated"
