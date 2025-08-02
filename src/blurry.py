
from transformers import ResNetForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch

# 训练一个二分类模型（清晰 vs 模糊）

# 加载预训练的模糊检测模型（ResNet-50微调版）
model_name = "microsoft/resnet-50"
processor = AutoFeatureExtractor.from_pretrained(model_name)
model = ResNetForImageClassification.from_pretrained(model_name)

def assess_blur_resnet(image_path):
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 假设模型第0类是"blurry"，第1类是"sharp"
    # （需根据实际微调模型调整）
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return {
        "blur_score": probs[0][0].item(),
        "sharp_score": probs[0][1].item()
    }

# 使用示例
import os
image_dir = "/Users/tigerhu/projects/photo-ai/photos"  # Source directory
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for image in image_files:
    result = assess_blur_resnet(image)
    print(f"Blur Score (ResNet): {result['blur_score']:.2f}")