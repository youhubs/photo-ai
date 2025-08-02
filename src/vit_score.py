
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

def assess_quality_vit(image_path):
    # 加载模型和处理器
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取所有类别及其概率
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    classes = model.config.id2label
    
    # 定义质量相关标签（根据ImageNet类别）
    quality_keywords = {
        "sharp": ["digital camera", "sports car", "watch"],
        "blurry": ["blur", "fog", "smoke"],
        "well_composed": ["altar", "painting", "vase"]
    }
    
    # 计算质量分数
    quality_scores = {"sharpness": 0.0, "blurry": 0.0, "composition": 0.0}
    for i, score in enumerate(probs):
        label = classes[i].lower()
        # 锐度评分
        if any(kw in label for kw in quality_keywords["sharp"]):
            quality_scores["sharpness"] += score.item()
        # 模糊减分
        if any(kw in label for kw in quality_keywords["blurry"]):
            quality_scores["blurry"] -= score.item()
        # 构图评分
        if any(kw in label for kw in quality_keywords["well_composed"]):
            quality_scores["composition"] += score.item()
    
    return quality_scores

# 使用示例
import os
image_dir = "/Users/tigerhu/projects/photo-ai/photos"  # Source directory
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for image in image_files:
    vit_scores = assess_quality_vit(image)
    print(f"ViT Quality Scores: {vit_scores}")