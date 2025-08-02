import os
import shutil
import glob
from datetime import datetime
from PIL import Image, ImageOps
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from transformers import ViTImageProcessor, ViTModel
from transformers import AutoFeatureExtractor, ResNetForImageClassification

class PhotoAI:
    def __init__(self, input_dir, good_dir='photo-good', bad_dir='photo-bad'):
        # 初始化目录
        self.input_dir = input_dir
        self.good_dir = good_dir
        self.bad_dir = bad_dir
        self._create_dirs()
        
        # 初始化模型（首次运行会自动下载）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_models()
        
        # 配置参数
        self.time_threshold = 300  # 5分钟（秒）
        self.cluster_eps = 0.4    # 聚类敏感度
        
        self.min_photos_to_cluster = 2  # 新增：最小聚类照片数

    def _create_dirs(self):
        """创建输出目录"""
        os.makedirs(self.good_dir, exist_ok=True)
        os.makedirs(self.bad_dir, exist_ok=True)

    def _init_models(self):
        """初始化所有需要的模型"""
        # 清晰度评估模型
        self.sharpness_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        self.sharpness_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(self.device)
        
        # 特征提取模型
        self.feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.feature_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(self.device)

    def process_pipeline(self):
        """完整处理流程（添加进度报告）"""
        # 第一阶段：清晰度分类
        print("Stage 1: Classifying sharpness...")
        image_paths = self._get_image_paths()
        print(f"Found {len(image_paths)} total photos")
        
        sharpness_results = self.classify_sharpness(image_paths)
        good_count = sum(sharpness_results.values())
        print(f"Classified {good_count} good / {len(image_paths)-good_count} bad photos")

        # 第二阶段：场景聚类
        print("\nStage 2: Clustering scenes...")
        valid_paths = [p for p, is_good in sharpness_results.items() if is_good]
        
        if len(valid_paths) < self.min_photos_to_cluster:
            print("Skipping clustering due to insufficient good photos")
            return

        clusters = self.cluster_photos(valid_paths)
        print(f"Created {len(clusters)} scene clusters")

        # 第三阶段：优选照片
        print("\nStage 3: Selecting best photos...")
        self.select_best_photos(clusters)
        print("Process completed successfully")

    # ------------------ 核心功能方法 ------------------
    def classify_sharpness(self, image_paths):
        """分类清晰/模糊照片"""
        results = {}
        for path in image_paths:
            try:
                image = Image.open(path)
                inputs = self.sharpness_extractor(image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.sharpness_model(**inputs)
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                sharp_prob = probs[0][1].item()  # 假设第二类是清晰
                
                dest = self.good_dir if sharp_prob > 0.7 else self.bad_dir
                shutil.copy(path, os.path.join(dest, os.path.basename(path)))
                results[path] = (sharp_prob > 0.7)
                
            except Exception as e:
                print(f"Skipping {path} due to error: {str(e)}")
                results[path] = False
        return results

    def cluster_photos(self, image_paths):
        """聚类相似场景照片（空输入保护版本）"""
        # 空输入检查
        if len(image_paths) < self.min_photos_to_cluster:
            print(f"Not enough photos to cluster ({len(image_paths)} available)")
            return {}

        features, timestamps = [], []
        
        # 特征提取
        for path in image_paths:
            img = Image.open(path)
            time = self._get_capture_time(path)
            
            # 提取ViT特征（保持二维）
            inputs = self.feature_extractor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.feature_model(**inputs)
            feature = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # 形状 [1, 768]
            features.append(feature[0])  # 转换为1D数组 [768,]
            
            timestamps.append(time.timestamp() if time else os.path.getmtime(path))

        # 数据验证
        if len(features) == 0 or len(timestamps) == 0:
            print("Feature extraction failed for all photos")
            return {}
        
        # 转换为NumPy数组并调整维度
        try:
            features = np.array(features)
            timestamps = np.array(timestamps)[:, np.newaxis]
            
            # 新增：时间归一化保护
            if timestamps.size > 0:
                timestamps = (timestamps - timestamps.min()) / self.time_threshold
            else:
                timestamps = np.zeros_like(features[:, [0]])  # 用零填充时间维度

            X = np.hstack([features, timestamps])
        except Exception as e:
            print(f"Feature combination failed: {str(e)}")
            return {}
        
        # 保持DBSCAN聚类逻辑不变.
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=2).fit(X)
        
        # 构建分组
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(image_paths[idx])
        
        return clusters

    def select_best_photos(self, clusters, num_select=2):
        """从每个聚类中选择最佳照片"""
        for cluster_id, paths in clusters.items():
            if cluster_id == -1:  # 忽略噪声点
                continue
                
            scores = []
            for path in paths:
                score = self._calculate_quality_score(path)
                scores.append((path, score))
            
            # 按分数排序并选择
            sorted_photos = sorted(scores, key=lambda x: x[1], reverse=True)
            for path, score in sorted_photos[:num_select]:
                best_dir = os.path.join(self.good_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                shutil.copy(path, os.path.join(best_dir, os.path.basename(path)))

    def _get_image_paths(self):
        """获取所有支持的图片路径"""
        # extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        # return [p for ext in extensions for p in glob.glob(os.path.join(self.input_dir, ext))]
        image_dir = self.input_dir
        return [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def _get_capture_time(self, path):
        """尝试获取拍摄时间"""
        try:
            exif = Image.open(path)._getexif()
            if exif and 36867 in exif:  # DateTimeOriginal
                return datetime.strptime(exif[36867], "%Y:%m:%d %H:%M:%S")
        except:
            pass
        return None

    def _calculate_quality_score(self, path):
        """综合质量评分"""
        img = Image.open(path)
        
        # 清晰度评分
        inputs = self.sharpness_extractor(img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sharpness_model(**inputs)
        sharp_prob = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][1].item()
        
        # 构图评分
        inputs = self.feature_extractor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.feature_model(**inputs)
        composition_score = outputs.last_hidden_state.mean().item()
        
        return sharp_prob * 0.7 + composition_score * 0.3

if __name__ == "__main__":
    image_dir = "/Users/tigerhu/projects/photo-ai/photos"
    processor = PhotoAI(input_dir=image_dir)
    processor.process_pipeline()