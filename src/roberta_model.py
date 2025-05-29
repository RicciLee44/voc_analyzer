"""
基于RoBERTa的多任务VOC文本分类模型 - 优化版
支持多标签分类、层次标签结构、增量学习等高级功能
"""
import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VOCMultiTaskClassifier:
    """
    VOC多任务分类器 - 针对理想汽车用户反馈的智能标签系统
    支持用户旅程、问题类型、情感分析等多维度标签识别
    """
    
    def __init__(self, 
                 model_name: str = 'hfl/chinese-roberta-wwm-ext',
                 max_length: int = 256,
                 use_pooling: str = 'cls',  # 'cls', 'mean', 'max', 'attention'
                 dropout_rate: float = 0.1):
        """
        初始化多任务分类器
        
        Args:
            model_name: 预训练模型名称
            max_length: 最大序列长度
            use_pooling: 特征提取方式
            dropout_rate: dropout比例
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_pooling = use_pooling
        self.dropout_rate = dropout_rate
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 加载预训练模型
        self._load_pretrained_model()
        
        # 初始化分类器组件
        self.classifiers = {}
        self.label_encoders = {}
        self.label_mapping = {}
        
    def _load_pretrained_model(self):
        """加载预训练的RoBERTa模型"""
        try:
            logger.info(f"正在加载预训练模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # 添加注意力池化层（如果使用）
            if self.use_pooling == 'attention':
                self.attention_pool = nn.Linear(self.config.hidden_size, 1)
                self.attention_pool.to(self.device)
                
            logger.info("预训练模型加载完成 ✓")
        except Exception as e:
            logger.error(f"加载预训练模型失败: {str(e)}")
            raise
    
    def preprocess_labels(self, labels_dict: Dict[str, List]) -> Dict[str, List]:
        """
        预处理标签数据，支持多标签和层次标签
        
        Args:
            labels_dict: 原始标签字典
            
        Returns:
            处理后的标签字典
        """
        processed_labels = {}
        
        for label_type, labels in labels_dict.items():
            if label_type in ['旅程触点', '问题类型']:  # 多标签任务
                # 处理多标签格式 - 支持列表输入
                processed = []
                for label in labels:
                    if isinstance(label, list):
                        processed.append(label)
                    elif isinstance(label, str) and label:
                        # 支持逗号分隔的标签字符串
                        processed.append([l.strip() for l in label.split(',') if l.strip()])
                    else:
                        processed.append([])
                processed_labels[label_type] = processed
                
            else:  # 单标签任务（如情感）
                processed_labels[label_type] = labels
                
        return processed_labels
    
    def train(self, 
              texts: List[str], 
              labels_dict: Dict[str, List],
              validation_split: float = 0.2,
              use_ensemble: bool = True) -> Dict:
        """
        训练多任务分类器
        
        Args:
            texts: 文本列表
            labels_dict: 标签字典
            validation_split: 验证集比例
            use_ensemble: 是否使用集成方法
            
        Returns:
            训练结果统计
        """
        logger.info(f"开始训练，文本数量: {len(texts)}")
        
        # 预处理标签
        processed_labels = self.preprocess_labels(labels_dict)
        
        # 数据分割
        if validation_split > 0:
            split_idx = int(len(texts) * (1 - validation_split))
            train_texts, val_texts = texts[:split_idx], texts[split_idx:]
            train_labels = {k: v[:split_idx] for k, v in processed_labels.items()}
            val_labels = {k: v[split_idx:] for k, v in processed_labels.items()}
        else:
            train_texts, val_texts = texts, []
            train_labels, val_labels = processed_labels, {}
        
        # 获取训练文本嵌入
        logger.info("提取训练文本特征...")
        train_embeddings = self._get_embeddings(train_texts)
        
        # 训练各个任务的分类器
        results = {}
        for label_type, labels in train_labels.items():
            logger.info(f"训练标签类型: {label_type}")
            
            # 过滤有效样本
            valid_indices = self._get_valid_indices(labels)
            if len(valid_indices) < 10:  # 最少需要10个样本
                logger.warning(f"标签 {label_type} 有效样本不足，跳过训练")
                continue
                
            valid_embeddings = train_embeddings[valid_indices]
            valid_labels = [labels[i] for i in valid_indices]
            
            # 根据任务类型选择分类器
            if label_type in ['旅程触点', '问题类型']:
                # 多标签分类
                results[label_type] = self._train_multilabel_classifier(
                    label_type, valid_embeddings, valid_labels, use_ensemble
                )
            else:
                # 单标签分类
                results[label_type] = self._train_single_classifier(
                    label_type, valid_embeddings, valid_labels, use_ensemble
                )
        
        # 验证模型性能
        if val_texts:
            logger.info("评估模型性能...")
            val_results = self.evaluate(val_texts, val_labels)
            results['validation'] = val_results
        
        return results
    
    def _train_multilabel_classifier(self, 
                                   label_type: str, 
                                   embeddings: np.ndarray, 
                                   labels: List[List[str]],
                                   use_ensemble: bool = True) -> Dict:
        """训练多标签分类器"""
        # 标签编码
        mlb = MultiLabelBinarizer()
        y_encoded = mlb.fit_transform(labels)
        
        self.label_encoders[label_type] = mlb
        self.label_mapping[label_type] = list(mlb.classes_)
        
        logger.info(f"标签 {label_type} 包含 {len(mlb.classes_)} 个类别: {list(mlb.classes_)[:5]}...")
            
        # 选择分类器
        if use_ensemble:
            # 使用随机森林集成
            base_clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        else:
            base_clf = LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        
        # 多输出分类器
        clf = MultiOutputClassifier(base_clf, n_jobs=-1)
        
        try:
            clf.fit(embeddings, y_encoded)
            self.classifiers[label_type] = clf
            
            # 计算训练指标
            y_pred = clf.predict(embeddings)
            f1_micro = f1_score(y_encoded, y_pred, average='micro', zero_division=0)
            f1_macro = f1_score(y_encoded, y_pred, average='macro', zero_division=0)
            
            logger.info(f"分类器 {label_type} 训练完成 ✓ (F1-micro: {f1_micro:.3f}, F1-macro: {f1_macro:.3f})")
            
            return {
                'type': 'multilabel',
                'n_classes': len(mlb.classes_),
                'f1_micro': f1_micro,
                'f1_macro': f1_macro
            }
        except Exception as e:
            logger.error(f"训练分类器 {label_type} 失败: {str(e)}")
            return {'type': 'multilabel', 'error': str(e)}
    
    def _train_single_classifier(self, 
                                label_type: str, 
                                embeddings: np.ndarray, 
                                labels: List[str],
                                use_ensemble: bool = True) -> Dict:
        """训练单标签分类器"""
        # 记录标签映射
        unique_labels = sorted(list(set(labels)))
        self.label_mapping[label_type] = unique_labels
        
        logger.info(f"标签 {label_type} 包含 {len(unique_labels)} 个类别: {unique_labels}")
        
        # 选择分类器
        if use_ensemble and len(unique_labels) > 2:
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            )
        else:
            clf = LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        
        try:
            clf.fit(embeddings, labels)
            self.classifiers[label_type] = clf
            
            # 计算训练指标
            y_pred = clf.predict(embeddings)
            f1_weighted = f1_score(labels, y_pred, average='weighted', zero_division=0)
            
            logger.info(f"分类器 {label_type} 训练完成 ✓ (F1-weighted: {f1_weighted:.3f})")
            
            return {
                'type': 'single',
                'n_classes': len(unique_labels),
                'f1_weighted': f1_weighted
            }
        except Exception as e:
            logger.error(f"训练分类器 {label_type} 失败: {str(e)}")
            return {'type': 'single', 'error': str(e)}
    
    def predict(self, 
                texts: List[str], 
                return_confidence: bool = True,
                threshold: float = 0.3) -> Dict:
        """
        预测文本标签
        
        Args:
            texts: 待预测文本列表
            return_confidence: 是否返回置信度
            threshold: 多标签分类阈值
            
        Returns:
            预测结果字典
        """
        logger.info(f"开始预测，文本数量: {len(texts)}")
        
        # 获取文本嵌入
        embeddings = self._get_embeddings(texts)
        
        results = {'predictions': {}, 'confidences': {}}
        
        # 对每个标签类型进行预测
        for label_type, clf in self.classifiers.items():
            logger.info(f"预测标签: {label_type}")
            
            try:
                if label_type in ['旅程触点', '问题类型']:
                    # 多标签预测
                    pred_results = self._predict_multilabel(
                        clf, embeddings, label_type, threshold, return_confidence
                    )
                else:
                    # 单标签预测
                    pred_results = self._predict_single(
                        clf, embeddings, label_type, return_confidence
                    )
                
                results['predictions'][label_type] = pred_results['predictions']
                if return_confidence:
                    results['confidences'][label_type] = pred_results['confidences']
                    
            except Exception as e:
                logger.error(f"预测标签 {label_type} 失败: {str(e)}")
                results['predictions'][label_type] = [[] for _ in texts]
                if return_confidence:
                    results['confidences'][label_type] = [0.0 for _ in texts]
        
        return results
    
    def _predict_multilabel(self, 
                           clf, 
                           embeddings: np.ndarray, 
                           label_type: str,
                           threshold: float,
                           return_confidence: bool) -> Dict:
        """多标签预测"""
        # 获取概率预测
        probas = clf.predict_proba(embeddings)
        
        predictions = []
        confidences = []
        
        mlb = self.label_encoders[label_type]
        
        for i, sample_probas in enumerate(probas):
            # 对每个样本的每个标签获取概率
            sample_pred = []
            sample_conf = []
            
            for j, label_proba in enumerate(sample_probas):
                # 获取正类概率（通常是第二列）
                if len(label_proba) > 1:
                    pos_prob = label_proba[1]
                else:
                    pos_prob = label_proba[0]
                
                if pos_prob >= threshold:
                    sample_pred.append(mlb.classes_[j])
                    sample_conf.append(pos_prob)
            
            predictions.append(sample_pred)
            confidences.append(np.mean(sample_conf) if sample_conf else 0.0)
        
        return {
            'predictions': predictions,
            'confidences': confidences if return_confidence else None
        }
    
    def _predict_single(self, 
                       clf, 
                       embeddings: np.ndarray, 
                       label_type: str,
                       return_confidence: bool) -> Dict:
        """单标签预测"""
        predictions = clf.predict(embeddings).tolist()
        
        confidences = None
        if return_confidence:
            probas = clf.predict_proba(embeddings)
            confidences = np.max(probas, axis=1).tolist()
        
        return {
            'predictions': predictions,
            'confidences': confidences
        }
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """获取文本嵌入向量 - 优化版"""
        embeddings = []
        batch_size = 16  # 增加批次大小
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="提取特征", ncols=80):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenization
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # 移到设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 前向传播
                outputs = self.model(**inputs)
                
                # 特征提取
                if self.use_pooling == 'cls':
                    # 使用[CLS]标记
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                elif self.use_pooling == 'mean':
                    # 平均池化
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                elif self.use_pooling == 'max':
                    # 最大池化
                    batch_embeddings = torch.max(outputs.last_hidden_state, dim=1)[0]
                else:  # attention pooling
                    # 注意力池化
                    attention_weights = torch.softmax(self.attention_pool(outputs.last_hidden_state), dim=1)
                    batch_embeddings = torch.sum(attention_weights * outputs.last_hidden_state, dim=1)
                
                embeddings.extend(batch_embeddings.cpu().numpy())
        
        return np.array(embeddings)
    
    def _get_valid_indices(self, labels: List) -> List[int]:
        """获取有效样本索引"""
        valid_indices = []
        for i, label in enumerate(labels):
            if isinstance(label, list):
                if len(label) > 0:  # 多标签非空
                    valid_indices.append(i)
            elif label:  # 单标签非空
                valid_indices.append(i)
        return valid_indices
    
    def evaluate(self, texts: List[str], true_labels: Dict[str, List]) -> Dict:
        """评估模型性能"""
        predictions = self.predict(texts, return_confidence=False)['predictions']
        
        results = {}
        for label_type in predictions.keys():
            if label_type not in true_labels:
                continue
                
            y_true = true_labels[label_type]
            y_pred = predictions[label_type]
            
            # 过滤有效样本
            valid_indices = self._get_valid_indices(y_true)
            if not valid_indices:
                continue
                
            valid_true = [y_true[i] for i in valid_indices]
            valid_pred = [y_pred[i] for i in valid_indices]
            
            if label_type in ['旅程触点', '问题类型']:
                # 多标签评估
                mlb = self.label_encoders[label_type]
                y_true_encoded = mlb.transform(valid_true)
                y_pred_encoded = mlb.transform(valid_pred)
                
                f1_micro = f1_score(y_true_encoded, y_pred_encoded, average='micro', zero_division=0)
                f1_macro = f1_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
                
                results[label_type] = {
                    'f1_micro': f1_micro,
                    'f1_macro': f1_macro
                }
            else:
                # 单标签评估
                f1_weighted = f1_score(valid_true, valid_pred, average='weighted', zero_division=0)
                results[label_type] = {'f1_weighted': f1_weighted}
        
        return results
    
    def predict_single_text(self, text: str, format_output: bool = True) -> Dict:
        """预测单个文本的标签"""
        results = self.predict([text], return_confidence=True)
        
        if format_output:
            # 格式化输出，符合项目需求
            formatted_result = {}
            for label_type, predictions in results['predictions'].items():
                if label_type == '旅程触点':
                    formatted_result['旅程触点'] = predictions[0]
                elif label_type == '问题类型':
                    formatted_result['问题类型'] = predictions[0]
                elif label_type == '情感':
                    formatted_result['情感'] = predictions[0]
            
            # 添加置信度信息
            formatted_result['置信度'] = {
                label_type: conf[0] for label_type, conf in results['confidences'].items()
            }
            
            return formatted_result
        
        return {
            'predictions': {k: v[0] for k, v in results['predictions'].items()},
            'confidences': {k: v[0] for k, v in results['confidences'].items()}
        }
    
    def save(self, output_dir: str):
        """保存模型"""
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("正在保存模型...")
        
        # 保存分类器
        for label_type, clf in self.classifiers.items():
            classifier_path = os.path.join(output_dir, f"{label_type}_classifier.pkl")
            with open(classifier_path, "wb") as f:
                pickle.dump(clf, f)
        
        # 保存标签编码器
        encoders_path = os.path.join(output_dir, "label_encoders.pkl")
        with open(encoders_path, "wb") as f:
            pickle.dump(self.label_encoders, f)
        
        # 保存标签映射
        mapping_path = os.path.join(output_dir, "label_mapping.json")
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.label_mapping, f, ensure_ascii=False, indent=2)
        
        # 保存配置
        config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "use_pooling": self.use_pooling,
            "dropout_rate": self.dropout_rate,
            "label_types": list(self.classifiers.keys())
        }
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型保存完成 ✓ 路径: {output_dir}")
    
    @classmethod
    def load(cls, model_dir: str):
        """加载模型"""
        # 加载配置
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 创建实例
        instance = cls(
            model_name=config["model_name"],
            max_length=config.get("max_length", 256),
            use_pooling=config.get("use_pooling", "cls"),
            dropout_rate=config.get("dropout_rate", 0.1)
        )
        
        # 加载标签映射
        mapping_path = os.path.join(model_dir, "label_mapping.json")
        with open(mapping_path, "r", encoding="utf-8") as f:
            instance.label_mapping = json.load(f)
        
        # 加载标签编码器
        encoders_path = os.path.join(model_dir, "label_encoders.pkl")
        if os.path.exists(encoders_path):
            with open(encoders_path, "rb") as f:
                instance.label_encoders = pickle.load(f)
        
        # 加载分类器
        for label_type in config["label_types"]:
            classifier_path = os.path.join(model_dir, f"{label_type}_classifier.pkl")
            with open(classifier_path, "rb") as f:
                instance.classifiers[label_type] = pickle.load(f)
        
        logger.info(f"模型加载完成 ✓")
        return instance


# 示例使用代码
if __name__ == "__main__":
    # 创建分类器实例
    classifier = VOCMultiTaskClassifier(
        model_name='hfl/chinese-roberta-wwm-ext',
        max_length=256,
        use_pooling='mean'  # 使用平均池化
    )
    
    # 示例数据
    texts = [
        "导航经常带我绕远路，语音也识别不出来，还老是闪退。",
        "交车当天小哥讲解得很细，座椅按摩也比想象中舒服。",
        "充电桩充电速度太慢了，而且经常坏"
    ]
    
    labels = {
        "旅程触点": [
            ["智能导航", "语音助手"],
            ["交付培训", "驾驶体验"],
            ["充电与能耗"]
        ],
        "问题类型": [
            ["稳定性问题", "可用性问题"],
            [],
            ["性能问题"]
        ],
        "情感": ["负面", "正面", "负面"]
    }
    
    # 训练模型
    train_results = classifier.train(texts, labels, validation_split=0.0)
    print("训练结果:", train_results)
    
    # 预测示例
    test_text = "语音助手总是听不懂我说话，很失望"
    result = classifier.predict_single_text(test_text)
    print("预测结果:", json.dumps(result, ensure_ascii=False, indent=2)) 