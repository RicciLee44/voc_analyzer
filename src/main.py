"""
VOC自动打标签系统 - 主程序 v3.0
"""
import os
import sys
import argparse
import pandas as pd
import shutil
import time
import json
import platform
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import warnings
import logging
warnings.filterwarnings('ignore')

# 获取项目根目录
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# 导入自定义模块
try:
    from roberta_model import VOCMultiTaskClassifier
    from text_processor import TextProcessor
    from data_loader import DataLoader
    from visualizer import VOCVisualizer
    print("✅ 新模块加载成功")
except ImportError as e:
    # 兼容性处理：如果新模块不存在，尝试导入旧版本
    print(f"⚠️ 导入新模块失败，尝试使用兼容模式: {e}")
    try:
        from roberta_model import RobertaClassifier as VOCMultiTaskClassifier
        from text_processor import TextProcessor
        from data_loader import DataLoader
        from visualizer import Visualizer as VOCVisualizer
        print("✅ 兼容模式加载成功")
    except ImportError as e2:
        print(f"❌ 模块导入失败: {e2}")
        sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, 'logs', 'main.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 颜色输出工具类
class ColorOutput:
    """彩色输出工具类，兼容不同终端"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    @staticmethod
    def is_color_supported():
        """检测终端是否支持彩色输出"""
        if platform.system() == 'Windows':
            return int(platform.version().split('.')[0]) >= 10
        return True
    
    @classmethod
    def print(cls, text, color=None, bold=False):
        """打印彩色文本"""
        if not cls.is_color_supported():
            print(text)
            return
            
        style = ''
        if color:
            style += color
        if bold:
            style += cls.BOLD
            
        if style:
            print(f"{style}{text}{cls.RESET}")
        else:
            print(text)
    
    @classmethod
    def success(cls, text):
        cls.print(f"✅ {text}", cls.GREEN)
    
    @classmethod
    def error(cls, text):
        cls.print(f"❌ {text}", cls.RED)
    
    @classmethod
    def warning(cls, text):
        cls.print(f"⚠️ {text}", cls.YELLOW)
    
    @classmethod
    def info(cls, text):
        cls.print(f"ℹ️ {text}", cls.CYAN)
    
    @classmethod
    def title(cls, text):
        cls.print(f"\n{'='*60}", cls.CYAN, bold=True)
        cls.print(f"{text.center(60)}", cls.CYAN, bold=True)
        cls.print(f"{'='*60}", cls.CYAN, bold=True)

class PerformanceMonitor:
    """性能监控器"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.checkpoints = {}
        
    def start(self, name: str = "main"):
        self.start_time = time.time()
        self.checkpoints[name] = {'start': self.start_time}
        
    def checkpoint(self, name: str):
        current_time = time.time()
        if name not in self.checkpoints:
            self.checkpoints[name] = {}
        self.checkpoints[name]['end'] = current_time
        if self.start_time:
            self.checkpoints[name]['duration'] = current_time - self.start_time
        
    def end(self):
        self.end_time = time.time()
        return self.get_duration()
        
    def get_duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
        
    def format_duration(self, duration=None):
        if duration is None:
            duration = self.get_duration()
        
        if duration < 60:
            return f"{duration:.2f}秒"
        elif duration < 3600:
            return f"{duration//60:.0f}分{duration%60:.0f}秒"
        else:
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            return f"{hours:.0f}小时{minutes:.0f}分"
    
    def get_report(self):
        """获取性能报告"""
        report = []
        for name, times in self.checkpoints.items():
            if 'duration' in times:
                report.append(f"  {name}: {self.format_duration(times['duration'])}")
        return "\n".join(report)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VOC自动打标签系统 v3.0')
    
    # 通用参数
    parser.add_argument('--mode', type=str, 
                        choices=['train', 'predict', 'batch', 'interactive', 'initialize', 'evaluate', 'demo'], 
                        default='interactive', help='运行模式')
    parser.add_argument('--model_dir', type=str, default=os.path.join(DATA_DIR, 'models/latest'),
                        help='模型保存/加载目录')
    parser.add_argument('--config_file', type=str, help='配置文件路径')
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')
    
    # 训练参数
    parser.add_argument('--train_file', type=str, help='训练数据文件路径')
    parser.add_argument('--text_column', type=str, default='text', help='文本列名')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--validation_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--cross_validation', action='store_true', help='是否执行交叉验证')
    parser.add_argument('--use_ensemble', action='store_true', help='是否使用集成方法')
    parser.add_argument('--max_length', type=int, default=256, help='最大序列长度')
    parser.add_argument('--pooling_strategy', type=str, default='mean', 
                        choices=['cls', 'mean', 'max', 'attention'], help='特征提取策略')
    parser.add_argument('--enable_text_segmentation', action='store_true', help='是否启用文本分词')
    
    # 预测参数
    parser.add_argument('--input_file', type=str, help='输入文件路径')
    parser.add_argument('--output_file', type=str, help='输出文件路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
    parser.add_argument('--threshold', type=float, default=0.3, help='多标签分类阈值')
    parser.add_argument('--generate_report', action='store_true', help='是否生成详细报告')
    parser.add_argument('--enable_cache', action='store_true', help='是否启用数据缓存')
    
    return parser.parse_args()

def load_config(config_file):
    """加载配置文件"""
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"配置文件加载成功: {config_file}")
                return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {str(e)}")
    return {}

def enhanced_train_model(args, interactive_mode=False):
    """增强版模型训练"""
    ColorOutput.title("VOC智能模型训练系统")
    monitor = PerformanceMonitor()
    monitor.start("training")
    
    try:
        # 加载配置
        config = load_config(args.config_file) if args.config_file else {}
        
        # 初始化组件
        ColorOutput.info("初始化系统组件...")
        monitor.checkpoint("init_components")
        
        # 数据加载器 - 使用新版本
        data_loader = DataLoader(
            cache_dir=os.path.join(DATA_DIR, 'cache'),
            enable_cache=getattr(args, 'enable_cache', True)
        )
        
        # 文本处理器 - 使用新版本
        text_processor = TextProcessor(
            enable_userdict=True,
            stopwords_path=config.get('stopwords_path')
        )
        
        # 加载训练数据
        ColorOutput.info("加载训练数据...")
        monitor.checkpoint("load_data")
        
        if args.train_file and os.path.exists(args.train_file):
            texts, labels_dict, df = data_loader.load_file(
                args.train_file, 
                text_column=args.text_column
            )
        else:
            ColorOutput.info("使用VOC示例数据进行训练")
            texts, labels_dict = data_loader.load_examples("voc")
        
        if not texts:
            ColorOutput.error("没有可用的训练数据")
            return None
        
        # 数据质量分析
        ColorOutput.info("分析数据质量...")
        quality_report = text_processor.analyze_text_quality(texts)
        ColorOutput.info(f"数据质量报告: 总文本{quality_report.get('total_texts', 0)}条, "
                        f"有效率{100-quality_report.get('empty_rate', 0):.1f}%")
    
        # 文本预处理
        ColorOutput.info("执行文本预处理...")
        monitor.checkpoint("preprocess_text")
        
        processed_texts = text_processor.batch_process(
            texts,
            enable_segmentation=getattr(args, 'enable_text_segmentation', False),
            show_progress=True
        )
        
        # 提取关键词
        if interactive_mode:
            ColorOutput.info("提取关键词...")
            keywords = text_processor.extract_keywords(processed_texts, top_k=10)
            ColorOutput.info(f"关键词: {', '.join(keywords.keys())}")
        
        # 初始化分类器
        ColorOutput.info("初始化智能分类器...")
        monitor.checkpoint("init_classifier")
        
        classifier_config = {
            'model_name': config.get('model_name', 'hfl/chinese-roberta-wwm-ext'),
            'max_length': args.max_length,
            'use_pooling': args.pooling_strategy,
            'dropout_rate': config.get('dropout_rate', 0.1)
        }
        
        classifier = VOCMultiTaskClassifier(**classifier_config)
        
        # 训练模型
        ColorOutput.info("开始智能训练...")
        monitor.checkpoint("model_training")
        
        train_results = classifier.train(
            processed_texts,
            labels_dict,
            validation_split=args.validation_split,
            use_ensemble=args.use_ensemble
        )
        
        # 显示训练结果
        ColorOutput.success("模型训练完成!")
        for label_type, result in train_results.items():
            if label_type == 'validation':
                continue
            if 'error' not in result:
                if result['type'] == 'multilabel':
                    ColorOutput.info(f"  📊 {label_type}: F1-micro={result.get('f1_micro', 0):.3f}, "
                                   f"F1-macro={result.get('f1_macro', 0):.3f}")
                else:
                    ColorOutput.info(f"  📊 {label_type}: F1-weighted={result.get('f1_weighted', 0):.3f}")
        
        # 交叉验证
        if args.cross_validation or (interactive_mode and 
                                    input("\n🔄 是否执行交叉验证以获得更准确的评估? (y/n): ").lower() == 'y'):
            ColorOutput.info("执行5折交叉验证...")
            monitor.checkpoint("cross_validation")
            
            cv_results = perform_cross_validation(
                processed_texts, labels_dict, classifier_config, args.use_ensemble
            )
            
            ColorOutput.success("交叉验证结果:")
            for label_type, scores in cv_results.items():
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                ColorOutput.info(f"  📈 {label_type}: {mean_score:.4f} ± {std_score:.4f}")
            
            train_results['cross_validation'] = cv_results
        
        # 保存模型
        ColorOutput.info("保存训练模型...")
        monitor.checkpoint("save_model")
        
        os.makedirs(args.model_dir, exist_ok=True)
        classifier.save(args.model_dir)
        
        # 保存训练报告
        save_training_report(train_results, args.model_dir, text_processor.get_stats())
        
        # 保存预处理器配置
        processor_config = {
            'enable_segmentation': getattr(args, 'enable_text_segmentation', False),
            'stats': text_processor.get_stats()
        }
        
        with open(os.path.join(args.model_dir, 'text_processor_config.json'), 'w', encoding='utf-8') as f:
            json.dump(processor_config, f, ensure_ascii=False, indent=2)
        
        # 性能总结
        duration = monitor.end()
        ColorOutput.success(f"🎉 训练完成! 总用时: {monitor.format_duration(duration)}")
        ColorOutput.success(f"📁 模型已保存到: {args.model_dir}")
        
        if interactive_mode:
            print(f"\n⏱️ 性能报告:")
            print(monitor.get_report())
        
        return classifier
        
    except Exception as e:
        ColorOutput.error(f"训练失败: {str(e)}")
        logger.error(f"训练失败: {str(e)}", exc_info=True)
        return None

def perform_cross_validation(texts, labels_dict, classifier_config, use_ensemble=True, n_splits=5):
    """执行交叉验证"""
    # 使用第一个标签类型进行分层抽样
    first_label_type = list(labels_dict.keys())[0]
    first_labels = labels_dict[first_label_type]
    
    # 为分层抽样准备标签
    if isinstance(first_labels[0], list) or ',' in str(first_labels[0]):
        # 多标签情况：使用第一个标签或空标签
        stratify_labels = []
        for label in first_labels:
            if isinstance(label, list):
                stratify_labels.append(label[0] if label else 'empty')
            else:
                label_list = [l.strip() for l in str(label).split(',') if l.strip()]
                stratify_labels.append(label_list[0] if label_list else 'empty')
    else:
        stratify_labels = first_labels
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = {label_type: [] for label_type in labels_dict.keys()}
        
    for fold, (train_idx, val_idx) in enumerate(kf.split(texts, stratify_labels)):
        ColorOutput.info(f"  📊 第{fold+1}/{n_splits}折验证中...")
            
        # 准备折叠数据
        fold_train_texts = [texts[i] for i in train_idx]
        fold_val_texts = [texts[i] for i in val_idx]
        fold_train_labels = {k: [v[i] for i in train_idx] for k, v in labels_dict.items()}
        fold_val_labels = {k: [v[i] for i in val_idx] for k, v in labels_dict.items()}
            
            # 训练模型
        fold_classifier = VOCMultiTaskClassifier(**classifier_config)
        fold_classifier.train(fold_train_texts, fold_train_labels, validation_split=0, use_ensemble=use_ensemble)
            
        # 评估
        fold_results = fold_classifier.evaluate(fold_val_texts, fold_val_labels)
        
        # 记录分数
        for label_type, metrics in fold_results.items():
            if 'f1_micro' in metrics:
                cv_scores[label_type].append(metrics['f1_micro'])
            elif 'f1_weighted' in metrics:
                cv_scores[label_type].append(metrics['f1_weighted'])
    
    return cv_scores

def enhanced_predict_batch(args):
    """增强版批量预测"""
    ColorOutput.title("VOC智能批量分析系统")
    monitor = PerformanceMonitor()
    monitor.start("prediction")
    
    try:
        # 检查输入文件
        if not args.input_file or not os.path.exists(args.input_file):
            ColorOutput.error("输入文件不存在")
            return
        
        # 设置输出文件
        if not args.output_file:
            base_name = os.path.splitext(os.path.basename(args.input_file))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_file = os.path.join(DATA_DIR, f"{base_name}_result_{timestamp}.csv")
        
        # 初始化组件
        ColorOutput.info("初始化系统组件...")
        monitor.checkpoint("init_components")
        
        data_loader = DataLoader(enable_cache=getattr(args, 'enable_cache', True))
        
        # 加载数据
        ColorOutput.info(f"加载数据: {args.input_file}")
        monitor.checkpoint("load_data")
        
        texts, _, df = data_loader.load_file(args.input_file, text_column=args.text_column)
        
        if not texts:
            ColorOutput.error("没有可预测的文本")
            return
        
        ColorOutput.info(f"数据加载完成: {len(texts)}条文本")
        
        # 加载模型
        ColorOutput.info(f"加载智能模型: {args.model_dir}")
        monitor.checkpoint("load_model")
        
        classifier = VOCMultiTaskClassifier.load(args.model_dir)
        
        # 加载文本处理器配置
        processor_config_path = os.path.join(args.model_dir, 'text_processor_config.json')
        processor_config = {}
        if os.path.exists(processor_config_path):
            with open(processor_config_path, 'r', encoding='utf-8') as f:
                processor_config = json.load(f)
        
        # 预处理文本
        ColorOutput.info("执行智能文本预处理...")
        monitor.checkpoint("preprocess_text")
        
        text_processor = TextProcessor()
        processed_texts = text_processor.batch_process(
            texts,
            enable_segmentation=processor_config.get('enable_segmentation', False),
            show_progress=True
        )
        
        # 预测
        ColorOutput.info("执行智能预测分析...")
        monitor.checkpoint("prediction")
        
        results = classifier.predict(
            processed_texts,
            return_confidence=True,
            threshold=args.threshold
        )
        
        # 处理结果
        predictions = results['predictions']
        confidences = results['confidences']
        
        # 添加结果到DataFrame
        ColorOutput.info("处理预测结果...")
        monitor.checkpoint("process_results")
        
        for label_type, preds in predictions.items():
            if label_type in ['旅程触点', '问题类型']:
                # 多标签结果：转换为字符串
                df[label_type] = [', '.join(pred) if pred else '' for pred in preds]
            else:
                # 单标签结果
                df[label_type] = preds
            
            # 添加置信度
            df[f"{label_type}_置信度"] = [f"{conf:.3f}" for conf in confidences[label_type]]
        
        # 删除不需要的统计列（原始文本长度、处理后文本长度、处理时间）
        for col in ['原始文本长度', '处理后文本长度', '处理时间']:
            if col in df.columns:
                df.drop(columns=col, inplace=True)
        
        # 保存结果
        ColorOutput.info("保存预测结果...")
        monitor.checkpoint("save_results")
        
        df.to_csv(args.output_file, index=False, encoding='utf-8')
        
        # 生成统计报告
        if args.generate_report:
            ColorOutput.info("生成分析报告...")
            monitor.checkpoint("generate_report")
            
            report_dir = os.path.splitext(args.output_file)[0] + "_report"
            generate_comprehensive_report(df, predictions, confidences, report_dir, text_processor.get_stats())
        
        # 性能总结
        duration = monitor.end()
        ColorOutput.success(f"🎉 批量分析完成! 总用时: {monitor.format_duration(duration)}")
        ColorOutput.success(f"📄 结果已保存到: {args.output_file}")
        
        if args.generate_report:
            ColorOutput.success(f"📊 报告已生成到: {report_dir}")
        
        print(f"\n⏱️ 性能报告:")
        print(monitor.get_report())
        
        # 显示预测统计
        show_prediction_statistics(predictions, confidences)
        
    except Exception as e:
        ColorOutput.error(f"批量预测失败: {str(e)}")
        logger.error(f"批量预测失败: {str(e)}", exc_info=True)

def generate_comprehensive_report(df, predictions, confidences, output_dir, text_stats):
    """生成综合分析报告"""
    try:
        # 创建可视化器
        visualizer = VOCVisualizer()
        
        # 生成可视化报告
        generated_files = visualizer.generate_report(df, output_dir, "VOC智能分析报告")
        
        # 生成文本处理报告
        text_report = {
            "timestamp": datetime.now().isoformat(),
            "text_processing_stats": text_stats,
            "prediction_stats": generate_prediction_statistics(predictions, confidences),
            "data_quality": {
                "total_records": len(df),
                "valid_predictions": sum(1 for pred in predictions.get('情感', []) if pred),
                "avg_confidence": {
                    label_type: float(np.mean(confs)) 
                    for label_type, confs in confidences.items()
                }
            }
        }
        
        # 保存文本报告
        text_report_path = os.path.join(output_dir, "processing_report.json")
        with open(text_report_path, 'w', encoding='utf-8') as f:
            json.dump(text_report, f, ensure_ascii=False, indent=2)
        
        generated_files['text_report'] = text_report_path
        
        ColorOutput.success(f"综合报告生成完成，包含 {len(generated_files)} 个文件")
        
    except Exception as e:
        ColorOutput.error(f"报告生成失败: {str(e)}")

def generate_prediction_statistics(predictions, confidences):
    """生成预测统计信息"""
    stats = {}
    
    for label_type, preds in predictions.items():
        if label_type in ['旅程触点', '问题类型']:
            # 多标签统计
            all_labels = []
            for pred in preds:
                if pred:
                    all_labels.extend(pred)
            
            label_counts = {}
            for label in all_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            stats[label_type] = {
                "type": "multilabel",
                "unique_labels": len(set(all_labels)),
                "total_predictions": len(all_labels),
                "avg_labels_per_sample": len(all_labels) / len(preds) if preds else 0,
                "top_labels": dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:5])
            }
        else:
            # 单标签统计
            label_counts = {}
            for pred in preds:
                if pred:
                    label_counts[pred] = label_counts.get(pred, 0) + 1
            
            stats[label_type] = {
                "type": "single",
                "unique_labels": len(set(preds)),
                "distribution": label_counts
            }
        
        # 置信度统计
        confs = confidences[label_type]
        stats[label_type]["confidence"] = {
            "mean": float(np.mean(confs)),
            "std": float(np.std(confs)),
            "min": float(np.min(confs)),
            "max": float(np.max(confs)),
            "high_confidence_rate": float(np.mean([c > 0.8 for c in confs]))
        }
    
    return stats

def show_prediction_statistics(predictions, confidences):
    """显示预测统计信息"""
    ColorOutput.title("预测结果统计")
    
    for label_type, preds in predictions.items():
        print(f"\n📊 {label_type}:")
        
        if label_type in ['旅程触点', '问题类型']:
            # 多标签统计
            all_labels = [label for pred_list in preds for label in pred_list if pred_list]
            unique_labels = len(set(all_labels))
            avg_labels = len(all_labels) / len(preds) if preds else 0
            
            print(f"  • 总预测数: {len(all_labels)}")
            print(f"  • 唯一标签数: {unique_labels}")
            print(f"  • 平均标签数/样本: {avg_labels:.2f}")
        else:
            # 单标签统计
            valid_preds = [pred for pred in preds if pred]
            unique_labels = len(set(valid_preds))
            
            print(f"  • 有效预测数: {len(valid_preds)}")
            print(f"  • 唯一标签数: {unique_labels}")
        
        # 置信度统计
        confs = confidences[label_type]
        mean_conf = np.mean(confs)
        high_conf_rate = np.mean([c > 0.8 for c in confs]) * 100
        
        print(f"  • 平均置信度: {mean_conf:.3f}")
        print(f"  • 高置信度比例: {high_conf_rate:.1f}%")

def predict_single_enhanced(text, model_dir, threshold=0.3):
    """增强版单文本预测"""
    try:
        # 加载模型
        classifier = VOCMultiTaskClassifier.load(model_dir)
        
        # 加载文本处理器配置
        processor_config_path = os.path.join(model_dir, 'text_processor_config.json')
        processor_config = {}
        if os.path.exists(processor_config_path):
            with open(processor_config_path, 'r', encoding='utf-8') as f:
                processor_config = json.load(f)
        
        # 预处理文本
        text_processor = TextProcessor()
        processed_text = text_processor.preprocess(
            text,
            enable_segmentation=processor_config.get('enable_segmentation', False)
        )
        
        # 预测
        result = classifier.predict_single_text(processed_text, format_output=True)
        
        # 添加处理信息
        result['处理信息'] = {
            '原始长度': len(text),
            '处理后长度': len(processed_text),
            '预测时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            '阈值': threshold
        }
        
        return result
    except Exception as e:
        ColorOutput.error(f"预测失败: {str(e)}")
        return None

def save_training_report(train_results, output_dir, text_stats):
    """保存训练报告"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "training_results": train_results,
        "text_processing_stats": text_stats,
        "model_info": {
            "framework": "transformers + sklearn",
            "base_model": "hfl/chinese-roberta-wwm-ext",
            "version": "3.0"
        },
        "system_info": {
            "python_version": platform.python_version(),
            "platform": platform.platform()
        }
    }
    
    report_path = os.path.join(output_dir, "training_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    ColorOutput.info(f"训练报告已保存: {report_path}")

def initialize_system_enhanced(model_dir):
    """增强版系统初始化"""
    ColorOutput.title("VOC系统初始化")
    
    try:
        # 创建备份
        if os.path.exists(model_dir):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{model_dir}_backup_{timestamp}"
            shutil.copytree(model_dir, backup_dir)
            ColorOutput.info(f"已备份现有模型到: {backup_dir}")
        
        # 清除现有数据
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            ColorOutput.success("现有模型数据已清除")
        
        # 创建目录结构
        dirs_to_create = [
            model_dir,
            os.path.join(DATA_DIR, 'cache'),
            os.path.join(DATA_DIR, 'logs'),
            os.path.join(DATA_DIR, 'reports'),
            os.path.join(DATA_DIR, 'temp')
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        ColorOutput.success("目录结构创建完成")
        
        # 清理缓存
        cache_dir = os.path.join(DATA_DIR, 'cache')
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            ColorOutput.success("缓存清理完成")
    
        # 验证环境
        try:
            import torch
            from transformers import AutoTokenizer
            
            ColorOutput.info("环境验证:")
            if torch.cuda.is_available():
                ColorOutput.info(f"  ✅ GPU可用: {torch.cuda.get_device_name()}")
            else:
                ColorOutput.info("  ℹ️ 使用CPU模式")
            
            # 测试模型访问
            tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
            ColorOutput.info("  ✅ 模型访问正常")
            
        except Exception as e:
            ColorOutput.warning(f"环境验证警告: {str(e)}")
        
        ColorOutput.success("🎉 系统初始化完成！")
        return True
        
    except Exception as e:
        ColorOutput.error(f"系统初始化失败: {str(e)}")
        return False

def demo_mode():
    """演示模式"""
    ColorOutput.title("VOC系统演示模式")
    
    # 演示数据
    demo_texts = [
        "理想ONE的语音助手识别很准确，但是导航偶尔会绕路",
        "充电速度比宣传的慢，续航也不如预期",
        "销售顾问服务很好，交车流程很顺畅",
        "座椅很舒适，隔音效果也不错，开起来很安静",
        "OTA升级后系统更流畅了，新增的功能也很实用"
    ]
    
    model_dir = os.path.join(DATA_DIR, 'models/latest')
    
    # 检查是否有训练好的模型
    if not os.path.exists(model_dir) or not any(f.endswith('.pkl') for f in os.listdir(model_dir)):
        ColorOutput.warning("未找到训练好的模型，将使用示例数据训练")
        
        # 创建临时参数对象
        demo_args = argparse.Namespace(
            train_file=None,
            text_column='text',
            test_size=0.2,
            validation_split=0.2,
            model_dir=model_dir,
            cross_validation=False,
            use_ensemble=True,
            max_length=256,
            pooling_strategy='mean',
            config_file=None,
            enable_text_segmentation=False
        )
        
        # 训练模型
        classifier = enhanced_train_model(demo_args, interactive_mode=False)
        if not classifier:
            ColorOutput.error("演示模型训练失败")
            return
    
    # 演示预测
    ColorOutput.info("\n🎯 开始演示预测分析...")
    
    for i, text in enumerate(demo_texts, 1):
        ColorOutput.info(f"\n--- 演示 {i}/{len(demo_texts)} ---")
        print(f"📝 原始文本: {text}")
        
        result = predict_single_enhanced(text, model_dir)
        
        if result:
            print("🎯 分析结果:")
            for key, value in result.items():
                if key == '置信度' or key == '处理信息':
                    continue
                
                confidence = result.get('置信度', {}).get(key, 0)
                if isinstance(value, list):
                    value_str = ', '.join(value) if value else '无'
                else:
                    value_str = str(value)
                
                print(f"  {key}: {value_str} (置信度: {confidence:.3f})")
        
        time.sleep(1)  # 演示间隔
    
    ColorOutput.success("🎉 演示完成！")

def interactive_mode_enhanced():
    """增强版交互式模式"""
    ColorOutput.title("VOC智能标签系统 v3.0")
    
    model_dir = os.path.join(DATA_DIR, 'models/latest')
    model_exists = os.path.exists(model_dir) and any(f.endswith('.pkl') for f in os.listdir(model_dir))
    
    if not model_exists:
        ColorOutput.warning("未检测到预训练模型")
        choice = input("是否使用VOC示例数据训练模型? (y/n): ")
        if choice.lower() == 'y':
            args = argparse.Namespace(
                train_file=None,
                text_column='text',
                test_size=0.2,
                validation_split=0.2,
                model_dir=model_dir,
                cross_validation=False,
                use_ensemble=True,
                max_length=256,
                pooling_strategy='mean',
                config_file=None,
                enable_text_segmentation=False
            )
            enhanced_train_model(args, interactive_mode=True)
        else:
            ColorOutput.warning("请先训练模型")
            return
    
    while True:
        print_enhanced_menu()
        choice = input("\n请选择功能 (1-8): ").strip()
        
        try:
            if choice == '1':
                handle_single_prediction(model_dir)
            elif choice == '2':
                handle_batch_prediction(model_dir)
            elif choice == '3':
                handle_model_training(model_dir)
            elif choice == '4':
                handle_model_evaluation(model_dir)
            elif choice == '5':
                handle_system_initialize(model_dir)
            elif choice == '6':
                handle_system_info()
            elif choice == '7':
                demo_mode()
            elif choice == '8':
                ColorOutput.success("感谢使用VOC智能标签系统！")
                break
            else:
                ColorOutput.error("无效选择，请重新输入")
        except KeyboardInterrupt:
            ColorOutput.warning("\n操作已中断")
        except Exception as e:
            ColorOutput.error(f"操作失败: {str(e)}")
            logger.error(f"操作失败: {str(e)}", exc_info=True)

def print_enhanced_menu():
    """打印增强版菜单（中文，带emoji），确保边框对齐美观"""
    from wcwidth import wcswidth

    menu_items = [
        "1. 💬 单条文本智能分析",
        "2. 📊 批量CSV文件处理",
        "3. 🎯 训练/更新模型",
        "4. 📈 模型性能评估",
        "5. 🔄 系统初始化",
        "6. 🧾 系统信息",
        "7. 🎪 演示模式",
        "8. 🚪 退出系统"
    ]
    title = "VOC智能分析系统 v3.0"
    # 计算最大显示宽度（菜单项和标题都要考虑）
    max_menu_width = max(wcswidth(item) for item in menu_items)
    title_width = wcswidth(title)
    box_width = max(max_menu_width, title_width) + 6  # 适当加长

    print("\n" + "┌" + "─" * box_width + "┐")
    # 手动居中标题
    title_pad = box_width - wcswidth(title)
    left_pad = title_pad // 2
    right_pad = title_pad - left_pad
    print("│" + " " * left_pad + title + " " * right_pad + "│")
    print("├" + "─" * box_width + "┤")
    for item in menu_items:
        pad = box_width - wcswidth(item)
        print("│ " + item + " " * (pad - 1) + "│")
    print("└" + "─" * box_width + "┘")

def handle_single_prediction(model_dir):
    """处理单文本预测"""
    ColorOutput.info("\n=== 单文本智能分析 ===")
    text = input("请输入要分析的文本: ").strip()
    
    if not text:
        ColorOutput.error("文本不能为空")
        return
    
    ColorOutput.info("🧠 智能分析中...")
    result = predict_single_enhanced(text, model_dir)
    
    if result:
        ColorOutput.success("\n📋 智能分析结果:")
        print("=" * 50)
        
        for key, value in result.items():
            if key in ['置信度', '处理信息']:
                continue
                
            confidence = result.get('置信度', {}).get(key, 0)
            if isinstance(value, list):
                value_str = ', '.join(value) if value else '无'
            else:
                value_str = str(value)
            
            print(f"{key:12}: {value_str}")
            print(f"{'置信度':12}: {confidence:.3f}")
            print("-" * 50)
        
        # 显示处理信息
        if '处理信息' in result:
            info = result['处理信息']
            print(f"\n📊 处理统计:")
            print(f"  原始长度: {info['原始长度']} 字符")
            print(f"  处理后长度: {info['处理后长度']} 字符")
            print(f"  预测时间: {info['预测时间']}")
    else:
        ColorOutput.error("分析失败，请检查模型或重试")

def handle_batch_prediction(model_dir):
    """处理批量预测"""
    ColorOutput.info("\n=== 批量文件处理 ===")
    
    input_file = input("请输入文件路径: ").strip()
    if not input_file or not os.path.exists(input_file):
        ColorOutput.error("文件不存在")
        return
    
    text_column = input("文本列名 (默认'text'): ").strip() or "text"
    
    output_file = input("输出文件路径 (可选): ").strip()
    if not output_file:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(DATA_DIR, f"{base_name}_result_{timestamp}.csv")
    
    threshold = input("多标签分类阈值 (默认0.3): ").strip()
    try:
        threshold = float(threshold) if threshold else 0.3
    except ValueError:
        threshold = 0.3
    
    generate_report = input("是否生成分析报告? (y/n): ").strip().lower() == 'y'
    
    # 创建参数对象
    args = argparse.Namespace(
        input_file=input_file,
        text_column=text_column,
        output_file=output_file,
        model_dir=model_dir,
        threshold=threshold,
        generate_report=generate_report,
        enable_cache=True
    )
    
    enhanced_predict_batch(args)

def handle_model_training(model_dir):
    """处理模型训练"""
    ColorOutput.info("\n=== 智能模型训练 ===")
    
    train_file = input("训练数据文件路径 (留空使用VOC示例数据): ").strip()
    if train_file and not os.path.exists(train_file):
        ColorOutput.error("文件不存在")
        return
    
    text_column = input("文本列名 (默认'text'): ").strip() or "text"
    
    # 高级选项
    print("\n🔧 高级训练选项:")
    use_ensemble = input("使用集成学习方法? (y/n, 默认y): ").strip().lower() != 'n'
    enable_segmentation = input("启用中文分词? (y/n, 默认n): ").strip().lower() == 'y'
    
    pooling_options = ['cls', 'mean', 'max', 'attention']
    print("特征提取策略:")
    for i, option in enumerate(pooling_options, 1):
        print(f"  {i}. {option}")
    
    pooling_choice = input("选择策略 (默认2-mean): ").strip()
    try:
        pooling_idx = int(pooling_choice) - 1
        pooling_strategy = pooling_options[pooling_idx] if 0 <= pooling_idx < len(pooling_options) else 'mean'
    except ValueError:
        pooling_strategy = 'mean'
    
    max_length = input("最大序列长度 (默认256): ").strip()
    try:
        max_length = int(max_length) if max_length else 256
    except ValueError:
        max_length = 256
    
    # 创建参数对象
    args = argparse.Namespace(
        train_file=train_file,
        text_column=text_column,
        test_size=0.2,
        validation_split=0.2,
        model_dir=model_dir,
        cross_validation=False,
        use_ensemble=use_ensemble,
        max_length=max_length,
        pooling_strategy=pooling_strategy,
        config_file=None,
        enable_text_segmentation=enable_segmentation
    )
    
    enhanced_train_model(args, interactive_mode=True)
            
def handle_model_evaluation(model_dir):
    """处理模型评估"""
    ColorOutput.info("\n=== 模型性能评估 ===")
    
    if not os.path.exists(model_dir):
        ColorOutput.error("模型不存在，请先训练模型")
        return
    
    test_file = input("测试数据文件路径: ").strip()
    if not test_file or not os.path.exists(test_file):
        ColorOutput.error("测试文件不存在")
        return
    
    text_column = input("文本列名 (默认'text'): ").strip() or "text"
    
    try:
        # 加载模型
        ColorOutput.info("加载模型...")
        classifier = VOCMultiTaskClassifier.load(model_dir)
        
        # 加载测试数据
        data_loader = DataLoader()
        texts, labels_dict, df = data_loader.load_file(test_file, text_column=text_column)
        
        if not texts or not labels_dict:
            ColorOutput.error("测试数据格式不正确")
            return
        
        # 预处理
        text_processor = TextProcessor()
        processed_texts = text_processor.batch_process(texts, show_progress=True)
        
        # 评估
        ColorOutput.info("执行模型评估...")
        eval_results = classifier.evaluate(processed_texts, labels_dict)
        
        # 显示结果
        ColorOutput.success("\n📊 模型评估结果:")
        print("=" * 60)
        for label_type, metrics in eval_results.items():
            print(f"\n📈 {label_type}:")
            for metric, value in metrics.items():
                print(f"  {metric:15}: {value:.4f}")
        
        # 保存评估报告
        report_path = os.path.join(model_dir, "evaluation_report.json")
        eval_report = {
            "timestamp": datetime.now().isoformat(),
            "test_file": test_file,
            "results": eval_results,
            "test_data_stats": {
                "total_samples": len(texts),
                "processed_samples": len(processed_texts),
                "text_processing_stats": text_processor.get_stats()
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(eval_report, f, ensure_ascii=False, indent=2)
        
        ColorOutput.success(f"\n📄 评估报告已保存: {report_path}")
        
    except Exception as e:
        ColorOutput.error(f"评估失败: {str(e)}")

def handle_system_initialize(model_dir):
    """处理系统初始化"""
    ColorOutput.warning("\n=== 系统初始化 ===")
    ColorOutput.warning("⚠️ 此操作将清除所有现有模型数据和缓存!")
    
    confirm = input("确认要初始化系统吗? (输入'YES'确认): ").strip()
    if confirm == 'YES':
        if initialize_system_enhanced(model_dir):
            ColorOutput.success("系统初始化成功!")
        else:
            ColorOutput.error("系统初始化失败!")
    else:
        ColorOutput.info("操作已取消")

def handle_system_info():
    """显示系统信息"""
    ColorOutput.info("\n=== 系统信息 ===")
    
    # 环境信息
    import torch
    import transformers
    import sklearn
    
    info = {
        "Python版本": platform.python_version(),
        "操作系统": f"{platform.system()} {platform.release()}",
        "PyTorch版本": torch.__version__,
        "Transformers版本": transformers.__version__,
        "Scikit-learn版本": sklearn.__version__,
        "CUDA可用": "是" if torch.cuda.is_available() else "否",
        "系统版本": "v3.0"
    }
    
    if torch.cuda.is_available():
        info["GPU设备"] = torch.cuda.get_device_name()
        info["GPU内存"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    
    print("=" * 50)
    for key, value in info.items():
        print(f"{key:20}: {value}")
    print("=" * 50)
    
    # 模型信息
    model_dir = os.path.join(DATA_DIR, 'models/latest')
    if os.path.exists(model_dir):
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
            
            print("\n🤖 当前模型配置:")
            for key, value in model_config.items():
                print(f"{key:20}: {value}")
        
        # 显示文件统计
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        print(f"\n📁 模型文件: {len(model_files)}个")
        
        # 显示训练报告
        report_path = os.path.join(model_dir, 'training_report.json')
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            print(f"📊 最后训练时间: {report.get('timestamp', '未知')}")
        else:
            ColorOutput.warning("未找到训练好的模型")
    else:
        ColorOutput.warning("未找到训练好的模型")

def main():
    """主程序入口"""
    try:
        # 创建必要目录
        dirs_to_create = [
            DATA_DIR,
            os.path.join(DATA_DIR, 'models'),
            os.path.join(DATA_DIR, 'logs'),
            os.path.join(DATA_DIR, 'reports'),
            os.path.join(DATA_DIR, 'cache'),
            os.path.join(DATA_DIR, 'temp')
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        # 解析参数
        args = parse_args()
        
        # 设置日志级别
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # 根据模式执行对应功能
        if args.mode == 'train':
            enhanced_train_model(args)
        elif args.mode == 'batch':
            enhanced_predict_batch(args)
        elif args.mode == 'predict':
            # 单文本预测模式
            if len(sys.argv) > 2:
                text = sys.argv[2]
                result = predict_single_enhanced(text, args.model_dir)
                if result:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    sys.exit(1)
            else:
                ColorOutput.error("predict模式需要提供文本参数")
        elif args.mode == 'evaluate':
            handle_model_evaluation(args.model_dir)
        elif args.mode == 'initialize':
            handle_system_initialize(args.model_dir)
        elif args.mode == 'demo':
            demo_mode()
        elif args.mode == 'interactive':
            interactive_mode_enhanced()
        else:
            ColorOutput.error(f"未支持的模式: {args.mode}")
            
    except KeyboardInterrupt:
        ColorOutput.warning("\n程序被用户中断")
    except Exception as e:
        ColorOutput.error(f"程序执行失败: {str(e)}")
        logger.error(f"程序执行失败: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 