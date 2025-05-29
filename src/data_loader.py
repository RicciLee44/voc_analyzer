"""
数据加载和处理器 - 改进版
支持多种数据源、自动编码检测、数据验证、增量加载等功能
"""
import os
import pandas as pd
import numpy as np
import json
import time
import chardet
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm
import logging
from datetime import datetime
import pickle

class DataLoader:
    """
    增强版数据加载器
    支持CSV、JSON、Excel等多种格式，自动编码检测，数据验证等功能
    """
    
    def __init__(self, cache_dir: Optional[str] = None, enable_cache: bool = True):
        """
        初始化数据加载器
        
        Args:
            cache_dir: 缓存目录
            enable_cache: 是否启用缓存
        """
        self.logger = logging.getLogger(__name__)
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir or "data/cache"
        
        # 创建缓存目录
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # 支持的文件格式
        self.supported_formats = {
            '.csv': self._load_csv_file,
            '.json': self._load_json_file,
            '.jsonl': self._load_jsonl_file,
            '.xlsx': self._load_excel_file,
            '.xls': self._load_excel_file,
            '.tsv': self._load_tsv_file,
            '.txt': self._load_txt_file,
            '.parquet': self._load_parquet_file
        }
        
        # 理想汽车VOC标签体系定义
        self.voc_label_schema = {
            '旅程触点': {
                'type': 'multilabel',
                'values': [
                    '品牌认知', '官网/App浏览体验', '到店咨询/试驾预约',
                    '订购流程', '销售服务态度', '价格透明度',
                    '交车速度与流程', '交付培训', '门店服务环境',
                    '加速、制动、操控', '悬挂与舒适度', '噪音与隔音体验',
                    '智能导航', '语音助手', 'HUD/中控交互', 'OTA更新体验',
                    '家用充电桩', '公共充电站体验', '续航表现',
                    '保养与维修', '客服响应速度', '二手车/置换服务'
                ]
            },
            '问题类型': {
                'type': 'multilabel',
                'values': [
                    '稳定性问题', '性能问题', '可用性问题', '兼容性问题',
                    '美观度问题', '交互逻辑问题', '安全隐患', 
                    '服务体验问题', '期待落差'
                ]
            },
            '情感': {
                'type': 'single',
                'values': ['正面', '中性', '负面']
            }
        }
        
        # 数据统计信息
        self.stats = {
            'loaded_files': 0,
            'total_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'encoding_detections': {},
            'load_times': []
        }
    
    def detect_encoding(self, file_path: str, sample_size: int = 10000) -> str:
        """
        自动检测文件编码
        
        Args:
            file_path: 文件路径
            sample_size: 检测样本大小
            
        Returns:
            检测到的编码
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(sample_size)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                self.logger.info(f"编码检测: {encoding} (置信度: {confidence:.2f})")
                self.stats['encoding_detections'][file_path] = {
                    'encoding': encoding,
                    'confidence': confidence
                }
                
                return encoding
        except Exception as e:
            self.logger.warning(f"编码检测失败: {str(e)}")
            return 'utf-8'
    
    def _get_cache_path(self, file_path: str) -> str:
        """获取缓存文件路径"""
        file_hash = hash(file_path + str(os.path.getmtime(file_path)))
        cache_name = f"cache_{abs(file_hash)}.pkl"
        return os.path.join(self.cache_dir, cache_name)
    
    def _load_from_cache(self, cache_path: str) -> Optional[Tuple]:
        """从缓存加载数据"""
        if not self.enable_cache or not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.logger.info(f"从缓存加载数据: {cache_path}")
                return cached_data
        except Exception as e:
            self.logger.warning(f"缓存加载失败: {str(e)}")
            return None
    
    def _save_to_cache(self, cache_path: str, data: Tuple):
        """保存数据到缓存"""
        if not self.enable_cache:
            return
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                self.logger.info(f"数据已缓存: {cache_path}")
        except Exception as e:
            self.logger.warning(f"缓存保存失败: {str(e)}")
    
    def _load_csv_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """加载CSV文件"""
        encoding = kwargs.get('encoding') or self.detect_encoding(file_path)
        
        # 尝试多种编码
        encodings = [encoding, 'utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']
        
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc, **{k: v for k, v in kwargs.items() if k != 'encoding'})
                self.logger.info(f"成功使用编码 {enc} 加载CSV文件")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"加载CSV文件失败 (编码: {enc}): {str(e)}")
                continue
        
        raise ValueError(f"无法加载CSV文件 {file_path}，尝试了所有编码")
    
    def _load_json_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """加载JSON文件"""
        encoding = kwargs.get('encoding') or self.detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
                
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("不支持的JSON格式")
                
            return df
        except Exception as e:
            self.logger.error(f"加载JSON文件失败: {str(e)}")
            raise
    
    def _load_jsonl_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """加载JSONL文件 (每行一个JSON对象)"""
        encoding = kwargs.get('encoding') or self.detect_encoding(file_path)
        
        try:
            data = []
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line.strip()))
            
            return pd.DataFrame(data)
        except Exception as e:
            self.logger.error(f"加载JSONL文件失败: {str(e)}")
            raise
    
    def _load_excel_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """加载Excel文件"""
        try:
            # 尝试读取第一个工作表
            df = pd.read_excel(file_path, **kwargs)
            return df
        except Exception as e:
            self.logger.error(f"加载Excel文件失败: {str(e)}")
            raise
    
    def _load_tsv_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """加载TSV文件"""
        kwargs['sep'] = '\t'
        return self._load_csv_file(file_path, **kwargs)
    
    def _load_txt_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """加载纯文本文件 (每行作为一条记录)"""
        encoding = kwargs.get('encoding') or self.detect_encoding(file_path)
        text_column = kwargs.get('text_column', 'text')
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = [line.strip() for line in f if line.strip()]
            
            return pd.DataFrame({text_column: lines})
        except Exception as e:
            self.logger.error(f"加载TXT文件失败: {str(e)}")
            raise
    
    def _load_parquet_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """加载Parquet文件"""
        try:
            import pyarrow.parquet as pq
            df = pd.read_parquet(file_path, **kwargs)
            return df
        except ImportError:
            self.logger.error("需要安装pyarrow库来读取Parquet文件: pip install pyarrow")
            raise
        except Exception as e:
            self.logger.error(f"加载Parquet文件失败: {str(e)}")
            raise
    
    def load_file(self, 
                  file_path: str, 
                  text_column: str = "text",
                  encoding: Optional[str] = None,
                  use_cache: bool = True,
                  **kwargs) -> Tuple[List[str], Dict[str, List], pd.DataFrame]:
        """
        通用文件加载接口
        
        Args:
            file_path: 文件路径
            text_column: 文本列名
            encoding: 指定编码
            use_cache: 是否使用缓存
            **kwargs: 其他参数
            
        Returns:
            (texts, labels_dict, dataframe)
        """
        start_time = time.time()
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 获取文件扩展名
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {file_ext}")
        
        # 尝试从缓存加载
        cache_path = self._get_cache_path(file_path)
        if use_cache and self.enable_cache:
            cached_data = self._load_from_cache(cache_path)
            if cached_data:
                load_time = time.time() - start_time
                self.stats['load_times'].append(load_time)
                return cached_data
        
        # 显示加载进度
        with tqdm(total=5, desc="数据加载", ncols=80) as pbar:
            # 步骤1: 检测文件格式
            pbar.set_description("检测文件格式")
            time.sleep(0.1)
            pbar.update(1)
            
            # 步骤2: 读取文件
            pbar.set_description("读取文件")
            try:
                loader_func = self.supported_formats[file_ext]
                df = loader_func(file_path, encoding=encoding, **kwargs)
                self.logger.info(f"成功加载文件: {file_path}")
            except Exception as e:
                self.logger.error(f"文件加载失败: {str(e)}")
                raise
            pbar.update(1)
            
            # 步骤3: 验证数据格式
            pbar.set_description("验证数据格式")
            self._validate_dataframe(df, text_column)
            pbar.update(1)
            
            # 步骤4: 提取文本和标签
            pbar.set_description("提取文本和标签")
            texts, labels_dict = self._extract_texts_and_labels(df, text_column)
            pbar.update(1)
            
            # 步骤5: 数据质量检查
            pbar.set_description("数据质量检查")
            self._check_data_quality(texts, labels_dict)
            pbar.update(1)
        
        # 保存到缓存
        result = (texts, labels_dict, df)
        if use_cache and self.enable_cache:
            self._save_to_cache(cache_path, result)
        
        # 更新统计信息
        load_time = time.time() - start_time
        self.stats['loaded_files'] += 1
        self.stats['total_records'] += len(df)
        self.stats['load_times'].append(load_time)
        
        self.logger.info(f"数据加载完成: {len(texts)}条文本, {len(labels_dict)}种标签, 用时{load_time:.2f}秒")
        
        return result
    
    def load_csv(self, 
                 file_path: str, 
                 text_column: str = "text", 
                 encoding: Optional[str] = None,
                 **kwargs) -> Tuple[List[str], Dict[str, List], pd.DataFrame]:
        """
        加载CSV文件 (保持向后兼容)
        
        Args:
            file_path: CSV文件路径
            text_column: 文本列名
            encoding: 文件编码
            
        Returns:
            (texts, labels_dict, dataframe)
        """
        return self.load_file(file_path, text_column, encoding, **kwargs)
    
    def _validate_dataframe(self, df: pd.DataFrame, text_column: str):
        """验证DataFrame格式"""
        if df.empty:
            raise ValueError("文件为空")
        
        if text_column not in df.columns:
            available_cols = df.columns.tolist()
            raise ValueError(f"未找到文本列 '{text_column}'。可用列: {available_cols}")
        
        # 检查文本列是否有有效数据
        valid_texts = df[text_column].dropna().astype(str).str.strip()
        valid_count = (valid_texts != '').sum()
        
        if valid_count == 0:
            raise ValueError(f"文本列 '{text_column}' 没有有效数据")
        
        self.logger.info(f"数据验证通过: {len(df)}行, {len(df.columns)}列, {valid_count}条有效文本")
    
    def _extract_texts_and_labels(self, df: pd.DataFrame, text_column: str) -> Tuple[List[str], Dict[str, List]]:
        """从DataFrame提取文本和标签"""
        # 提取文本
        texts = df[text_column].fillna("").astype(str).tolist()
        
        # 提取标签
        labels_dict = {}
        for col in df.columns:
            if col != text_column:
                labels_dict[col] = df[col].fillna("").astype(str).tolist()
        
        return texts, labels_dict
    
    def _check_data_quality(self, texts: List[str], labels_dict: Dict[str, List]):
        """检查数据质量"""
        total_texts = len(texts)
        
        # 文本质量检查
        empty_texts = sum(1 for text in texts if not text.strip())
        short_texts = sum(1 for text in texts if len(text.strip()) < 5)
        long_texts = sum(1 for text in texts if len(text.strip()) > 1000)
        
        self.stats['valid_records'] = total_texts - empty_texts
        self.stats['invalid_records'] = empty_texts
        
        # 标签质量检查
        label_stats = {}
        for label_type, labels in labels_dict.items():
            non_empty_labels = sum(1 for label in labels if label.strip())
            unique_labels = len(set(label.strip() for label in labels if label.strip()))
            
            label_stats[label_type] = {
                'total': len(labels),
                'non_empty': non_empty_labels,
                'unique': unique_labels,
                'coverage': non_empty_labels / len(labels) * 100 if labels else 0
            }
        
        # 打印质量报告
        print(f"\n📊 数据质量报告:")
        print(f"  总文本数: {total_texts}")
        print(f"  有效文本: {total_texts - empty_texts} ({(total_texts - empty_texts) / total_texts * 100:.1f}%)")
        print(f"  空文本: {empty_texts}")
        print(f"  短文本(<5字符): {short_texts}")
        print(f"  长文本(>1000字符): {long_texts}")
        
        if label_stats:
            print(f"\n🏷️ 标签质量:")
            for label_type, stats in label_stats.items():
                print(f"  {label_type}:")
                print(f"    覆盖率: {stats['coverage']:.1f}%")
                print(f"    唯一值: {stats['unique']}")
    
    def load_examples(self, example_type: str = "default") -> Tuple[List[str], Dict[str, List]]:
        """
        加载示例数据
        
        Args:
            example_type: 示例类型 ('default', 'voc', 'multilabel')
            
        Returns:
            (texts, labels_dict)
        """
        print(f"🔄 加载示例数据 ({example_type})...")
        
        if example_type == "voc":
            # 理想汽车VOC示例数据
            examples = [
                {
                    "text": "导航经常带我绕远路，语音也识别不出来，还老是闪退",
                    "旅程触点": ["智能导航", "语音助手"],
                    "问题类型": ["稳定性问题", "可用性问题"],
                    "情感": "负面"
                },
                {
                    "text": "交车当天小哥讲解得很细，座椅按摩也比想象中舒服",
                    "旅程触点": ["交付培训", "悬挂与舒适度"],
                    "问题类型": [],
                    "情感": "正面"
                },
                {
                    "text": "充电桩充电速度太慢了，而且经常坏",
                    "旅程触点": ["公共充电站体验"],
                    "问题类型": ["性能问题", "稳定性问题"],
                    "情感": "负面"
                },
                {
                    "text": "OTA升级后系统更流畅了，新功能也很实用",
                    "旅程触点": ["OTA更新体验"],
                    "问题类型": [],
                    "情感": "正面"
                },
                {
                    "text": "销售顾问很专业，价格也很透明，没有乱收费",
                    "旅程触点": ["销售服务态度", "价格透明度"],
                    "问题类型": [],
                    "情感": "正面"
                },
                {
                    "text": "车机系统卡顿，触摸不灵敏，影响驾驶体验",
                    "旅程触点": ["HUD/中控交互"],
                    "问题类型": ["稳定性问题", "可用性问题"],
                    "情感": "负面"
                },
                {
                    "text": "续航比官方宣传的要短，冬天掉电特别快",
                    "旅程触点": ["续航表现"],
                    "问题类型": ["期待落差", "性能问题"],
                    "情感": "负面"
                },
                {
                    "text": "客服响应很及时，问题解决得也很快",
                    "旅程触点": ["客服响应速度"],
                    "问题类型": [],
                    "情感": "正面"
                }
            ]
        else:
            # 默认示例数据
            examples = [
                {
                    "text": "这个产品的价格太贵了",
                    "journey_touchpoints": "购买期",
                    "problem_categories": "价格问题",
                    "sentiment": "负面"
                },
                {
                    "text": "客服响应很快，问题都解决了",
                    "journey_touchpoints": "服务期",
                    "problem_categories": "",
                    "sentiment": "正面"
                },
                {
                    "text": "系统经常闪退，很影响使用",
                    "journey_touchpoints": "使用期",
                    "problem_categories": "系统故障",
                    "sentiment": "负面"
                },
                {
                    "text": "包装很精美，质感很好",
                    "journey_touchpoints": "购买期",
                    "problem_categories": "",
                    "sentiment": "正面"
                }
            ]
        
        # 模拟加载过程
        with tqdm(total=len(examples), desc="示例数据准备", ncols=80) as pbar:
            for i in range(len(examples)):
                time.sleep(0.1)  # 模拟处理时间
                pbar.update(1)
        
        # 转换为标准格式
        texts = [ex["text"] for ex in examples]
        
        # 构建标签字典
        labels_dict = {}
        if examples:
            # 获取所有标签键（除了text）
            all_keys = set()
            for ex in examples:
                all_keys.update(k for k in ex.keys() if k != "text")
            
            # 初始化标签字典
            for key in all_keys:
                labels_dict[key] = []
            
            # 填充标签值
            for ex in examples:
                for key in all_keys:
                    value = ex.get(key, "")
                    if isinstance(value, list):
                        # 多标签转换为逗号分隔字符串
                        labels_dict[key].append(",".join(value) if value else "")
                    else:
                        labels_dict[key].append(str(value) if value else "")
        
        print(f"✅ 示例数据加载完成: {len(texts)}条文本, {len(labels_dict)}种标签")
        
        # 显示标签统计
        for label_type, labels in labels_dict.items():
            non_empty = sum(1 for label in labels if label.strip())
            print(f"  {label_type}: {non_empty}/{len(labels)}条有标签")
        
        return texts, labels_dict
    
    def batch_load_files(self, 
                        file_paths: List[str], 
                        text_column: str = "text") -> Tuple[List[str], Dict[str, List], pd.DataFrame]:
        """
        批量加载多个文件
        
        Args:
            file_paths: 文件路径列表
            text_column: 文本列名
            
        Returns:
            合并后的数据
        """
        all_texts = []
        all_labels = {}
        all_dfs = []
        
        print(f"📁 批量加载 {len(file_paths)} 个文件...")
        
        for file_path in tqdm(file_paths, desc="批量加载", ncols=80):
            try:
                texts, labels_dict, df = self.load_file(file_path, text_column)
                
                all_texts.extend(texts)
                all_dfs.append(df)
                
                # 合并标签
                for label_type, labels in labels_dict.items():
                    if label_type not in all_labels:
                        all_labels[label_type] = []
                    all_labels[label_type].extend(labels)
                
            except Exception as e:
                self.logger.error(f"加载文件失败 {file_path}: {str(e)}")
                continue
        
        # 合并DataFrame
        combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        
        print(f"✅ 批量加载完成: {len(all_texts)}条文本")
        return all_texts, all_labels, combined_df
    
    def save_data(self, 
                  texts: List[str], 
                  labels_dict: Dict[str, List], 
                  output_path: str,
                  text_column: str = "text"):
        """
        保存数据到文件
        
        Args:
            texts: 文本列表
            labels_dict: 标签字典
            output_path: 输出路径
            text_column: 文本列名
        """
        try:
            # 构建DataFrame
            data = {text_column: texts}
            data.update(labels_dict)
            
            df = pd.DataFrame(data)
            
            # 根据文件扩展名选择保存格式
            file_ext = Path(output_path).suffix.lower()
            
            if file_ext == '.csv':
                df.to_csv(output_path, index=False, encoding='utf-8')
            elif file_ext == '.json':
                df.to_json(output_path, orient='records', ensure_ascii=False, indent=2)
            elif file_ext in ['.xlsx', '.xls']:
                df.to_excel(output_path, index=False)
            else:
                # 默认保存为CSV
                df.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"✅ 数据已保存到: {output_path}")
            
        except Exception as e:
            print(f"❌ 保存数据失败: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取加载统计信息"""
        avg_load_time = np.mean(self.stats['load_times']) if self.stats['load_times'] else 0
        
        return {
            'loaded_files': self.stats['loaded_files'],
            'total_records': self.stats['total_records'],
            'valid_records': self.stats['valid_records'],
            'invalid_records': self.stats['invalid_records'],
            'avg_load_time': avg_load_time,
            'encoding_detections': self.stats['encoding_detections']
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        
        print(f"\n📊 数据加载统计:")
        print(f"  已加载文件: {stats['loaded_files']}")
        print(f"  总记录数: {stats['total_records']}")
        print(f"  有效记录: {stats['valid_records']}")
        print(f"  无效记录: {stats['invalid_records']}")
        print(f"  平均加载时间: {stats['avg_load_time']:.2f}秒")
        
        if stats['encoding_detections']:
            print(f"\n🔤 编码检测:")
            for file_path, detection in stats['encoding_detections'].items():
                file_name = os.path.basename(file_path)
                print(f"  {file_name}: {detection['encoding']} ({detection['confidence']:.2f})")
    
    def validate_voc_labels(self, labels_dict: Dict[str, List]) -> Dict[str, Any]:
        """
        验证VOC标签格式
        
        Args:
            labels_dict: 标签字典
            
        Returns:
            验证结果
        """
        validation_results = {}
        
        for label_type, labels in labels_dict.items():
            if label_type not in self.voc_label_schema:
                validation_results[label_type] = {
                    'valid': False,
                    'error': f"未知标签类型: {label_type}"
                }
                continue
            
            schema = self.voc_label_schema[label_type]
            expected_values = set(schema['values'])
            
            # 收集所有实际标签值
            actual_values = set()
            for label in labels:
                if label.strip():
                    if schema['type'] == 'multilabel':
                        # 分割多标签
                        label_list = [l.strip() for l in label.split(',') if l.strip()]
                        actual_values.update(label_list)
                    else:
                        actual_values.add(label.strip())
            
            # 检查无效标签
            invalid_labels = actual_values - expected_values
            
            validation_results[label_type] = {
                'valid': len(invalid_labels) == 0,
                'invalid_labels': list(invalid_labels),
                'valid_labels': list(actual_values & expected_values),
                'coverage': len(actual_values & expected_values) / len(expected_values) * 100
            }
        
        return validation_results


# 使用示例
if __name__ == "__main__":
    # 创建数据加载器
    loader = DataLoader(enable_cache=True)
    
    # 加载示例数据
    texts, labels_dict = loader.load_examples("voc")
    
    # 验证VOC标签
    validation = loader.validate_voc_labels(labels_dict)
    print(f"\n🔍 标签验证结果:")
    for label_type, result in validation.items():
        status = "✅" if result['valid'] else "❌"
        print(f"  {status} {label_type}: {result}")
    
    # 显示统计信息
    loader.print_stats()