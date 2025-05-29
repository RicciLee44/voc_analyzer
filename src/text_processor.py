"""
文本预处理器 - 改进版
专为VOC文本分析优化，支持中文分词、清洗、标准化等功能
"""
import re
import string
import unicodedata
from typing import List, Dict, Optional, Union
import logging
from tqdm import tqdm
import jieba
import jieba.posseg as pseg
from collections import Counter
import pandas as pd

# 配置jieba日志级别，避免过多输出
jieba.setLogLevel(logging.INFO)

class TextProcessor:
    """
    专为VOC分析设计的文本预处理器
    支持中文分词、停用词过滤、文本标准化等功能
    """
    
    def __init__(self, 
                 custom_dict_path: Optional[str] = None,
                 stopwords_path: Optional[str] = None,
                 enable_userdict: bool = True):
        """
        初始化文本处理器
        
        Args:
            custom_dict_path: 自定义词典路径
            stopwords_path: 停用词文件路径
            enable_userdict: 是否启用用户词典
        """
        self.logger = logging.getLogger(__name__)
        
        # 初始化jieba分词器
        self._init_jieba(custom_dict_path, enable_userdict)
        
        # 加载停用词
        self.stopwords = self._load_stopwords(stopwords_path)
        
        # 定义清洗规则
        self._init_cleaning_rules()
        
        # 统计信息
        self.stats = {
            'processed_count': 0,
            'avg_length_before': 0,
            'avg_length_after': 0,
            'empty_texts': 0
        }
    
    def _init_jieba(self, custom_dict_path: Optional[str], enable_userdict: bool):
        """初始化jieba分词器"""
        try:
            # 加载自定义词典
            if custom_dict_path and enable_userdict:
                jieba.load_userdict(custom_dict_path)
                self.logger.info(f"已加载自定义词典: {custom_dict_path}")
            
            # 添加VOC相关专业词汇
            if enable_userdict:
                self._add_voc_vocabulary()
                
        except Exception as e:
            self.logger.warning(f"初始化jieba失败: {str(e)}")
    
    def _add_voc_vocabulary(self):
        """添加VOC和汽车行业相关词汇"""
        voc_words = [
            # 汽车相关
            '理想汽车', '理想ONE', '理想L9', '理想L8', '理想L7',
            '增程式', '纯电动', '混合动力', 'NEDC', 'WLTP',
            '自动驾驶', '辅助驾驶', 'NOA', 'AEB', 'ACC',
            
            # 用户体验相关
            '用户体验', '交互体验', '语音助手', '车机系统',
            '智能座舱', '智能网联', 'OTA升级', '远程控制',
            
            # 问题类型相关
            '续航里程', '充电速度', '充电桩', '能耗表现',
            '驾驶体验', '乘坐体验', '隔音效果', '悬挂调校',
            
            # 服务相关
            '销售顾问', '交付中心', '售后服务', '客服热线',
            '保养维修', '零配件', '质保期', '召回通知'
        ]
        
        for word in voc_words:
            jieba.add_word(word, freq=1000)  # 设置较高频率确保正确分词
    
    def _load_stopwords(self, stopwords_path: Optional[str]) -> set:
        """加载停用词表"""
        stopwords = set()
        
        # 默认停用词
        default_stopwords = {
            # 标点符号
            '。', '，', '、', '；', '：', '？', '！', '"', '"', ''', ''',
            '（', '）', '【', '】', '《', '》', '·', '…', '—', '－',
            '.', ',', ';', ':', '?', '!', '"', "'", '(', ')', '[', ']',
            '{', '}', '<', '>', '/', '\\', '|', '-', '_', '+', '=',
            '*', '&', '^', '%', '$', '#', '@', '~', '`',
            
            # 常见停用词
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那',
            '但是', '然后', '因为', '所以', '如果', '虽然', '或者',
            
            # 网络用语
            '哈哈', '呵呵', '嗯', '啊', '哦', '额', '嗯嗯', '哎',
            
            # 无意义词汇
            '东西', '什么', '怎么', '这样', '那样', '这种', '那种'
        }
        
        stopwords.update(default_stopwords)
        
        # 从文件加载停用词
        if stopwords_path:
            try:
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    file_stopwords = {line.strip() for line in f if line.strip()}
                    stopwords.update(file_stopwords)
                self.logger.info(f"已加载停用词文件: {stopwords_path}, 共{len(file_stopwords)}个")
            except Exception as e:
                self.logger.warning(f"加载停用词文件失败: {str(e)}")
        
        self.logger.info(f"停用词总数: {len(stopwords)}")
        return stopwords
    
    def _init_cleaning_rules(self):
        """初始化文本清洗规则"""
        # 编译正则表达式以提高性能
        self.patterns = {
            # 移除URL
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            
            # 移除邮箱
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            
            # 移除手机号
            'phone': re.compile(r'1[3-9]\d{9}'),
            
            # 移除多余空白字符
            'whitespace': re.compile(r'\s+'),
            
            # 移除重复标点
            'repeat_punct': re.compile(r'([。！？，、；：])\1+'),
            
            # 移除HTML标签
            'html': re.compile(r'<[^>]+>'),
            
            # 移除特殊字符（保留中文、英文、数字、基本标点）
            'special_chars': re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9。，！？、；：""''（）【】《》\s]'),
            
            # 移除emoji
            'emoji': re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+'),
            
            # 匹配纯数字
            'pure_number': re.compile(r'^\d+$'),
            
            # 匹配单个字符
            'single_char': re.compile(r'^.$')
        }
    
    def clean_text(self, text: str) -> str:
        """
        清洗单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            清洗后的文本
        """
        if not isinstance(text, str):
            return ""
        
        original_text = text
        
        # 1. 转换为小写（英文部分）
        # text = text.lower()  # 注释掉，保持原始大小写以保留语义信息
        
        # 2. Unicode标准化
        text = unicodedata.normalize('NFKC', text)
        
        # 3. 移除URL
        text = self.patterns['url'].sub('', text)
        
        # 4. 移除邮箱
        text = self.patterns['email'].sub('', text)
        
        # 5. 移除手机号
        text = self.patterns['phone'].sub('', text)
        
        # 6. 移除HTML标签
        text = self.patterns['html'].sub('', text)
        
        # 7. 移除emoji
        text = self.patterns['emoji'].sub('', text)
        
        # 8. 移除特殊字符
        text = self.patterns['special_chars'].sub('', text)
        
        # 9. 处理重复标点
        text = self.patterns['repeat_punct'].sub(r'\1', text)
        
        # 10. 统一空白字符
        text = self.patterns['whitespace'].sub(' ', text)
        
        # 11. 去除首尾空格
        text = text.strip()
        
        return text
    
    def segment_text(self, text: str, pos_filter: Optional[List[str]] = None) -> List[str]:
        """
        对文本进行分词
        
        Args:
            text: 输入文本
            pos_filter: 词性过滤列表，如['n', 'v', 'a']表示只保留名词、动词、形容词
            
        Returns:
            分词结果列表
        """
        if not text:
            return []
        
        try:
            if pos_filter:
                # 带词性标注的分词
                words = []
                for word, pos in pseg.cut(text):
                    if (pos[0] in pos_filter and  # 词性过滤
                        word not in self.stopwords and  # 停用词过滤
                        len(word.strip()) > 1 and  # 长度过滤
                        not self.patterns['pure_number'].match(word) and  # 纯数字过滤
                        not self.patterns['single_char'].match(word)):  # 单字符过滤
                        words.append(word.strip())
            else:
                # 普通分词
                words = []
                for word in jieba.cut(text):
                    if (word not in self.stopwords and
                        len(word.strip()) > 1 and
                        not self.patterns['pure_number'].match(word) and
                        not self.patterns['single_char'].match(word)):
                        words.append(word.strip())
            
            return words
            
        except Exception as e:
            self.logger.warning(f"分词失败: {str(e)}")
            return text.split()
    
    def preprocess(self, text: str, 
                   enable_segmentation: bool = True,
                   pos_filter: Optional[List[str]] = None,
                   min_length: int = 2,
                   max_length: int = 500) -> str:
        """
        完整的文本预处理
        
        Args:
            text: 输入文本
            enable_segmentation: 是否启用分词
            pos_filter: 词性过滤
            min_length: 最小文本长度
            max_length: 最大文本长度
            
        Returns:
            预处理后的文本
        """
        if not isinstance(text, str):
            return ""
        
        original_length = len(text)
        
        # 1. 基础清洗
        cleaned_text = self.clean_text(text)
        
        # 2. 长度检查
        if len(cleaned_text) < min_length:
            self.stats['empty_texts'] += 1
            return ""
        
        # 3. 截断过长文本
        if len(cleaned_text) > max_length:
            cleaned_text = cleaned_text[:max_length]
        
        # 4. 分词处理
        if enable_segmentation:
            words = self.segment_text(cleaned_text, pos_filter)
            processed_text = ' '.join(words) if words else ""
        else:
            processed_text = cleaned_text
        
        # 5. 更新统计信息
        self.stats['processed_count'] += 1
        self.stats['avg_length_before'] = (
            (self.stats['avg_length_before'] * (self.stats['processed_count'] - 1) + original_length) /
            self.stats['processed_count']
        )
        self.stats['avg_length_after'] = (
            (self.stats['avg_length_after'] * (self.stats['processed_count'] - 1) + len(processed_text)) /
            self.stats['processed_count']
        )
        
        return processed_text
    
    def batch_process(self, 
                     texts: List[str], 
                     enable_segmentation: bool = True,
                     pos_filter: Optional[List[str]] = None,
                     min_length: int = 2,
                     max_length: int = 500,
                     show_progress: bool = True) -> List[str]:
        """
        批量处理文本
        
        Args:
            texts: 文本列表
            enable_segmentation: 是否启用分词
            pos_filter: 词性过滤
            min_length: 最小文本长度
            max_length: 最大文本长度
            show_progress: 是否显示进度条
            
        Returns:
            处理后的文本列表
        """
        if not texts:
            return []
        
        # 重置统计信息
        self.stats = {
            'processed_count': 0,
            'avg_length_before': 0,
            'avg_length_after': 0,
            'empty_texts': 0
        }
        
        processed_texts = []
        
        # 创建进度条
        iterator = tqdm(texts, desc="文本预处理", disable=not show_progress, ncols=80)
        
        for text in iterator:
            processed_text = self.preprocess(
                text, 
                enable_segmentation=enable_segmentation,
                pos_filter=pos_filter,
                min_length=min_length,
                max_length=max_length
            )
            processed_texts.append(processed_text)
            
            # 更新进度条描述
            if show_progress and self.stats['processed_count'] % 100 == 0:
                iterator.set_postfix({
                    '有效': f"{self.stats['processed_count'] - self.stats['empty_texts']}/{self.stats['processed_count']}",
                    '平均长度': f"{self.stats['avg_length_after']:.1f}"
                })
        
        # 打印统计信息
        self._print_stats()
        
        return processed_texts
    
    def extract_keywords(self, 
                        texts: List[str], 
                        top_k: int = 20,
                        min_freq: int = 2) -> Dict[str, int]:
        """
        从文本集合中提取关键词
        
        Args:
            texts: 文本列表
            top_k: 返回前k个关键词
            min_freq: 最小词频
            
        Returns:
            关键词及其频率的字典
        """
        all_words = []
        
        for text in tqdm(texts, desc="提取关键词", ncols=80):
            words = self.segment_text(text, pos_filter=['n', 'v', 'a'])  # 只保留名词、动词、形容词
            all_words.extend(words)
        
        # 统计词频
        word_freq = Counter(all_words)
        
        # 过滤低频词
        filtered_freq = {word: freq for word, freq in word_freq.items() if freq >= min_freq}
        
        # 返回top_k个关键词
        return dict(Counter(filtered_freq).most_common(top_k))
    
    def analyze_text_quality(self, texts: List[str]) -> Dict[str, Union[int, float]]:
        """
        分析文本质量
        
        Args:
            texts: 文本列表
            
        Returns:
            文本质量分析结果
        """
        if not texts:
            return {}
        
        # 统计指标
        total_texts = len(texts)
        non_empty_texts = sum(1 for text in texts if text.strip())
        total_chars = sum(len(text) for text in texts)
        total_words = sum(len(self.segment_text(text)) for text in texts if text.strip())
        
        # 长度分布
        lengths = [len(text) for text in texts]
        
        return {
            'total_texts': total_texts,
            'non_empty_texts': non_empty_texts,
            'empty_rate': (total_texts - non_empty_texts) / total_texts * 100,
            'avg_chars': total_chars / total_texts if total_texts > 0 else 0,
            'avg_words': total_words / non_empty_texts if non_empty_texts > 0 else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'median_length': sorted(lengths)[len(lengths)//2] if lengths else 0
        }
    
    def _print_stats(self):
        """打印处理统计信息"""
        if self.stats['processed_count'] > 0:
            print(f"\n📊 文本处理统计:")
            print(f"  总处理数量: {self.stats['processed_count']}")
            print(f"  有效文本: {self.stats['processed_count'] - self.stats['empty_texts']}")
            print(f"  空文本数量: {self.stats['empty_texts']}")
            print(f"  平均长度(处理前): {self.stats['avg_length_before']:.1f}")
            print(f"  平均长度(处理后): {self.stats['avg_length_after']:.1f}")
            print(f"  处理效率: {((self.stats['processed_count'] - self.stats['empty_texts']) / self.stats['processed_count'] * 100):.1f}%")
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """获取处理统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'processed_count': 0,
            'avg_length_before': 0,
            'avg_length_after': 0,
            'empty_texts': 0
        }
    
    def save_processed_data(self, 
                           original_texts: List[str], 
                           processed_texts: List[str], 
                           output_path: str):
        """
        保存处理结果到文件
        
        Args:
            original_texts: 原始文本列表
            processed_texts: 处理后文本列表
            output_path: 输出文件路径
        """
        try:
            df = pd.DataFrame({
                'original_text': original_texts,
                'processed_text': processed_texts,
                'original_length': [len(text) for text in original_texts],
                'processed_length': [len(text) for text in processed_texts]
            })
            
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✅ 处理结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"❌ 保存失败: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 创建处理器实例
    processor = TextProcessor(enable_userdict=True)
    
    # 示例文本
    test_texts = [
        "理想汽车的语音助手真的很智能！！！但是有时候识别不准确😅",
        "充电速度还可以，但是充电桩太少了，希望能多建一些。",
        "驾驶体验非常好，座椅很舒适，隔音效果也不错👍👍👍",
        "OTA升级后系统更流畅了，但是偶尔还是会卡顿。",
        ""  # 空文本测试
    ]
    
    # 批量处理
    processed_texts = processor.batch_process(test_texts)
    
    # 显示结果
    print("\n🔍 处理结果对比:")
    for i, (original, processed) in enumerate(zip(test_texts, processed_texts)):
        if original:  # 跳过空文本
            print(f"\n{i+1}. 原文: {original}")
            print(f"   处理后: {processed}")
    
    # 提取关键词
    keywords = processor.extract_keywords(processed_texts, top_k=10)
    print(f"\n🔑 关键词提取: {keywords}")
    
    # 分析文本质量
    quality = processor.analyze_text_quality(test_texts)
    print(f"\n📈 文本质量分析: {quality}")