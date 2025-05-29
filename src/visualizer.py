"""
数据可视化工具 - 改进版
专为VOC分析设计，支持多维度图表、交互式可视化、报告生成等功能
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import List, Dict, Optional, Tuple, Union, Any
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class VOCVisualizer:
    """
    VOC数据可视化器
    专为理想汽车用户反馈分析设计的可视化工具
    """
    
    def __init__(self, style: str = 'whitegrid', palette: str = 'Set2'):
        """
        初始化可视化工具
        
        Args:
            style: seaborn样式
            palette: 颜色方案
        """
        # 设置绘图样式
        sns.set_style(style)
        sns.set_palette(palette)
        
        # 定义理想汽车品牌色系
        self.brand_colors = {
            'primary': '#0066CC',      # 理想蓝
            'secondary': '#FF6B35',    # 活力橙
            'success': '#28A745',      # 成功绿
            'warning': '#FFC107',      # 警告黄
            'danger': '#DC3545',       # 危险红
            'info': '#17A2B8',         # 信息青
            'light': '#F8F9FA',        # 浅灰
            'dark': '#343A40'          # 深灰
        }
        
        # VOC专用配色方案
        self.voc_colors = {
            '正面': self.brand_colors['success'],
            '中性': self.brand_colors['warning'], 
            '负面': self.brand_colors['danger'],
            '用户旅程': self.brand_colors['primary'],
            '问题类型': self.brand_colors['secondary']
        }
        
        # 图表计数器
        self.chart_counter = 0
        
    def _get_next_filename(self, base_name: str, extension: str = '.png') -> str:
        """生成下一个文件名"""
        self.chart_counter += 1
        return f"{base_name}_{self.chart_counter:02d}{extension}"
    
    def plot_sentiment_distribution(self, 
                                   sentiments: List[str], 
                                   title: str = "情感分布分析",
                                   figsize: Tuple[int, int] = (10, 6),
                                   show_percentages: bool = True) -> plt.Figure:
        """
        绘制情感分布图
        
        Args:
            sentiments: 情感标签列表
            title: 图表标题
            figsize: 图表大小
            show_percentages: 是否显示百分比
            
        Returns:
            matplotlib图表对象
        """
        # 统计情感分布
        sentiment_counts = Counter([s for s in sentiments if s])
        
        if not sentiment_counts:
            return self._create_empty_chart(title, "暂无情感数据")
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 饼图
        labels = list(sentiment_counts.keys())
        values = list(sentiment_counts.values())
        colors = [self.voc_colors.get(label, self.brand_colors['info']) for label in labels]
        
        wedges, texts, autotexts = ax1.pie(values, labels=labels, colors=colors, 
                                          autopct='%1.1f%%' if show_percentages else None,
                                          startangle=90)
        ax1.set_title(f"{title} - 占比分布")
        
        # 条形图
        bars = ax2.bar(labels, values, color=colors)
        ax2.set_title(f"{title} - 数量分布")
        ax2.set_xlabel("情感类型")
        ax2.set_ylabel("数量")
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_journey_touchpoints(self, 
                                touchpoints: List[str], 
                                title: str = "用户旅程触点分析",
                                figsize: Tuple[int, int] = (14, 8),
                                top_n: int = 15) -> plt.Figure:
        """
        绘制用户旅程触点分布图
        
        Args:
            touchpoints: 触点标签列表（支持逗号分隔的多标签）
            title: 图表标题
            figsize: 图表大小
            top_n: 显示前N个触点
            
        Returns:
            matplotlib图表对象
        """
        # 展开多标签数据
        all_touchpoints = []
        for tp in touchpoints:
            if tp and isinstance(tp, str):
                # 分割逗号分隔的标签
                labels = [label.strip() for label in tp.split(',') if label.strip()]
                all_touchpoints.extend(labels)
        
        if not all_touchpoints:
            return self._create_empty_chart(title, "暂无用户旅程数据")
        
        # 统计频率
        tp_counts = Counter(all_touchpoints)
        top_touchpoints = tp_counts.most_common(top_n)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        labels = [item[0] for item in top_touchpoints]
        values = [item[1] for item in top_touchpoints]
        
        # 创建渐变色
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(labels)))
        
        # 水平条形图
        bars = ax.barh(range(len(labels)), values, color=colors)
        
        # 设置标签
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("提及次数")
        ax.set_title(title)
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value}', ha='left', va='center')
        
        # 反转y轴，使频率最高的在顶部
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def plot_problem_categories(self, 
                               problem_types: List[str],
                               title: str = "问题类型分析",
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        绘制问题类型分布图
        
        Args:
            problem_types: 问题类型列表
            title: 图表标题
            figsize: 图表大小
            
        Returns:
            matplotlib图表对象
        """
        # 展开多标签数据
        all_problems = []
        for pt in problem_types:
            if pt and isinstance(pt, str):
                labels = [label.strip() for label in pt.split(',') if label.strip()]
                all_problems.extend(labels)
        
        if not all_problems:
            return self._create_empty_chart(title, "暂无问题类型数据")
        
        # 统计频率
        problem_counts = Counter(all_problems)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        labels = list(problem_counts.keys())
        values = list(problem_counts.values())
        
        # 使用橙色系配色
        colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(labels)))
        
        # 创建条形图
        bars = ax.bar(labels, values, color=colors)
        
        # 设置标签
        ax.set_xlabel("问题类型")
        ax.set_ylabel("出现次数")
        ax.set_title(title)
        
        # 旋转x轴标签避免重叠
        plt.xticks(rotation=45, ha='right')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_confidence_distribution(self, 
                                   confidences: List[float],
                                   title: str = "预测置信度分布",
                                   figsize: Tuple[int, int] = (10, 6),
                                   bins: int = 20) -> plt.Figure:
        """
        绘制置信度分布图
        
        Args:
            confidences: 置信度列表
            title: 图表标题
            figsize: 图表大小
            bins: 直方图分箱数
            
        Returns:
            matplotlib图表对象
        """
        if not confidences:
            return self._create_empty_chart(title, "暂无置信度数据")
        
        # 过滤有效置信度值
        valid_confidences = [c for c in confidences if isinstance(c, (int, float)) and 0 <= c <= 1]
        
        if not valid_confidences:
            return self._create_empty_chart(title, "暂无有效置信度数据")
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 直方图
        ax1.hist(valid_confidences, bins=bins, color=self.brand_colors['primary'], 
                alpha=0.7, edgecolor='black')
        ax1.set_xlabel("置信度")
        ax1.set_ylabel("频率")
        ax1.set_title(f"{title} - 分布直方图")
        ax1.grid(True, alpha=0.3)
        
        # 箱线图
        box_plot = ax2.boxplot(valid_confidences, patch_artist=True)
        box_plot['boxes'][0].set_facecolor(self.brand_colors['primary'])
        ax2.set_ylabel("置信度")
        ax2.set_title(f"{title} - 箱线图")
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_conf = np.mean(valid_confidences)
        median_conf = np.median(valid_confidences)
        std_conf = np.std(valid_confidences)
        
        stats_text = f"均值: {mean_conf:.3f}\n中位数: {median_conf:.3f}\n标准差: {std_conf:.3f}"
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_sentiment_journey_heatmap(self, 
                                     touchpoints: List[str],
                                     sentiments: List[str],
                                     title: str = "用户旅程-情感热力图",
                                     figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        绘制用户旅程与情感的关联热力图
        
        Args:
            touchpoints: 触点列表
            sentiments: 情感列表
            title: 图表标题
            figsize: 图表大小
            
        Returns:
            matplotlib图表对象
        """
        # 构建数据矩阵
        sentiment_touchpoint_matrix = defaultdict(lambda: defaultdict(int))
        
        for tp, sentiment in zip(touchpoints, sentiments):
            if tp and sentiment:
                # 处理多标签触点
                if isinstance(tp, str):
                    tp_list = [label.strip() for label in tp.split(',') if label.strip()]
                    for touchpoint in tp_list:
                        sentiment_touchpoint_matrix[sentiment][touchpoint] += 1
        
        if not sentiment_touchpoint_matrix:
            return self._create_empty_chart(title, "暂无关联数据")
        
        # 转换为DataFrame
        df_matrix = pd.DataFrame(sentiment_touchpoint_matrix).fillna(0)
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=figsize)
        
        # 使用自定义颜色映射
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        
        sns.heatmap(df_matrix.T, annot=True, fmt='g', cmap=cmap, 
                   cbar_kws={'label': '提及次数'}, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel("情感类型")
        ax.set_ylabel("用户旅程触点")
        
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return fig
    
    def plot_time_series_sentiment(self, 
                                  timestamps: List[str],
                                  sentiments: List[str],
                                  title: str = "情感趋势分析",
                                  figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        绘制情感时间序列图
        
        Args:
            timestamps: 时间戳列表
            sentiments: 情感列表
            title: 图表标题
            figsize: 图表大小
            
        Returns:
            matplotlib图表对象
        """
        try:
            # 创建DataFrame
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps),
                'sentiment': sentiments
            })
            
            # 按时间分组统计
            df_grouped = df.groupby([df['timestamp'].dt.date, 'sentiment']).size().unstack(fill_value=0)
            
            # 创建图表
            fig, ax = plt.subplots(figsize=figsize)
            
            # 绘制堆叠面积图
            df_grouped.plot.area(ax=ax, color=[self.voc_colors.get(col, self.brand_colors['info']) 
                                              for col in df_grouped.columns])
            
            ax.set_title(title)
            ax.set_xlabel("日期")
            ax.set_ylabel("评论数量")
            ax.legend(title="情感类型")
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            return self._create_empty_chart(title, f"时间序列数据处理失败: {str(e)}")
    
    def create_interactive_dashboard(self, 
                                   data: Dict[str, List],
                                   output_path: str = "voc_dashboard.html") -> str:
        """
        创建交互式VOC分析仪表板
        
        Args:
            data: 包含各种数据的字典
            output_path: 输出HTML文件路径
            
        Returns:
            生成的HTML文件路径
        """
        # 创建子图布局
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('情感分布', '用户旅程触点', '问题类型分布', 
                          '置信度分布', '情感-触点关联', '数据质量'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # 1. 情感分布饼图
        if 'sentiments' in data:
            sentiment_counts = Counter([s for s in data['sentiments'] if s])
            if sentiment_counts:
                fig.add_trace(
                    go.Pie(labels=list(sentiment_counts.keys()),
                          values=list(sentiment_counts.values()),
                          name="情感分布"),
                    row=1, col=1
                )
        
        # 2. 用户旅程触点
        if 'touchpoints' in data:
            all_touchpoints = []
            for tp in data['touchpoints']:
                if tp:
                    all_touchpoints.extend([t.strip() for t in str(tp).split(',') if t.strip()])
            
            if all_touchpoints:
                tp_counts = Counter(all_touchpoints)
                top_tp = dict(tp_counts.most_common(10))
                
                fig.add_trace(
                    go.Bar(x=list(top_tp.values()),
                          y=list(top_tp.keys()),
                          orientation='h',
                          name="触点频率"),
                    row=1, col=2
                )
        
        # 3. 问题类型分布
        if 'problem_types' in data:
            all_problems = []
            for pt in data['problem_types']:
                if pt:
                    all_problems.extend([p.strip() for p in str(pt).split(',') if p.strip()])
            
            if all_problems:
                problem_counts = Counter(all_problems)
                
                fig.add_trace(
                    go.Bar(x=list(problem_counts.keys()),
                          y=list(problem_counts.values()),
                          name="问题类型"),
                    row=2, col=1
                )
        
        # 4. 置信度分布
        if 'confidences' in data:
            valid_conf = [c for c in data['confidences'] if isinstance(c, (int, float))]
            if valid_conf:
                fig.add_trace(
                    go.Histogram(x=valid_conf,
                               name="置信度分布"),
                    row=2, col=2
                )
        
        # 更新布局
        fig.update_layout(
            title_text="VOC智能分析仪表板",
            title_x=0.5,
            height=1200,
            showlegend=False
        )
        
        # 保存为HTML
        pyo.plot(fig, filename=output_path, auto_open=False)
        
        return output_path
    
    def generate_report(self, 
                       df: pd.DataFrame,
                       output_dir: str,
                       report_title: str = "VOC分析报告") -> Dict[str, str]:
        """
        生成完整的VOC分析报告
        
        Args:
            df: 包含VOC数据的DataFrame
            output_dir: 输出目录
            report_title: 报告标题
            
        Returns:
            生成的文件路径字典
        """
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = {}
        
        try:
            # 1. 情感分析图
            if '情感' in df.columns:
                fig = self.plot_sentiment_distribution(df['情感'].tolist())
                sentiment_path = os.path.join(output_dir, "sentiment_analysis.png")
                fig.savefig(sentiment_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                generated_files['sentiment'] = sentiment_path
            
            # 2. 用户旅程分析图
            if '旅程触点' in df.columns:
                fig = self.plot_journey_touchpoints(df['旅程触点'].tolist())
                journey_path = os.path.join(output_dir, "journey_analysis.png")
                fig.savefig(journey_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                generated_files['journey'] = journey_path
            
            # 3. 问题类型分析图
            if '问题类型' in df.columns:
                fig = self.plot_problem_categories(df['问题类型'].tolist())
                problem_path = os.path.join(output_dir, "problem_analysis.png")
                fig.savefig(problem_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                generated_files['problems'] = problem_path
            
            # 4. 置信度分析图
            confidence_cols = [col for col in df.columns if 'confidence' in col.lower()]
            if confidence_cols:
                all_confidences = []
                for col in confidence_cols:
                    valid_conf = pd.to_numeric(df[col], errors='coerce').dropna().tolist()
                    all_confidences.extend(valid_conf)
                
                if all_confidences:
                    fig = self.plot_confidence_distribution(all_confidences)
                    conf_path = os.path.join(output_dir, "confidence_analysis.png")
                    fig.savefig(conf_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    generated_files['confidence'] = conf_path
            
            # 5. 关联分析热力图
            if '旅程触点' in df.columns and '情感' in df.columns:
                fig = self.plot_sentiment_journey_heatmap(
                    df['旅程触点'].tolist(),
                    df['情感'].tolist()
                )
                heatmap_path = os.path.join(output_dir, "sentiment_journey_heatmap.png")
                fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                generated_files['heatmap'] = heatmap_path
            
            # 6. 交互式仪表板
            dashboard_data = {}
            if '情感' in df.columns:
                dashboard_data['sentiments'] = df['情感'].tolist()
            if '旅程触点' in df.columns:
                dashboard_data['touchpoints'] = df['旅程触点'].tolist()
            if '问题类型' in df.columns:
                dashboard_data['problem_types'] = df['问题类型'].tolist()
            
            if dashboard_data:
                dashboard_path = os.path.join(output_dir, "interactive_dashboard.html")
                self.create_interactive_dashboard(dashboard_data, dashboard_path)
                generated_files['dashboard'] = dashboard_path
            
            # 7. 生成HTML报告摘要
            summary_path = self._generate_html_summary(df, output_dir, report_title, generated_files)
            generated_files['summary'] = summary_path
            
            print(f"✅ 报告生成完成！共生成 {len(generated_files)} 个文件")
            print(f"📁 报告目录: {output_dir}")
            
            return generated_files
            
        except Exception as e:
            print(f"❌ 报告生成失败: {str(e)}")
            return generated_files
    
    def _generate_html_summary(self, 
                              df: pd.DataFrame, 
                              output_dir: str,
                              report_title: str,
                              generated_files: Dict[str, str]) -> str:
        """生成HTML报告摘要"""
        
        # 计算统计信息
        total_records = len(df)
        
        # 情感统计
        sentiment_stats = {}
        if '情感' in df.columns:
            sentiment_counts = df['情感'].value_counts()
            sentiment_stats = sentiment_counts.to_dict()
        
        # 触点统计
        touchpoint_stats = {}
        if '旅程触点' in df.columns:
            all_touchpoints = []
            for tp in df['旅程触点'].dropna():
                if tp:
                    all_touchpoints.extend([t.strip() for t in str(tp).split(',') if t.strip()])
            touchpoint_stats = dict(Counter(all_touchpoints).most_common(5))
        
        # 问题类型统计
        problem_stats = {}
        if '问题类型' in df.columns:
            all_problems = []
            for pt in df['问题类型'].dropna():
                if pt:
                    all_problems.extend([p.strip() for p in str(pt).split(',') if p.strip()])
            problem_stats = dict(Counter(all_problems).most_common(5))
        
        # HTML模板
        html_template = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{report_title}</title>
            <style>
                body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #0066CC, #FF6B35); color: white; padding: 30px; border-radius: 10px; }}
                .content {{ margin: 20px 0; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #0066CC; }}
                .chart-gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
                .chart-item {{ text-align: center; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .chart-item img {{ max-width: 100%; height: auto; border-radius: 4px; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
                h2 {{ color: #0066CC; border-bottom: 2px solid #0066CC; padding-bottom: 5px; }}
                .highlight {{ background: #fff3cd; padding: 10px; border-radius: 4px; border-left: 4px solid #ffc107; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚗 {report_title}</h1>
                <p>理想汽车用户反馈智能分析系统</p>
                <p class="timestamp">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="content">
                <h2>📊 数据概览</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>📈 总体统计</h3>
                        <p>总记录数: <strong>{total_records:,}</strong></p>
                        <p>分析维度: <strong>{len([col for col in df.columns if col in ['情感', '旅程触点', '问题类型']])}</strong></p>
                    </div>
                    
                    <div class="stat-card">
                        <h3>😊 情感分布</h3>
                        {''.join([f'<p>{k}: <strong>{v}</strong> ({v/sum(sentiment_stats.values())*100:.1f}%)</p>' for k, v in sentiment_stats.items()]) if sentiment_stats else '<p>暂无数据</p>'}
                    </div>
                    
                    <div class="stat-card">
                        <h3>🎯 热门触点</h3>
                        {''.join([f'<p>{k}: <strong>{v}</strong></p>' for k, v in list(touchpoint_stats.items())[:3]]) if touchpoint_stats else '<p>暂无数据</p>'}
                    </div>
                    
                    <div class="stat-card">
                        <h3>⚠️ 主要问题</h3>
                        {''.join([f'<p>{k}: <strong>{v}</strong></p>' for k, v in list(problem_stats.items())[:3]]) if problem_stats else '<p>暂无数据</p>'}
                    </div>
                </div>
                
                <h2>📈 可视化分析</h2>
                <div class="chart-gallery">
                    {''.join([f'''
                    <div class="chart-item">
                        <h3>{title}</h3>
                        <img src="{os.path.basename(path)}" alt="{title}">
                    </div>
                    ''' for title, path in [
                        ('情感分布分析', generated_files.get('sentiment', '')),
                        ('用户旅程分析', generated_files.get('journey', '')),
                        ('问题类型分析', generated_files.get('problems', '')),
                        ('置信度分析', generated_files.get('confidence', '')),
                        ('关联分析热力图', generated_files.get('heatmap', ''))
                    ] if path])}
                </div>
                
                <div class="highlight">
                    <h3>🎯 关键洞察</h3>
                    <ul>
                        <li>共分析了 <strong>{total_records:,}</strong> 条用户反馈</li>
                        {f'<li>情感倾向: {max(sentiment_stats, key=sentiment_stats.get) if sentiment_stats else "未知"} 占主导地位</li>' if sentiment_stats else ''}
                        {f'<li>最受关注的触点: <strong>{list(touchpoint_stats.keys())[0] if touchpoint_stats else "无"}</strong></li>' if touchpoint_stats else ''}
                        {f'<li>最突出的问题: <strong>{list(problem_stats.keys())[0] if problem_stats else "无"}</strong></li>' if problem_stats else ''}
                        <li>详细的交互式分析请查看: <a href="interactive_dashboard.html" target="_blank">交互式仪表板</a></li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        # 保存HTML文件
        summary_path = os.path.join(output_dir, "report_summary.html")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        return summary_path
    
    def _create_empty_chart(self, title: str, message: str) -> plt.Figure:
        """创建空图表"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, message, transform=ax.transAxes, 
               fontsize=16, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title(title)
        ax.axis('off')
        return fig
    
    def save_figure(self, 
                   fig: plt.Figure, 
                   output_path: str, 
                   dpi: int = 300,
                   format: str = 'png') -> bool:
        """
        保存图表到文件
        
        Args:
            fig: matplotlib图表对象
            output_path: 输出路径
            dpi: 分辨率
            format: 输出格式
            
        Returns:
            是否保存成功
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存图表
            fig.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            print(f"✅ 图表已保存: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ 保存图表失败: {str(e)}")
            return False
        finally:
            plt.close(fig)


# 使用示例
if __name__ == "__main__":
    # 创建可视化器
    visualizer = VOCVisualizer()
    
    # 示例数据
    sample_data = {
        'sentiments': ['正面', '负面', '正面', '中性', '负面', '正面'],
        'touchpoints': ['智能导航,语音助手', '充电与能耗', '交付培训', '销售服务', '智能导航', '客服响应'],
        'problem_types': ['稳定性问题,可用性问题', '', '服务体验问题', '', '稳定性问题', ''],
        'confidences': [0.85, 0.92, 0.78, 0.65, 0.88, 0.91]
    }
    
    # 创建DataFrame
    df = pd.DataFrame(sample_data)
    
    # 生成报告
    output_dir = "sample_report"
    generated_files = visualizer.generate_report(df, output_dir, "VOC分析示例报告")
    
    print(f"\n📋 生成的文件:")
    for file_type, file_path in generated_files.items():
        print(f"  {file_type}: {file_path}")