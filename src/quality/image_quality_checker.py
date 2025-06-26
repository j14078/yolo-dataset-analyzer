"""
画像品質チェックエンジン
初心者向け：分かりやすい結果表示
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageStat
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json


class ImageQualityChecker:
    """画像品質チェック機能"""
    
    def __init__(self):
        self.quality_thresholds = {
            # 解像度チェック
            'min_width': 320,
            'min_height': 320,
            'recommended_width': 640,
            'recommended_height': 640,
            
            # 明度チェック
            'brightness_min': 30,   # 暗すぎる
            'brightness_max': 225,  # 明るすぎる
            'brightness_optimal_min': 50,
            'brightness_optimal_max': 200,
            
            # コントラストチェック
            'contrast_min': 20,     # 低コントラスト
            'contrast_optimal': 50,
            
            # ぼけ検出
            'blur_threshold': 100,  # Laplacian variance
            'blur_warning': 200,
            
            # ファイルサイズ
            'file_size_min': 5 * 1024,      # 5KB未満は小さすぎ
            'file_size_max': 10 * 1024 * 1024,  # 10MB超は大きすぎ
            
            # アスペクト比
            'aspect_ratio_min': 0.5,  # 縦長すぎ
            'aspect_ratio_max': 2.0,  # 横長すぎ
        }
    
    def check_dataset_quality(self, folder_path: str, 
                            image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')) -> Dict:
        """
        データセット全体の品質チェック
        
        Args:
            folder_path: 画像フォルダパス
            image_extensions: チェック対象の拡張子
            
        Returns:
            品質チェック結果
        """
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"フォルダが見つかりません: {folder_path}")
        
        # 画像ファイル一覧取得
        image_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(image_extensions):
                image_files.append(file)
        
        if not image_files:
            return {
                'エラー': True,
                'メッセージ': '画像ファイルが見つかりません',
                '詳細': f'対応形式: {image_extensions}'
            }
        
        # 品質チェック実行
        results = {
            'エラー': False,
            '総画像数': len(image_files),
            '処理完了数': 0,
            '品質問題': {
                '解像度不足': [],
                '明度問題': [],
                'コントラスト不足': [],
                'ぼけ画像': [],
                'ファイルサイズ異常': [],
                'アスペクト比異常': [],
                '読み込みエラー': []
            },
            '品質統計': {
                '平均解像度': {'width': 0, 'height': 0},
                '平均明度': 0,
                '平均コントラスト': 0,
                '平均ファイルサイズ': 0,
                '解像度分布': defaultdict(int),
                'アスペクト比分布': defaultdict(int)
            },
            '詳細結果': [],
            '推奨改善': []
        }
        
        # 統計用変数
        total_width, total_height = 0, 0
        total_brightness, total_contrast = 0, 0
        total_file_size = 0
        valid_images = 0
        
        # 各画像をチェック
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(folder_path, image_file)
            
            try:
                # 個別画像チェック
                quality_result = self.check_single_image(image_path)
                
                if not quality_result['エラー']:
                    valid_images += 1
                    
                    # 統計更新
                    width, height = quality_result['解像度']['width'], quality_result['解像度']['height']
                    total_width += width
                    total_height += height
                    total_brightness += quality_result['明度']
                    total_contrast += quality_result['コントラスト']
                    total_file_size += quality_result['ファイルサイズ']
                    
                    # 分布更新
                    resolution_key = f"{width}x{height}"
                    results['品質統計']['解像度分布'][resolution_key] += 1
                    
                    aspect_ratio = width / height
                    aspect_key = f"{aspect_ratio:.1f}:1"
                    results['品質統計']['アスペクト比分布'][aspect_key] += 1
                    
                    # 問題のチェック
                    self._categorize_quality_issues(quality_result, results['品質問題'], image_file)
                    
                    # 詳細結果に追加
                    results['詳細結果'].append({
                        'ファイル名': image_file,
                        **quality_result
                    })
                
                else:
                    results['品質問題']['読み込みエラー'].append({
                        'ファイル名': image_file,
                        'エラー': quality_result['エラーメッセージ']
                    })
                
                results['処理完了数'] = i + 1
                
            except Exception as e:
                results['品質問題']['読み込みエラー'].append({
                    'ファイル名': image_file,
                    'エラー': str(e)
                })
        
        # 統計計算
        if valid_images > 0:
            results['品質統計']['平均解像度'] = {
                'width': int(total_width / valid_images),
                'height': int(total_height / valid_images)
            }
            results['品質統計']['平均明度'] = round(total_brightness / valid_images, 1)
            results['品質統計']['平均コントラスト'] = round(total_contrast / valid_images, 1)
            results['品質統計']['平均ファイルサイズ'] = int(total_file_size / valid_images)
        
        # 改善提案生成
        results['推奨改善'] = self._generate_recommendations(results['品質問題'], results['品質統計'])
        
        return results
    
    def check_single_image(self, image_path: str) -> Dict:
        """
        単一画像の品質チェック
        
        Args:
            image_path: 画像ファイルパス
            
        Returns:
            品質チェック結果
        """
        
        try:
            # 基本情報取得
            file_size = os.path.getsize(image_path)
            
            # PIL で画像読み込み
            with Image.open(image_path) as pil_img:
                width, height = pil_img.size
                mode = pil_img.mode
                
                # RGB変換（統計計算用）
                if mode != 'RGB':
                    rgb_img = pil_img.convert('RGB')
                else:
                    rgb_img = pil_img
                
                # 明度・コントラスト計算
                stat = ImageStat.Stat(rgb_img)
                brightness = sum(stat.mean) / len(stat.mean)  # RGB平均
                contrast = sum(stat.stddev) / len(stat.stddev)  # 標準偏差平均
            
            # OpenCV でぼけ検出
            cv_img = cv2.imread(image_path)
            if cv_img is not None:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            else:
                blur_score = 0
            
            # アスペクト比計算
            aspect_ratio = width / height if height > 0 else 0
            
            return {
                'エラー': False,
                '解像度': {'width': width, 'height': height},
                'ファイルサイズ': file_size,
                '明度': round(brightness, 1),
                'コントラスト': round(contrast, 1),
                'ぼけスコア': round(blur_score, 1),
                'アスペクト比': round(aspect_ratio, 2),
                'フォーマット': mode,
                'ファイルパス': image_path
            }
            
        except Exception as e:
            return {
                'エラー': True,
                'エラーメッセージ': str(e),
                'ファイルパス': image_path
            }
    
    def _categorize_quality_issues(self, quality_result: Dict, issues: Dict, filename: str):
        """品質問題の分類"""
        
        thresholds = self.quality_thresholds
        
        # 解像度チェック
        width, height = quality_result['解像度']['width'], quality_result['解像度']['height']
        if width < thresholds['min_width'] or height < thresholds['min_height']:
            issues['解像度不足'].append({
                'ファイル名': filename,
                '現在の解像度': f"{width}x{height}",
                '推奨解像度': f"{thresholds['recommended_width']}x{thresholds['recommended_height']}以上",
                '重要度': 'high' if width < 224 or height < 224 else 'medium'
            })
        
        # 明度チェック
        brightness = quality_result['明度']
        if brightness < thresholds['brightness_min']:
            issues['明度問題'].append({
                'ファイル名': filename,
                '問題': '暗すぎる',
                '明度値': brightness,
                '推奨範囲': f"{thresholds['brightness_optimal_min']}-{thresholds['brightness_optimal_max']}",
                '重要度': 'high' if brightness < 20 else 'medium'
            })
        elif brightness > thresholds['brightness_max']:
            issues['明度問題'].append({
                'ファイル名': filename,
                '問題': '明るすぎる',
                '明度値': brightness,
                '推奨範囲': f"{thresholds['brightness_optimal_min']}-{thresholds['brightness_optimal_max']}",
                '重要度': 'high' if brightness > 240 else 'medium'
            })
        
        # コントラストチェック
        contrast = quality_result['コントラスト']
        if contrast < thresholds['contrast_min']:
            issues['コントラスト不足'].append({
                'ファイル名': filename,
                'コントラスト値': contrast,
                '推奨値': f"{thresholds['contrast_optimal']}以上",
                '重要度': 'high' if contrast < 10 else 'medium'
            })
        
        # ぼけチェック
        blur_score = quality_result['ぼけスコア']
        if blur_score < thresholds['blur_threshold']:
            issues['ぼけ画像'].append({
                'ファイル名': filename,
                'ぼけスコア': blur_score,
                '推奨値': f"{thresholds['blur_warning']}以上",
                '重要度': 'high' if blur_score < 50 else 'medium'
            })
        
        # ファイルサイズチェック
        file_size = quality_result['ファイルサイズ']
        if file_size < thresholds['file_size_min']:
            issues['ファイルサイズ異常'].append({
                'ファイル名': filename,
                '問題': 'ファイルサイズが小さすぎる',
                'サイズ': f"{file_size / 1024:.1f}KB",
                '重要度': 'medium'
            })
        elif file_size > thresholds['file_size_max']:
            issues['ファイルサイズ異常'].append({
                'ファイル名': filename,
                '問題': 'ファイルサイズが大きすぎる',
                'サイズ': f"{file_size / (1024*1024):.1f}MB",
                '重要度': 'low'
            })
        
        # アスペクト比チェック
        aspect_ratio = quality_result['アスペクト比']
        if aspect_ratio < thresholds['aspect_ratio_min'] or aspect_ratio > thresholds['aspect_ratio_max']:
            issues['アスペクト比異常'].append({
                'ファイル名': filename,
                'アスペクト比': f"{aspect_ratio:.2f}:1",
                '推奨範囲': f"{thresholds['aspect_ratio_min']}-{thresholds['aspect_ratio_max']}:1",
                '重要度': 'low'
            })
    
    def _generate_recommendations(self, issues: Dict, stats: Dict) -> List[str]:
        """改善提案を生成"""
        
        recommendations = []
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        
        if total_issues == 0:
            recommendations.append("🎉 品質チェック完了！問題のある画像は見つかりませんでした。")
            return recommendations
        
        # 優先度の高い問題から提案
        if issues['解像度不足']:
            count = len(issues['解像度不足'])
            recommendations.append(f"📐 解像度不足の画像が{count}枚あります。リサイズまたは再撮影を検討してください。")
        
        if issues['ぼけ画像']:
            count = len(issues['ぼけ画像'])
            recommendations.append(f"🌫️ ぼけている画像が{count}枚あります。ピントが合った画像に差し替えることをお勧めします。")
        
        if issues['明度問題']:
            count = len(issues['明度問題'])
            recommendations.append(f"💡 明度に問題のある画像が{count}枚あります。露出補正や照明調整を検討してください。")
        
        if issues['コントラスト不足']:
            count = len(issues['コントラスト不足'])
            recommendations.append(f"🎨 コントラストが低い画像が{count}枚あります。画像編集でコントラストを上げてください。")
        
        if issues['ファイルサイズ異常']:
            count = len(issues['ファイルサイズ異常'])
            recommendations.append(f"📁 ファイルサイズに問題のある画像が{count}枚あります。適切な品質設定で保存し直してください。")
        
        if issues['読み込みエラー']:
            count = len(issues['読み込みエラー'])
            recommendations.append(f"❌ 読み込みエラーの画像が{count}枚あります。ファイルが破損している可能性があります。")
        
        # 全体的な提案
        if total_issues > len(issues.get('アスペクト比異常', [])):  # アスペクト比以外の問題がある場合
            recommendations.append("🔧 品質改善後、再度チェックを実行して確認することをお勧めします。")
        
        return recommendations
    
    def get_quality_summary(self, quality_result: Dict) -> Dict:
        """品質チェック結果のサマリー生成"""
        
        if quality_result.get('エラー', False):
            return {'エラー': True, 'メッセージ': quality_result.get('メッセージ', '不明なエラー')}
        
        total_images = quality_result['総画像数']
        issues = quality_result['品質問題']
        
        # 問題の総数計算
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        problem_images = set()
        
        for issue_list in issues.values():
            for issue in issue_list:
                if isinstance(issue, dict) and 'ファイル名' in issue:
                    problem_images.add(issue['ファイル名'])
        
        problem_count = len(problem_images)
        healthy_count = total_images - problem_count
        
        # 品質スコア計算（0-100）
        quality_score = max(0, int((healthy_count / total_images) * 100)) if total_images > 0 else 0
        
        return {
            'エラー': False,
            '品質スコア': quality_score,
            '総画像数': total_images,
            '健全画像数': healthy_count,
            '問題画像数': problem_count,
            '主要問題': self._get_major_issues(issues),
            '状態': self._get_quality_status(quality_score),
            '推奨アクション': quality_result['推奨改善'][:3]  # 上位3つ
        }
    
    def _get_major_issues(self, issues: Dict) -> List[Dict]:
        """主要な問題を抽出"""
        
        major_issues = []
        
        for issue_type, issue_list in issues.items():
            if issue_list:
                # 重要度の高い問題を優先
                high_priority = [issue for issue in issue_list if issue.get('重要度') == 'high']
                count = len(issue_list)
                priority_count = len(high_priority)
                
                major_issues.append({
                    '問題種別': issue_type,
                    '件数': count,
                    '高重要度': priority_count,
                    '重要度': 'high' if priority_count > 0 else 'medium' if count > 0 else 'low'
                })
        
        # 重要度と件数でソート
        major_issues.sort(key=lambda x: (x['重要度'] == 'high', x['件数']), reverse=True)
        
        return major_issues[:5]  # 上位5つ
    
    def _get_quality_status(self, score: int) -> str:
        """品質スコアから状態を判定"""
        
        if score >= 90:
            return "🟢 優秀"
        elif score >= 75:
            return "🟡 良好"
        elif score >= 60:
            return "🟠 改善推奨"
        else:
            return "🔴 要改善"
    
    def export_quality_report(self, quality_result: Dict, output_path: str):
        """品質チェックレポートをJSON形式でエクスポート"""
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(quality_result, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"レポート出力エラー: {e}")
            return False