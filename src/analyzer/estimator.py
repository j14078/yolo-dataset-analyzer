"""
YOLOv9データセット分析・推定エンジン
初心者向けシンプル設計
"""

import os
import json
import glob
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class BeginnerFriendlyYOLOEstimator:
    """初心者向けYOLOv9軽量モデル用ラベル推定"""
    
    # 軽量モデル用ベースライン（経験則）
    LIGHTWEIGHT_BASE_SAMPLES = {
        'simple_objects': {      # 車、人、動物など
            '正解率60%目標': 70,
            '正解率70%目標': 120,
            '正解率80%目標': 200,
        },
        'medium_objects': {      # 家具、看板、道具など
            '正解率60%目標': 150,
            '正解率70%目標': 250,
            '正解率80%目標': 400,
        },
        'complex_objects': {     # 小物、部品、文字など
            '正解率60%目標': 300,
            '正解率70%目標': 500,
            '正解率80%目標': 800,
        }
    }
    
    # 画像サイズ補正
    IMAGE_SIZE_FACTOR = {
        320: {'倍率': 0.8, '説明': '高速だが精度控えめ'},
        640: {'倍率': 1.0, '説明': '標準的なバランス'},
        1280: {'倍率': 1.4, '説明': '高精度だが時間かかる'},
    }

    def analyze_mixed_folder(self, folder_path: str) -> Dict:
        """
        画像とJSONが混在したフォルダを自動分析
        
        Args:
            folder_path: 分析対象フォルダパス
            
        Returns:
            分析結果辞書
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"フォルダが見つかりません: {folder_path}")
        
        # ファイル自動識別
        all_files = os.listdir(folder_path)
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
        json_files = [f for f in all_files if f.lower().endswith('.json')]
        
        # ペア確認（同名の画像とJSONがあるか）
        labeled_pairs = []
        unlabeled_images = []
        
        for img_file in image_files:
            img_name = os.path.splitext(img_file)[0]
            json_name = img_name + '.json'
            
            if json_name in json_files:
                labeled_pairs.append((img_file, json_name))
            else:
                unlabeled_images.append(img_file)
        
        label_rate = (len(labeled_pairs) / len(image_files) * 100) if image_files else 0
        
        return {
            'ラベル済み画像数': len(labeled_pairs),
            '未ラベル画像数': len(unlabeled_images),
            '全画像数': len(image_files),
            'ラベル率': round(label_rate, 1),
            'labeled_pairs': labeled_pairs,
            'unlabeled_images': unlabeled_images,
            'folder_path': folder_path
        }

    def analyze_classes_from_json(self, folder_path: str, labeled_pairs: List[Tuple]) -> Dict[str, int]:
        """
        labelme JSONファイルからクラス情報を抽出
        
        Args:
            folder_path: フォルダパス
            labeled_pairs: (画像ファイル, JSONファイル)のペアリスト
            
        Returns:
            クラス名: 出現回数の辞書
        """
        class_counts = defaultdict(int)
        
        for img_file, json_file in labeled_pairs:
            json_path = os.path.join(folder_path, json_file)
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # labelmeのshapes配列からラベルを抽出
                for shape in data.get('shapes', []):
                    if shape.get('shape_type') == 'rectangle':
                        label = shape.get('label', '').strip()
                        if label:
                            class_counts[label] += 1
                            
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                print(f"警告: {json_file} の読み込みに失敗: {e}")
                continue
        
        return dict(class_counts)

    def judge_complexity_simple(self, class_name: str) -> str:
        """
        クラス名から複雑度を簡易判定
        
        Args:
            class_name: クラス名
            
        Returns:
            'simple_objects' | 'medium_objects' | 'complex_objects'
        """
        simple_keywords = ['人', 'car', 'truck', 'dog', 'cat', 'bird', '車', '犬', '猫']
        complex_keywords = ['screw', 'component', 'part', 'text', 'label', 'ネジ', '部品', '文字']
        
        class_lower = class_name.lower()
        
        # シンプルなオブジェクト
        if any(keyword in class_lower for keyword in simple_keywords):
            return 'simple_objects'
        
        # 複雑なオブジェクト
        if any(keyword in class_lower for keyword in complex_keywords):
            return 'complex_objects'
        
        # デフォルトは中程度
        return 'medium_objects'

    def get_simple_recommendation(self, 
                                folder_path: str,
                                target_accuracy: str = '正解率70%目標',
                                image_size: int = 640) -> Dict:
        """
        初心者向けシンプル推奨値
        
        Args:
            folder_path: 画像・JSONフォルダ
            target_accuracy: '正解率60%目標' | '正解率70%目標' | '正解率80%目標'
            image_size: 320 | 640 | 1280
            
        Returns:
            分かりやすい推奨結果
        """
        # フォルダ分析
        folder_stats = self.analyze_mixed_folder(folder_path)
        
        if folder_stats['ラベル済み画像数'] == 0:
            return {
                'エラー': True,
                'メッセージ': 'まだラベル付けされた画像がありません',
                'アクション': 'labelmeでラベル付けを始めてください',
                '推奨開始数': '最初は各クラス10-20枚から始めましょう'
            }
        
        # クラス分析
        class_stats = self.analyze_classes_from_json(
            folder_stats['folder_path'], 
            folder_stats['labeled_pairs']
        )
        
        if not class_stats:
            return {
                'エラー': True,
                'メッセージ': 'ラベル情報が見つかりません',
                'アクション': 'JSONファイルの形式を確認してください'
            }
        
        recommendations = {}
        
        for class_name, current_count in class_stats.items():
            # オブジェクトの複雑さを自動判定
            complexity = self.judge_complexity_simple(class_name)
            
            # 基本必要数
            base_needed = self.LIGHTWEIGHT_BASE_SAMPLES[complexity][target_accuracy]
            
            # 画像サイズ補正
            size_factor = self.IMAGE_SIZE_FACTOR[image_size]['倍率']
            needed = int(base_needed * size_factor)
            
            shortage = max(0, needed - current_count)
            
            recommendations[class_name] = {
                '現在の数': current_count,
                '推奨数': needed,
                '不足数': shortage,
                '進捗率': f"{min(100, current_count/needed*100):.0f}%",
                '状態': self._get_status_message(current_count, needed),
                '次にやること': self._get_next_action(shortage, current_count),
                '複雑度': complexity
            }
        
        # 全体サマリー
        total_current = sum(class_stats.values())
        total_needed = sum(rec['推奨数'] for rec in recommendations.values())
        
        return {
            'エラー': False,
            '全体サマリー': {
                '現在の総ラベル数': total_current,
                '推奨総ラベル数': total_needed,
                '全体進捗': f"{min(100, total_current/total_needed*100):.0f}%",
                '画像サイズ設定': f"{image_size}x{image_size} ({self.IMAGE_SIZE_FACTOR[image_size]['説明']})",
                '目標精度': target_accuracy,
                'ラベル率': f"{folder_stats['ラベル率']:.1f}%"
            },
            'クラス別詳細': recommendations,
            '次のステップ': self._get_overall_next_step(recommendations),
            'フォルダ情報': folder_stats
        }

    def _get_status_message(self, current: int, needed: int) -> str:
        """状態を分かりやすく表現"""
        ratio = current / needed
        
        if ratio >= 1.0:
            return "✅ 充分です！学習を始められます"
        elif ratio >= 0.7:
            return "🟡 もう少しです。あと少し頑張りましょう"
        elif ratio >= 0.3:
            return "🟠 まだ不足しています。継続してラベル付けしましょう"
        else:
            return "🔴 大幅に不足しています。コツコツ続けましょう"

    def _get_next_action(self, shortage: int, current: int) -> str:
        """具体的な次のアクション"""
        if shortage == 0:
            return "学習を開始してください"
        elif shortage <= 10:
            return f"あと{shortage}枚ラベル付けしてください"
        elif shortage <= 50:
            return f"今週中に{shortage}枚を目標にしましょう"
        else:
            days_needed = max(7, shortage // 7)
            return f"1日5-10枚ずつ、約{days_needed}日間で完了予定"

    def _get_overall_next_step(self, recommendations: Dict) -> List[str]:
        """全体の次にやるべきこと"""
        steps = []
        
        # 最も不足しているクラスを特定
        max_shortage_class = max(
            recommendations.items(), 
            key=lambda x: x[1]['不足数']
        )
        
        if max_shortage_class[1]['不足数'] > 0:
            steps.append(f"1. 「{max_shortage_class[0]}」のラベル付けを優先しましょう")
            steps.append("2. labelmeで画像を開き、矩形でオブジェクトを囲んでください")
            steps.append("3. 正確なラベル名を入力してください")
            steps.append("4. 1日10枚を目標に継続しましょう")
        else:
            steps.append("1. 学習用のスクリプト準備をしましょう")
            steps.append("2. データを train/val に分割してください")
            steps.append("3. YOLO形式に変換してください")
        
        return steps