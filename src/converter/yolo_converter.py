"""
labelme → YOLO形式変換エンジン
初心者向けシンプル設計
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import yaml


class YOLOConverter:
    """labelme形式からYOLO形式への変換"""
    
    def __init__(self):
        self.class_mapping = {}  # クラス名 → インデックス
        self.conversion_stats = {
            'total_images': 0,
            'converted_images': 0,
            'skipped_images': 0,
            'total_annotations': 0,
            'class_counts': defaultdict(int),
            'errors': []
        }
    
    def convert_dataset(self, 
                       input_folder: str, 
                       output_folder: str,
                       train_ratio: float = 0.8,
                       copy_images: bool = True,
                       target_size: Optional[Tuple[int, int]] = None) -> Dict:
        """
        データセット全体を変換
        
        Args:
            input_folder: labelme形式のデータフォルダ
            output_folder: YOLO形式の出力フォルダ
            train_ratio: 訓練データの割合 (0.0-1.0)
            copy_images: 画像ファイルもコピーするか
            target_size: リサイズ目標サイズ (width, height)
            
        Returns:
            変換結果の詳細
        """
        
        # 初期化
        self._reset_stats()
        
        # 入力フォルダの検証
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"入力フォルダが見つかりません: {input_folder}")
        
        # 出力フォルダの準備
        self._prepare_output_folders(output_folder)
        
        # ファイル一覧取得
        image_json_pairs = self._get_image_json_pairs(input_folder)
        
        if not image_json_pairs:
            raise ValueError("有効な画像とJSONのペアが見つかりません")
        
        # クラス情報の収集
        self._collect_class_info(input_folder, image_json_pairs)
        
        # train/val分割
        train_pairs, val_pairs = self._split_train_val(image_json_pairs, train_ratio)
        
        # 変換実行
        train_stats = self._convert_split(
            input_folder, output_folder, train_pairs, 'train', 
            copy_images, target_size
        )
        
        val_stats = self._convert_split(
            input_folder, output_folder, val_pairs, 'val', 
            copy_images, target_size
        )
        
        # 設定ファイル生成
        self._generate_config_files(output_folder)
        
        # 結果サマリー
        return self._generate_summary(output_folder, train_stats, val_stats)
    
    def _reset_stats(self):
        """統計情報をリセット"""
        self.conversion_stats = {
            'total_images': 0,
            'converted_images': 0,
            'skipped_images': 0,
            'total_annotations': 0,
            'class_counts': defaultdict(int),
            'errors': []
        }
        self.class_mapping = {}
    
    def _prepare_output_folders(self, output_folder: str):
        """出力フォルダ構造を準備"""
        folders = [
            'images/train',
            'images/val', 
            'labels/train',
            'labels/val'
        ]
        
        for folder in folders:
            folder_path = os.path.join(output_folder, folder)
            os.makedirs(folder_path, exist_ok=True)
    
    def _get_image_json_pairs(self, folder_path: str) -> List[Tuple[str, str]]:
        """画像とJSONファイルのペア一覧を取得"""
        pairs = []
        
        # 対応する拡張子
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        
        all_files = os.listdir(folder_path)
        image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
        
        for img_file in image_files:
            img_name = os.path.splitext(img_file)[0]
            json_file = img_name + '.json'
            
            if json_file in all_files:
                pairs.append((img_file, json_file))
        
        return pairs
    
    def _collect_class_info(self, input_folder: str, pairs: List[Tuple[str, str]]):
        """全JSONファイルからクラス情報を収集"""
        all_classes = set()
        
        for img_file, json_file in pairs:
            json_path = os.path.join(input_folder, json_file)
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for shape in data.get('shapes', []):
                    if shape.get('shape_type') == 'rectangle':
                        label = shape.get('label', '').strip()
                        if label:
                            all_classes.add(label)
                            
            except Exception as e:
                self.conversion_stats['errors'].append(f"クラス情報収集エラー {json_file}: {e}")
        
        # クラス名をソートしてインデックス割り当て
        sorted_classes = sorted(list(all_classes))
        self.class_mapping = {cls: idx for idx, cls in enumerate(sorted_classes)}
    
    def _split_train_val(self, pairs: List[Tuple[str, str]], train_ratio: float) -> Tuple[List, List]:
        """データをtrain/valに分割"""
        # シャッフル
        shuffled_pairs = pairs.copy()
        random.shuffle(shuffled_pairs)
        
        # 分割
        split_idx = int(len(shuffled_pairs) * train_ratio)
        train_pairs = shuffled_pairs[:split_idx]
        val_pairs = shuffled_pairs[split_idx:]
        
        return train_pairs, val_pairs
    
    def _convert_split(self, 
                      input_folder: str, 
                      output_folder: str, 
                      pairs: List[Tuple[str, str]], 
                      split_name: str,
                      copy_images: bool,
                      target_size: Optional[Tuple[int, int]]) -> Dict:
        """分割データを変換"""
        
        split_stats = {
            'converted': 0,
            'skipped': 0,
            'annotations': 0,
            'errors': []
        }
        
        for img_file, json_file in pairs:
            try:
                result = self._convert_single_pair(
                    input_folder, output_folder, img_file, json_file, 
                    split_name, copy_images, target_size
                )
                
                if result['success']:
                    split_stats['converted'] += 1
                    split_stats['annotations'] += result['annotation_count']
                else:
                    split_stats['skipped'] += 1
                    split_stats['errors'].append(result['error'])
                    
            except Exception as e:
                split_stats['skipped'] += 1
                split_stats['errors'].append(f"{img_file}: {e}")
        
        return split_stats
    
    def _convert_single_pair(self, 
                           input_folder: str, 
                           output_folder: str,
                           img_file: str, 
                           json_file: str, 
                           split_name: str,
                           copy_images: bool,
                           target_size: Optional[Tuple[int, int]]) -> Dict:
        """単一の画像+JSONペアを変換"""
        
        # ファイルパス
        img_path = os.path.join(input_folder, img_file)
        json_path = os.path.join(input_folder, json_file)
        
        # JSON読み込み
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 画像サイズ取得
        img_width = data['imageWidth']
        img_height = data['imageHeight']
        
        # アノテーション変換
        yolo_annotations = []
        annotation_count = 0
        
        for shape in data.get('shapes', []):
            if shape.get('shape_type') == 'rectangle':
                label = shape.get('label', '').strip()
                
                if label in self.class_mapping:
                    # 座標変換
                    points = shape['points']
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    
                    # YOLO形式に変換（中心点 + 幅・高さ、正規化）
                    center_x = (x1 + x2) / 2 / img_width
                    center_y = (y1 + y2) / 2 / img_height
                    width = abs(x2 - x1) / img_width
                    height = abs(y2 - y1) / img_height
                    
                    class_id = self.class_mapping[label]
                    yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
                    annotation_count += 1
                    
                    # 統計更新
                    self.conversion_stats['class_counts'][label] += 1
        
        # ファイル名準備
        base_name = os.path.splitext(img_file)[0]
        
        # 画像コピー
        if copy_images:
            dst_img_path = os.path.join(output_folder, 'images', split_name, img_file)
            shutil.copy2(img_path, dst_img_path)
        
        # ラベルファイル作成
        label_file = base_name + '.txt'
        dst_label_path = os.path.join(output_folder, 'labels', split_name, label_file)
        
        with open(dst_label_path, 'w', encoding='utf-8') as f:
            for annotation in yolo_annotations:
                f.write(annotation + '\n')
        
        return {
            'success': True,
            'annotation_count': annotation_count,
            'error': None
        }
    
    def _generate_config_files(self, output_folder: str):
        """YOLO設定ファイル生成"""
        
        # classes.names ファイル
        names_path = os.path.join(output_folder, 'classes.names')
        with open(names_path, 'w', encoding='utf-8') as f:
            for class_name in sorted(self.class_mapping.keys()):
                f.write(class_name + '\n')
        
        # dataset.yaml ファイル
        yaml_data = {
            'path': os.path.abspath(output_folder),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_mapping),
            'names': sorted(self.class_mapping.keys())
        }
        
        yaml_path = os.path.join(output_folder, 'dataset.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        
        # README.txt ファイル
        readme_path = os.path.join(output_folder, 'README.txt')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("YOLO形式データセット\n")
            f.write("=" * 30 + "\n\n")
            f.write("フォルダ構成:\n")
            f.write("├── images/\n")
            f.write("│   ├── train/     # 訓練用画像\n")
            f.write("│   └── val/       # 検証用画像\n")
            f.write("├── labels/\n")
            f.write("│   ├── train/     # 訓練用ラベル\n")
            f.write("│   └── val/       # 検証用ラベル\n")
            f.write("├── dataset.yaml   # YOLOv5/v8/v9用設定\n")
            f.write("├── classes.names  # クラス名一覧\n")
            f.write("└── README.txt     # このファイル\n\n")
            f.write("使用方法:\n")
            f.write("YOLOv9の学習時に dataset.yaml を指定してください\n")
            f.write("例: python train.py --data dataset.yaml\n")
    
    def _generate_summary(self, output_folder: str, train_stats: Dict, val_stats: Dict) -> Dict:
        """変換結果のサマリーを生成"""
        
        total_converted = train_stats['converted'] + val_stats['converted']
        total_skipped = train_stats['skipped'] + val_stats['skipped']
        total_annotations = train_stats['annotations'] + val_stats['annotations']
        
        return {
            '変換成功': True,
            '出力フォルダ': output_folder,
            '統計情報': {
                '変換成功画像数': total_converted,
                'スキップ画像数': total_skipped,
                '総アノテーション数': total_annotations,
                '訓練データ数': train_stats['converted'],
                '検証データ数': val_stats['converted'],
                'クラス数': len(self.class_mapping)
            },
            'クラス情報': dict(self.conversion_stats['class_counts']),
            'クラスマッピング': self.class_mapping,
            '出力ファイル': {
                'dataset.yaml': os.path.join(output_folder, 'dataset.yaml'),
                'classes.names': os.path.join(output_folder, 'classes.names'),
                'README.txt': os.path.join(output_folder, 'README.txt')
            },
            'エラー': train_stats['errors'] + val_stats['errors']
        }
    
    def validate_yolo_dataset(self, yolo_folder: str) -> Dict:
        """YOLO形式データセットの検証"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # 必須ファイル・フォルダの確認
        required_items = [
            'images/train',
            'images/val',
            'labels/train', 
            'labels/val',
            'dataset.yaml'
        ]
        
        for item in required_items:
            path = os.path.join(yolo_folder, item)
            if not os.path.exists(path):
                validation_result['errors'].append(f"必須項目が見つかりません: {item}")
                validation_result['valid'] = False
        
        if validation_result['valid']:
            # 統計情報収集
            train_images = len(os.listdir(os.path.join(yolo_folder, 'images/train')))
            val_images = len(os.listdir(os.path.join(yolo_folder, 'images/val')))
            train_labels = len(os.listdir(os.path.join(yolo_folder, 'labels/train')))
            val_labels = len(os.listdir(os.path.join(yolo_folder, 'labels/val')))
            
            validation_result['statistics'] = {
                '訓練画像数': train_images,
                '検証画像数': val_images,
                '訓練ラベル数': train_labels,
                '検証ラベル数': val_labels
            }
            
            # 警告チェック
            if train_images != train_labels:
                validation_result['warnings'].append(f"訓練データの画像数とラベル数が不一致: {train_images} vs {train_labels}")
            
            if val_images != val_labels:
                validation_result['warnings'].append(f"検証データの画像数とラベル数が不一致: {val_images} vs {val_labels}")
        
        return validation_result