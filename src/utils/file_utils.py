"""
ファイル操作ユーティリティ
"""

import os
from pathlib import Path
from typing import List, Tuple


def get_image_json_pairs(folder_path: str) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    フォルダから画像ファイルとJSONファイルのペアを取得
    
    Args:
        folder_path: 対象フォルダパス
        
    Returns:
        (画像ファイルリスト, JSONファイルリスト, ペアリスト)
    """
    if not os.path.exists(folder_path):
        return [], [], []
    
    all_files = os.listdir(folder_path)
    
    # 対応する拡張子
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    # ファイル分類
    image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
    json_files = [f for f in all_files if f.lower().endswith('.json')]
    
    # ペア作成
    pairs = []
    for img_file in image_files:
        img_name = os.path.splitext(img_file)[0]
        json_name = img_name + '.json'
        
        if json_name in json_files:
            pairs.append((img_file, json_name))
    
    return image_files, json_files, pairs


def validate_folder_structure(folder_path: str) -> dict:
    """
    フォルダ構造の妥当性チェック
    
    Args:
        folder_path: チェック対象フォルダ
        
    Returns:
        検証結果辞書
    """
    result = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    if not os.path.exists(folder_path):
        result['errors'].append(f"フォルダが存在しません: {folder_path}")
        return result
    
    if not os.path.isdir(folder_path):
        result['errors'].append(f"指定されたパスはフォルダではありません: {folder_path}")
        return result
    
    try:
        image_files, json_files, pairs = get_image_json_pairs(folder_path)
        
        result['info'] = {
            '画像ファイル数': len(image_files),
            'JSONファイル数': len(json_files),
            'ペア数': len(pairs)
        }
        
        # エラーチェック
        if len(image_files) == 0:
            result['errors'].append("画像ファイルが見つかりません")
        
        if len(json_files) == 0:
            result['errors'].append("JSONファイルが見つかりません")
        
        # 警告チェック
        if len(pairs) < len(image_files):
            unpaired = len(image_files) - len(pairs)
            result['warnings'].append(f"{unpaired}枚の画像にラベルが付いていません")
        
        if len(json_files) > len(image_files):
            extra_json = len(json_files) - len(image_files)
            result['warnings'].append(f"{extra_json}個の余分なJSONファイルがあります")
        
        # 成功判定
        if not result['errors']:
            result['valid'] = True
            
    except Exception as e:
        result['errors'].append(f"フォルダの読み込み中にエラーが発生: {str(e)}")
    
    return result


def get_folder_summary(folder_path: str) -> str:
    """
    フォルダの概要を文字列で取得
    
    Args:
        folder_path: 対象フォルダ
        
    Returns:
        概要文字列
    """
    validation = validate_folder_structure(folder_path)
    
    if not validation['valid']:
        return f"❌ エラー: {', '.join(validation['errors'])}"
    
    info = validation['info']
    summary_parts = [
        f"📁 {os.path.basename(folder_path)}",
        f"🖼️ 画像 {info['画像ファイル数']}枚",
        f"📋 ラベル済み {info['ペア数']}枚",
    ]
    
    if validation['warnings']:
        summary_parts.append(f"⚠️ {len(validation['warnings'])}件の注意事項")
    
    return " | ".join(summary_parts)