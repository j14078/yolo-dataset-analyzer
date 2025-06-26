#!/usr/bin/env python3
"""
YOLO データセット分析ツール 起動スクリプト
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """メイン起動関数"""
    print("🎯 YOLO データセット分析ツール")
    print("=" * 50)
    
    # プロジェクトルートを取得
    project_root = Path(__file__).parent
    ui_path = project_root / "src" / "ui" / "streamlit_app.py"
    
    # パス確認
    if not ui_path.exists():
        print(f"❌ エラー: UIファイルが見つかりません: {ui_path}")
        return
    
    # 依存関係チェック
    print("📦 依存関係をチェック中...")
    
    try:
        import streamlit
        import pandas
        import plotly
        import numpy
        import PIL
        print("✅ 必要なライブラリがインストールされています")
    except ImportError as e:
        print(f"❌ 必要なライブラリが不足しています: {e}")
        print("💡 以下のコマンドでインストールしてください:")
        print(f"   pip install -r {project_root / 'requirements.txt'}")
        return
    
    # Streamlit起動
    print("🚀 Streamlitを起動中...")
    print("📱 ブラウザが自動で開きます")
    print("🔄 終了するには Ctrl+C を押してください")
    print("-" * 50)
    
    try:
        # Streamlitコマンド実行
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(ui_path),
            "--server.port", "8501",
            "--browser.serverAddress", "localhost",
            "--theme.base", "light"
        ]
        
        subprocess.run(cmd, cwd=str(project_root))
        
    except KeyboardInterrupt:
        print("\n👋 ツールを終了しました")
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")


if __name__ == "__main__":
    main()