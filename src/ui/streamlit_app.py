"""
YOLOv9 データセット分析ツール
Streamlit UI メイン画面
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# パスを追加してモジュールをインポート
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from analyzer.estimator import BeginnerFriendlyYOLOEstimator
from utils.file_utils import validate_folder_structure, get_folder_summary
from converter.yolo_converter import YOLOConverter


def main():
    """メイン関数"""
    st.set_page_config(
        page_title="YOLO データセット分析ツール",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # ヘッダー
    st.title("🎯 YOLO データセット分析ツール")
    st.markdown("**初心者向け** - フォルダを選ぶだけで簡単分析＆変換！")
    
    # タブで機能切り替え
    tab1, tab2 = st.tabs(["📊 分析", "🔄 YOLO変換"])
    
    with tab1:
        show_analysis_tab()
    
    with tab2:
        show_conversion_tab()


def show_analysis_tab():
    """分析タブの内容"""
    
    # サイドバー: 設定
    with st.sidebar:
        st.header("📊 分析設定")
        
        # フォルダ選択
        folder_path = get_folder_input()
        
        if folder_path and os.path.exists(folder_path):
            # フォルダ概要表示
            st.success(get_folder_summary(folder_path))
            
            # 設定項目
            target_accuracy = st.selectbox(
                "🎯 目標精度",
                ["正解率60%目標", "正解率70%目標", "正解率80%目標"],
                index=1,
                help="現実的な目標を選んでください。初心者は70%がおすすめです。"
            )
            
            image_size = st.selectbox(
                "📏 画像サイズ",
                [320, 640, 1280],
                index=1,
                format_func=lambda x: f"{x}x{x} ({'高速' if x==320 else '標準' if x==640 else '高精度'})",
                help="640x640が標準的でバランスが良いです。"
            )
            
            # 分析実行ボタン
            analyze_button = st.button("🔍 分析開始", type="primary", use_container_width=True)
        else:
            analyze_button = False
    
    # メインエリア
    if analyze_button and folder_path:
        analyze_and_display(folder_path, target_accuracy, image_size)
    else:
        show_welcome_page()


def show_conversion_tab():
    """YOLO変換タブの内容"""
    
    # サイドバー: 変換設定
    with st.sidebar:
        st.header("🔄 変換設定")
        
        # 入力フォルダ選択
        input_folder = st.text_input(
            "📁 入力フォルダ（labelme形式）",
            placeholder="C:\\path\\to\\labelme\\dataset",
            help="画像とJSONファイルが入っているフォルダ"
        )
        
        # 出力フォルダ選択
        output_folder = st.text_input(
            "📂 出力フォルダ（YOLO形式）",
            placeholder="C:\\path\\to\\yolo\\dataset",
            help="YOLO形式データセットの保存先"
        )
        
        # 変換設定
        st.subheader("⚙️ 変換オプション")
        
        train_ratio = st.slider(
            "📈 訓練データ割合",
            min_value=0.5,
            max_value=0.95,
            value=0.8,
            step=0.05,
            help="残りが検証データになります"
        )
        
        copy_images = st.checkbox(
            "🖼️ 画像ファイルもコピー",
            value=True,
            help="チェックを外すとラベルファイルのみ生成"
        )
        
        # 変換実行ボタン
        convert_button = st.button(
            "🚀 YOLO形式に変換", 
            type="primary", 
            use_container_width=True,
            disabled=not (input_folder and output_folder)
        )
    
    # メインエリア
    if convert_button and input_folder and output_folder:
        perform_conversion(input_folder, output_folder, train_ratio, copy_images)
    else:
        show_conversion_help()


def show_conversion_help():
    """変換機能のヘルプページ"""
    st.markdown("""
    ## 🔄 YOLO形式変換機能
    
    labelme形式のアノテーションデータをYOLOv9で使用できる形式に変換します。
    
    ### 📋 変換前の準備
    
    **入力データ（labelme形式）:**
    ```
    input_folder/
    ├── image1.jpg
    ├── image1.json
    ├── image2.jpg
    ├── image2.json
    └── ...
    ```
    
    ### 📦 変換後の出力
    
    **出力データ（YOLO形式）:**
    ```
    output_folder/
    ├── images/
    │   ├── train/          # 訓練用画像
    │   └── val/            # 検証用画像
    ├── labels/
    │   ├── train/          # 訓練用ラベル(.txt)
    │   └── val/            # 検証用ラベル(.txt)
    ├── dataset.yaml        # YOLOv9設定ファイル
    ├── classes.names       # クラス名一覧
    └── README.txt          # 使用方法
    ```
    
    ### ✨ 主な機能
    
    - 📊 **自動train/val分割**: 指定した割合で自動分割
    - 🎯 **座標変換**: labelme → YOLO形式の座標変換
    - 📝 **設定ファイル生成**: dataset.yaml等を自動生成
    - ✅ **検証機能**: 変換結果の妥当性チェック
    
    ### 🚀 YOLOv9での使用方法
    
    変換後、YOLOv9で以下のように学習できます：
    
    ```bash
    python train.py --data /path/to/output_folder/dataset.yaml
    ```
    """)
    
    st.info("💡 左側のサイドバーで入力・出力フォルダを指定して「変換」ボタンを押してください")


def perform_conversion(input_folder: str, output_folder: str, train_ratio: float, copy_images: bool):
    """YOLO変換を実行"""
    
    # プログレスバー
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 入力フォルダを検証中...")
        progress_bar.progress(10)
        
        # 入力フォルダ検証
        validation = validate_folder_structure(input_folder)
        if not validation['valid']:
            st.error("❌ 入力フォルダに問題があります:")
            for error in validation['errors']:
                st.error(f"• {error}")
            return
        
        status_text.text("🔄 変換処理を開始中...")
        progress_bar.progress(25)
        
        # 変換実行
        converter = YOLOConverter()
        
        status_text.text("📁 ファイルを変換中...")
        progress_bar.progress(50)
        
        result = converter.convert_dataset(
            input_folder=input_folder,
            output_folder=output_folder,
            train_ratio=train_ratio,
            copy_images=copy_images
        )
        
        status_text.text("📊 結果を表示中...")
        progress_bar.progress(75)
        
        # 結果表示
        display_conversion_results(result)
        
        progress_bar.progress(100)
        status_text.text("✅ 変換完了！")
        
        # 少し待ってからクリア
        import time
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ 変換中にエラーが発生しました: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def display_conversion_results(result: dict):
    """変換結果の表示"""
    
    if result['変換成功']:
        st.success("🎉 YOLO形式への変換が完了しました！")
        
        # 統計情報
        st.header("📊 変換統計")
        
        stats = result['統計情報']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("変換成功", stats['変換成功画像数'])
        with col2:
            st.metric("訓練データ", stats['訓練データ数'])
        with col3:
            st.metric("検証データ", stats['検証データ数'])
        with col4:
            st.metric("クラス数", stats['クラス数'])
        
        # クラス分布
        st.subheader("🎯 クラス分布")
        class_df = pd.DataFrame(
            list(result['クラス情報'].items()),
            columns=['クラス名', 'アノテーション数']
        )
        
        # 棒グラフ
        fig = px.bar(
            class_df,
            x='クラス名',
            y='アノテーション数',
            title="クラス別アノテーション数"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # データフレーム表示
        st.dataframe(class_df, use_container_width=True, hide_index=True)
        
        # 出力ファイル情報
        st.header("📁 生成ファイル")
        
        output_files = result['出力ファイル']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"📂 **出力フォルダ**: {result['出力フォルダ']}")
            st.info(f"📝 **設定ファイル**: dataset.yaml")
            st.info(f"📋 **クラス一覧**: classes.names")
        
        with col2:
            st.info(f"🖼️ **画像フォルダ**: images/train, images/val")
            st.info(f"🏷️ **ラベルフォルダ**: labels/train, labels/val") 
            st.info(f"📄 **説明書**: README.txt")
        
        # YOLOv9学習コマンド
        st.header("🚀 次のステップ")
        
        dataset_yaml_path = output_files['dataset.yaml']
        
        st.subheader("YOLOv9での学習コマンド")
        st.code(f"python train.py --data {dataset_yaml_path}", language="bash")
        
        # エラー・警告
        if result['エラー']:
            st.warning("⚠️ 変換中に以下の問題が発生しました:")
            for error in result['エラー']:
                st.warning(f"• {error}")
        
        # 検証実行
        st.subheader("✅ データセット検証")
        if st.button("🔍 変換結果を検証"):
            converter = YOLOConverter()
            validation = converter.validate_yolo_dataset(result['出力フォルダ'])
            
            if validation['valid']:
                st.success("✅ データセットは正常です")
                
                # 統計表示
                val_stats = validation['statistics']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("訓練画像", val_stats['訓練画像数'])
                with col2:
                    st.metric("検証画像", val_stats['検証画像数'])
                with col3:
                    st.metric("訓練ラベル", val_stats['訓練ラベル数'])
                with col4:
                    st.metric("検証ラベル", val_stats['検証ラベル数'])
                
            else:
                st.error("❌ データセットに問題があります:")
                for error in validation['errors']:
                    st.error(f"• {error}")
            
            # 警告表示
            for warning in validation['warnings']:
                st.warning(f"⚠️ {warning}")
    
    else:
        st.error("❌ 変換に失敗しました")
        if 'エラー' in result:
            for error in result['エラー']:
                st.error(f"• {error}")


def get_folder_input():
    """フォルダ入力UI"""
    
    # タブで入力方法を切り替え
    tab1, tab2 = st.tabs(["📁 パス入力", "💡 ヘルプ"])
    
    with tab1:
        # よく使うパステンプレート
        st.subheader("フォルダパス入力")
        
        # OS別テンプレート
        os_templates = {
            "Windows": "C:\\Users\\username\\Documents\\dataset",
            "Mac": "/Users/username/Documents/dataset",
            "Linux": "/home/username/dataset"
        }
        
        # テンプレート選択
        template_type = st.selectbox("OS種類", list(os_templates.keys()))
        
        # パス入力
        folder_path = st.text_input(
            "フォルダパス",
            value="",
            placeholder=os_templates[template_type],
            help="画像ファイル(.jpg, .png等)とlabelmeのJSONファイルが入っているフォルダのパスを入力してください"
        )
        
        # パス検証
        if folder_path:
            validation = validate_folder_structure(folder_path)
            
            if validation['valid']:
                st.success(f"✅ フォルダが見つかりました")
                return folder_path
            else:
                for error in validation['errors']:
                    st.error(f"❌ {error}")
                for warning in validation['warnings']:
                    st.warning(f"⚠️ {warning}")
        
        return folder_path if folder_path else None
    
    with tab2:
        st.subheader("💡 フォルダの準備方法")
        st.markdown("""
        ### 1. フォルダ構成
        ```
        your_dataset/
        ├── image1.jpg     # 画像ファイル
        ├── image1.json    # labelmeで作成したアノテーション
        ├── image2.jpg
        ├── image2.json
        └── ...
        ```
        
        ### 2. 対応ファイル形式
        - **画像**: .jpg, .jpeg, .png, .bmp, .tiff, .webp
        - **アノテーション**: .json (labelme形式)
        
        ### 3. 注意事項
        - 画像とJSONファイルは同じ名前にしてください
        - labelmeで矩形(rectangle)でラベル付けしてください
        - フォルダパスに日本語が含まれていても大丈夫です
        """)


def show_welcome_page():
    """ウェルカムページ"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## 📋 使い方
        
        1. **左側のサイドバーで設定をします**
        2. **画像フォルダのパスを入力してください**  
        3. **目標精度を選んでください**
        4. **「分析開始」ボタンを押してください**
        
        ## 💡 このツールでできること
        
        - 📊 **現在のラベル状況を確認**
        - 📈 **あと何枚ラベル付けが必要か分析**
        - 🎯 **クラス別の進捗確認**
        - 📋 **次にやるべきことを提案**
        
        ## 🔧 必要な準備
        
        - labelmeでラベル付けした画像とJSONファイル
        - 画像とJSONファイルが同じフォルダに入っている
        - 矩形(rectangle)でのラベル付け
        """)
        
        st.info("💡 初めての方は「ヘルプ」タブでフォルダの準備方法を確認してください")


def analyze_and_display(folder_path: str, target_accuracy: str, image_size: int):
    """分析実行・結果表示"""
    
    # プログレスバー
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 分析実行
        status_text.text("📁 フォルダを読み込み中...")
        progress_bar.progress(25)
        
        estimator = BeginnerFriendlyYOLOEstimator()
        
        status_text.text("🔍 ラベル状況を分析中...")
        progress_bar.progress(50)
        
        result = estimator.get_simple_recommendation(
            folder_path, target_accuracy, image_size
        )
        
        status_text.text("📊 結果を表示中...")
        progress_bar.progress(75)
        
        # エラーチェック
        if result.get('エラー', False):
            st.error(f"❌ {result['メッセージ']}")
            st.info(f"💡 {result['アクション']}")
            if '推奨開始数' in result:
                st.info(f"📝 {result['推奨開始数']}")
            return
        
        # 結果表示
        display_analysis_results(result)
        
        progress_bar.progress(100)
        status_text.text("✅ 分析完了！")
        
        # 少し待ってからクリア
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ エラーが発生しました: {str(e)}")
        st.info("💡 フォルダパスが正しいか、ファイルが存在するか確認してください")
        progress_bar.empty()
        status_text.empty()


def display_analysis_results(result: dict):
    """分析結果の表示"""
    
    # 全体サマリー
    st.header("📊 全体サマリー")
    
    summary = result['全体サマリー']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "現在のラベル数", 
            summary['現在の総ラベル数'],
            help="現在アノテーションされているオブジェクトの総数"
        )
    
    with col2:
        st.metric(
            "推奨ラベル数", 
            summary['推奨総ラベル数'],
            help="目標精度達成に必要な推奨ラベル数"
        )
    
    with col3:
        progress_val = summary['全体進捗']
        st.metric(
            "進捗", 
            progress_val,
            help="目標に対する現在の進捗率"
        )
    
    with col4:
        st.metric(
            "ラベル率", 
            summary['ラベル率'],
            help="全画像のうちラベル付けされた画像の割合"
        )
    
    # 進捗バー
    progress_value = float(progress_val.replace('%', '')) / 100
    st.progress(progress_value)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🎯 **目標精度**: {summary['目標精度']}")
    with col2:
        st.info(f"📏 **画像サイズ**: {summary['画像サイズ設定']}")
    
    # クラス別詳細
    st.header("📈 クラス別詳細")
    
    # データフレーム作成
    df_data = []
    for class_name, info in result['クラス別詳細'].items():
        df_data.append({
            'クラス名': class_name,
            '現在の数': info['現在の数'],
            '推奨数': info['推奨数'], 
            '不足数': info['不足数'],
            '進捗率': info['進捗率'],
            '状態': info['状態'],
            '次にやること': info['次にやること']
        })
    
    df = pd.DataFrame(df_data)
    
    # データフレーム表示
    st.dataframe(
        df[['クラス名', '現在の数', '推奨数', '不足数', '進捗率', '状態']], 
        use_container_width=True,
        hide_index=True
    )
    
    # グラフ表示
    st.subheader("📊 クラス別比較グラフ")
    
    # 棒グラフ
    fig = px.bar(
        df, 
        x='クラス名', 
        y=['現在の数', '推奨数'],
        title="クラス別ラベル数比較",
        barmode='group',
        color_discrete_map={
            '現在の数': '#1f77b4',
            '推奨数': '#ff7f0e'
        }
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # 進捗円グラフ
    progress_data = []
    for class_name, info in result['クラス別詳細'].items():
        progress_val = float(info['進捗率'].replace('%', ''))
        progress_data.append({
            'クラス名': class_name,
            '進捗率': progress_val,
            '完了': min(progress_val, 100),
            '残り': max(0, 100 - progress_val)
        })
    
    progress_df = pd.DataFrame(progress_data)
    
    fig_pie = px.pie(
        progress_df, 
        values='完了', 
        names='クラス名',
        title="クラス別進捗率"
    )
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # 詳細アクション
    st.header("🎯 クラス別の次のアクション")
    
    for class_name, info in result['クラス別詳細'].items():
        with st.expander(f"📋 {class_name} ({info['状態']})"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("不足数", info['不足数'])
                st.metric("進捗", info['進捗率'])
            
            with col2:
                st.write(f"**次にやること:** {info['次にやること']}")
                st.write(f"**状態:** {info['状態']}")
    
    # 全体の次のステップ
    st.header("🚀 次にやること")
    
    for i, step in enumerate(result['次のステップ'], 1):
        st.write(f"**{i}.** {step}")
    
    # レポートダウンロード機能
    st.header("📄 レポート出力")
    
    # CSV形式
    csv_data = df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📊 CSV形式でダウンロード",
        data=csv_data,
        file_name=f"yolo_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime='text/csv',
        help="Excelで開ける形式でデータをダウンロードします"
    )


if __name__ == "__main__":
    main()