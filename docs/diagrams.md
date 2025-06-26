# 📊 システム図解

このファイルには、YOLO Dataset Analyzerの各種図解を掲載しています。

## 🔄 全体ワークフロー

```mermaid
graph TD
    A[📁 labelme形式データ<br/>• image1.jpg<br/>• image1.json] --> B[📊 分析エンジン]
    A --> C[🔄 変換エンジン]
    
    B --> D[📈 分析結果<br/>• 推奨ラベル数<br/>• 進捗レポート<br/>• アクション提案]
    
    C --> E[🎯 YOLO形式データ<br/>• images/train/<br/>• labels/train/<br/>• dataset.yaml]
    
    E --> F[🚀 YOLOv9学習<br/>python train.py<br/>--data dataset.yaml]
    
    style A fill:#dae8fc,stroke:#6c8ebf
    style B fill:#fff2cc,stroke:#d6b656
    style C fill:#e1d5e7,stroke:#9673a6
    style D fill:#d5e8d4,stroke:#82b366
    style E fill:#f8cecc,stroke:#b85450
    style F fill:#ffe6cc,stroke:#d79b00
```

## 🏗️ システムアーキテクチャ

```mermaid
graph TB
    subgraph "🌐 Streamlit UI Layer"
        UI1[📊 分析タブ]
        UI2[🔄 変換タブ]
    end
    
    subgraph "🧮 Core Engine Layer"
        ENG1[📊 BeginnerFriendlyYOLOEstimator<br/>推定アルゴリズム]
        ENG2[🔄 YOLOConverter<br/>変換エンジン]
    end
    
    subgraph "🔧 Utility Layer"
        UTIL1[📁 file_utils<br/>ファイル操作]
        UTIL2[📋 validation<br/>データ検証]
    end
    
    subgraph "💾 Data Layer"
        DATA1[📁 labelme JSON]
        DATA2[🖼️ 画像ファイル]
        DATA3[🎯 YOLO txt]
        DATA4[⚙️ 設定ファイル]
    end
    
    UI1 --> ENG1
    UI2 --> ENG2
    ENG1 --> UTIL1
    ENG2 --> UTIL1
    ENG1 --> UTIL2
    ENG2 --> UTIL2
    
    UTIL1 --> DATA1
    UTIL1 --> DATA2
    ENG2 --> DATA3
    ENG2 --> DATA4
    
    style UI1 fill:#e1f5fe
    style UI2 fill:#f3e5f5
    style ENG1 fill:#fff3e0
    style ENG2 fill:#e8f5e8
    style UTIL1 fill:#fce4ec
    style UTIL2 fill:#f1f8e9
```

## 📊 データ変換フロー

```mermaid
flowchart TD
    A[📁 入力フォルダ検証] --> B{✅ 有効？}
    B -->|No| C[❌ エラー表示]
    B -->|Yes| D[📋 ファイルペア取得]
    
    D --> E[🏷️ クラス情報収集]
    E --> F[🔀 train/val分割]
    
    F --> G[🔄 座標変換処理]
    G --> H[📁 ファイル出力]
    
    H --> I[📝 dataset.yaml生成]
    H --> J[📋 classes.names生成]
    H --> K[📄 README.txt生成]
    
    I --> L[✅ 変換完了]
    J --> L
    K --> L
    
    L --> M[🔍 検証実行]
    M --> N[📊 結果表示]
    
    style A fill:#e3f2fd
    style G fill:#fff3e0
    style L fill:#e8f5e8
    style N fill:#f3e5f5
```

## 🎯 推定アルゴリズムフロー

```mermaid
flowchart TD
    A[📊 現在のデータセット分析] --> B[🏷️ クラス別統計取得]
    B --> C[🧮 複雑度判定]
    
    C --> D{📝 オブジェクト種別}
    D -->|簡単| E[🚗 ベース値: 70-200]
    D -->|中程度| F[🏠 ベース値: 150-400]
    D -->|複雑| G[🔧 ベース値: 300-800]
    
    E --> H[⚙️ 補正計算]
    F --> H
    G --> H
    
    H --> I[🎯 精度補正<br/>×1.0-4.0]
    I --> J[📏 サイズ補正<br/>×0.8-1.4]
    
    J --> K[📈 最終推奨値]
    K --> L[📋 不足数計算]
    L --> M[🎯 優先度判定]
    M --> N[📊 レポート生成]
    
    style A fill:#e3f2fd
    style H fill:#fff3e0
    style K fill:#e8f5e8
    style N fill:#f3e5f5
```

## 📁 フォルダ構造

```
yolo_dataset_analyzer/
├── 📄 README.md              # プロジェクト説明
├── 📄 requirements.txt       # 依存関係
├── 📄 pyproject.toml         # プロジェクト設定
├── 🚀 start.py              # 起動スクリプト
├── 📁 src/                   # ソースコード
│   ├── 🧮 analyzer/          # 分析エンジン
│   │   ├── estimator.py      # 推定アルゴリズム
│   │   └── __init__.py
│   ├── 🔄 converter/         # 変換エンジン
│   │   ├── yolo_converter.py # YOLO変換
│   │   └── __init__.py
│   ├── 🖥️ ui/               # ユーザーインターフェース
│   │   ├── streamlit_app.py  # メインUI
│   │   └── __init__.py
│   ├── 🔧 utils/            # ユーティリティ
│   │   ├── file_utils.py     # ファイル操作
│   │   └── __init__.py
│   └── __init__.py
├── 📁 docs/                  # ドキュメント
│   └── diagrams.md           # このファイル
├── 📁 tests/                 # テスト（将来用）
├── 📁 examples/              # サンプルデータ（将来用）
└── 📁 dist/                  # 配布パッケージ（将来用）
```

## 🔄 データ形式変換

### labelme形式 → YOLO形式

```mermaid
graph LR
    subgraph "📥 入力 (labelme)"
        A1[image1.jpg]
        A2[image1.json<br/>{<br/>&nbsp;&nbsp;shapes: [<br/>&nbsp;&nbsp;&nbsp;&nbsp;{<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;label: 'car',<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;points: [[x1,y1],[x2,y2]]<br/>&nbsp;&nbsp;&nbsp;&nbsp;}<br/>&nbsp;&nbsp;]<br/>}]
    end
    
    subgraph "🔄 変換処理"
        B1[座標変換<br/>矩形 → 中心点+幅高さ]
        B2[正規化<br/>0-1範囲に変換]
        B3[クラスID<br/>文字列 → 数値]
    end
    
    subgraph "📤 出力 (YOLO)"
        C1[image1.jpg]
        C2[image1.txt<br/>0 0.5 0.3 0.2 0.1]
        C3[dataset.yaml<br/>classes: ['car']<br/>train: images/train<br/>val: images/val]
    end
    
    A1 --> B1
    A2 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> C1
    B3 --> C2
    B3 --> C3
    
    style A2 fill:#dae8fc
    style B1 fill:#fff2cc
    style B2 fill:#fff2cc
    style B3 fill:#fff2cc
    style C2 fill:#f8cecc
    style C3 fill:#f8cecc
```

---

## 📋 使い方（図解付き）

### Step 1: 分析タブでデータセット確認
```
📊 分析タブ → フォルダ指定 → 分析実行 → 📈 結果確認
```

### Step 2: 変換タブでYOLO形式に変換
```
🔄 変換タブ → 入力/出力指定 → 変換実行 → 🎯 YOLO形式出力
```

### Step 3: YOLOv9で学習
```
🚀 生成されたdataset.yamlを使用してYOLOv9学習開始
```