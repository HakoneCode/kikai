# 課題1.産業用製品画像の欠陥検出（画像検出）　概要   
産業用の画像データから、製品が欠陥であるか否かを予測するモデルを作成します。  
企業は製品の品​​質を保証する必要がありますが、製品の中には稀に欠陥品が含まれてしまうことがあります。  
欠陥品は検査によって検出しますが、その過程を自動化することができれば、品質保証のコストを大幅に削減することができます。  

# チュートリアルコードの使い方  
チュートリアルコードはそれぞれ以下のリンクで確認できます。    
 ① [データアップロード](https://gitlab.intra.oki.co.jp/ml_contest/2023_CableConnectorClassification/-/blob/main/%E3%83%81%E3%83%A5%E3%83%BC%E3%83%88%E3%83%AA%E3%82%A2%E3%83%AB/MlCompe2023_01_%E3%83%87%E3%83%BC%E3%82%BF%E3%82%A2%E3%83%83%E3%83%97%E3%83%AD%E3%83%BC%E3%83%89_%E3%83%87%E3%83%BC%E3%82%BF%E6%BA%96%E5%82%99.ipynb)  
 ② [データの可視化](https://gitlab.intra.oki.co.jp/ml_contest/2023_CableConnectorClassification/-/blob/main/%E3%83%81%E3%83%A5%E3%83%BC%E3%83%88%E3%83%AA%E3%82%A2%E3%83%AB/MlCompe2023_02_%E5%AD%A6%E7%BF%92%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E5%8F%AF%E8%A6%96%E5%8C%96.ipynb)  
 ③ [学習の実行と提出ファイルの作成](https://gitlab.intra.oki.co.jp/ml_contest/2023_CableConnectorClassification/-/blob/main/%E3%83%81%E3%83%A5%E3%83%BC%E3%83%88%E3%83%AA%E3%82%A2%E3%83%AB/MlCompe2023_03_%E3%83%A2%E3%83%87%E3%83%AB%E4%BD%9C%E6%88%90_%E6%8A%95%E7%A8%BF%E3%83%87%E3%83%BC%E3%82%BF%E4%BD%9C%E6%88%90.ipynb)  
  
本ページでは、上記チュートリアルコードの解説になります。  

## GoogleColabへファイルをアップロード方法
※ 任意のGoogleアカウントを使用します。  
※※ 機械学習コンテスト用のアカウントを準備されることを お勧めします。  
  
ブラウザ（GoogleChrome）でGoogleColabを表示します。  
GoogleColaboratory（GoogleColab）　URL：https://colab.research.google.com/  

<p><img src="image/image_h-0.jpg" alt="GoogleColab" width=50%/></p>  

① 左上にある「ファイル」から[ノートブックをアップロード]を選択します。  

<p><img src="image/image_h-0-1.jpg" alt="GoogleColab1" width=50%/></p>  

② ファイルを選択します。 

# チュートリアル　概要  
## データアップロード 『データの準備』  
※　参考コード　① [データアップロード](https://gitlab.intra.oki.co.jp/ml_contest/2023_CableConnectorClassification/-/blob/main/%E3%83%81%E3%83%A5%E3%83%BC%E3%83%88%E3%83%AA%E3%82%A2%E3%83%AB/MlCompe2023_01_%E3%83%87%E3%83%BC%E3%82%BF%E3%82%A2%E3%83%83%E3%83%97%E3%83%AD%E3%83%BC%E3%83%89_%E3%83%87%E3%83%BC%E3%82%BF%E6%BA%96%E5%82%99.ipynb)   
SIGNATEのコンテストページから以下をダウンロードし、ローカルからgoogle driveへアップロードします。  
**※※ Google Drive との同期のため、実行完了後、30分程放置しておくことをお勧めします。**  

・ 学習用画像データ　train_images.zip  
・ 学習用画像データのアノテーション　train_annotations.tsv  
・ 評価用学習データ　test_images.zip  
・ 投稿用サンプルファイル　sample_submit.tsv  
  
## 学習データの可視化　『学習データ数、データの確認』  
※　参考コード② [データの可視化](https://gitlab.intra.oki.co.jp/ml_contest/2023_CableConnectorClassification/-/blob/main/%E3%83%81%E3%83%A5%E3%83%BC%E3%83%88%E3%83%AA%E3%82%A2%E3%83%AB/MlCompe2023_02_%E5%AD%A6%E7%BF%92%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E5%8F%AF%E8%A6%96%E5%8C%96.ipynb)  
画像を見て、その画像が欠陥である確率値を予測します。  
・　学習データをヒストグラムを使用して確認  
・　正常画像と異常画像を可視化し、学習の工夫（データ拡張）について検討  
  
## モデル作成、投稿データ作成  
※　参考コード　③ [学習の実行と提出ファイルの作成](https://gitlab.intra.oki.co.jp/ml_contest/2023_CableConnectorClassification/-/blob/main/%E3%83%81%E3%83%A5%E3%83%BC%E3%83%88%E3%83%AA%E3%82%A2%E3%83%AB/MlCompe2023_03_%E3%83%A2%E3%83%87%E3%83%AB%E4%BD%9C%E6%88%90_%E6%8A%95%E7%A8%BF%E3%83%87%E3%83%BC%E3%82%BF%E4%BD%9C%E6%88%90.ipynb)  
pytorchフレームワークを使用して学習・評価、提出データの作成をします。  
・ 学習過程の可視化設定  
・ 学習データの定義　前処理・データ拡張・データセット作成  
・ 学習の設定  
　　ハイパーパラメータの定義  
　　モデル定義  
　　データローダ作成  
　　損失関数  
　　最適化関数  
　　学習率スケジューラ  
・ 学習の開始  
・ モデルの保存  
・ 投稿データ作成  
