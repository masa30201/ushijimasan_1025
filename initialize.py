"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import constants as ct

############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()

# trafilatura(WebBaseLoader内部)向けのUSER_AGENTを標準で設定
os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Streamlit/1.x")

# Streamlit Secrets が無い/構造が違っても落ちない安全ラッパー
def _get_secret_safe(key: str):
    """secrets.toml が無い・構造が異なる環境でも落ちない安全ラッパー"""
    try:
        return st.secrets[key]
    except Exception:
        # 例えば secrets = {"openai": {"api_key": "..."}} の場合にも対応
        try:
            if key == "OPENAI_API_KEY":
                return st.secrets["openai"]["api_key"]
        except Exception:
            return None
        return None

def ensure_openai_key():
    """
    OPENAI_API_KEY を .env → Streamlit Secrets の順で探し、
    見つけたら環境変数に設定（LangChain / OpenAI SDK が参照）
    """
    key = os.getenv("OPENAI_API_KEY") or _get_secret_safe("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY が見つかりません。.env か Streamlit Secrets に設定してください。"
        )
    os.environ["OPENAI_API_KEY"] = key

############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()

    # APIキー確認（ロガー準備後に行い、失敗してもログに残せるように）
    ensure_openai_key()

    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        return
    
    # RAGの参照先となるデータソースの読み込み
    docs_all = load_data_sources()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in list(doc.metadata.keys()):
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    try:
        # 埋め込みモデルの用意
        embeddings = OpenAIEmbeddings()
        
        # チャンク分割用のオブジェクトを作成
        text_splitter = CharacterTextSplitter(
            chunk_size=ct.CHUNK_SIZE,
            chunk_overlap=ct.CHUNK_OVERLAP,
            separator="\n"
        )

        # チャンク分割を実施
        splitted_docs = text_splitter.split_documents(docs_all)

        # ベクターストアの作成
        db = Chroma.from_documents(splitted_docs, embedding=embeddings)

        # ベクターストアを検索するRetrieverの作成
        st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RETRIEVER_K})
    except Exception as e:
        logger.error(f"ベクターストア初期化失敗: {e}")
        # ここでは従来挙動どおり停止させる（上位でハンドリング）
        raise


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        try:
            loader = WebBaseLoader(web_url)
            web_docs = loader.load()
            web_docs_all.extend(web_docs)
        except Exception as e:
            logger.warning(f"Web読み込み失敗: {web_url} : {e}")
            # Webはスキップして続行
            continue

    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        try:
            # ファイルの拡張子に合った data loader を使ってデータ読み込み
            loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
            docs = loader.load()
            docs_all.extend(docs)
        except Exception as e:
            # 壊れ/非対応ファイルはスキップして続行（原因はログに残す）
            logger.warning(f"ファイル読み込み失敗: {path} ({e})")


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s