"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain

import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# ユーティリティ関数
############################################################

def get_source_icon(source: str):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか（ファイルパス or URL）

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    if isinstance(source, str) and source.startswith("http"):
        return ct.LINK_SOURCE_ICON
    else:
        return ct.DOC_SOURCE_ICON


def build_error_message(message: str) -> str:
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def _update_chat_history(user_text: str, ai_text: str):
    """
    chat_history (会話の文脈) を HumanMessage / AIMessage で更新する。
    失敗してもアプリ自体は止めない。
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    try:
        st.session_state.chat_history.extend(
            [
                HumanMessage(content=user_text),
                AIMessage(content=ai_text),
            ]
        )
    except Exception as e:
        logger.warning(f"chat_history 更新に失敗しました: {e}")


############################################################
# LLM応答生成のメイン関数
############################################################

def get_llm_response(chat_message: str):
    """
    ユーザー入力(chat_message)に対する回答を生成する。

    2つのモードで動作する:
    - ANSWER_MODE_1（社内文書検索）:
        RAG（ベクターストア検索）を使い、関連ドキュメントも返す
    - ANSWER_MODE_2（社内問い合わせ）:
        RAGを使わず、LLMのみで回答を作る（contextは空）

    返り値の形式は共通で、
        {
            "answer": "...",
            "context": [Document(...), ...]  # or []
        }
    を返す。
    components.py / main.py はこの形式を前提にしている。
    """

    logger = logging.getLogger(ct.LOGGER_NAME)

    # -----------------------------------------------------
    # 0. LLMインスタンスを準備（ここも例外になることがあるので try で囲む）
    #    - APIキー未設定
    #    - モデル名が不正・権限なし
    #    - pydanticバリデーションエラー 等
    # -----------------------------------------------------
    try:
        llm = ChatOpenAI(
            model=ct.MODEL,            # langchain-openai>=0.2系では model_name ではなく model
            temperature=ct.TEMPERATURE,
        )
    except Exception as e:
        debug_answer = (
            "【LLM初期化エラー】\n"
            "ChatOpenAI の初期化に失敗しました。\n"
            f"詳細: {e}\n\n"
            "考えられる原因:\n"
            " - OPENAI_API_KEY が読み込まれていない\n"
            " - ct.MODEL のモデル名がこのAPIキーで使えない\n"
            " - モデル名のタイプミス\n"
        )
        _update_chat_history(chat_message, debug_answer)
        return {
            "answer": debug_answer,
            "context": [],
        }

    # いまのモード（initialize.py / main.py 側で設定済みのはず）
    mode = getattr(st.session_state, "mode", ct.ANSWER_MODE_1)

    # -----------------------------------------------------
    # モード分岐
    # -----------------------------------------------------

    if mode == ct.ANSWER_MODE_1:
        # =================================================
        # 社内文書検索 (RAGあり)
        # =================================================

        # 1. 会話履歴を踏まえた検索クエリを作るためのプロンプト
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # 2. Retriever を会話履歴込みにするチェーン
        #    ここではまだ retriever を実行しないので
        #    Chroma 内部の "no such table" は起きないはず
        try:
            history_aware_retriever = create_history_aware_retriever(
                llm=llm,
                retriever=st.session_state.retriever,
                prompt=contextualize_q_prompt,
            )
        except Exception as e:
            debug_answer = (
                "【RAG初期化エラー】\n"
                "履歴付きRetrieverの構築に失敗しました。\n"
                f"詳細: {e}\n\n"
                "考えられる原因:\n"
                " - ベクターストア(retriever)が壊れている/未初期化\n"
                " - Chromaの永続ディレクトリが壊れた\n"
            )
            _update_chat_history(chat_message, debug_answer)
            return {
                "answer": debug
