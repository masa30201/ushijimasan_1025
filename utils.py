"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

import logging
from typing import Optional
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

# .env の読み込み（ローカル実行用）
load_dotenv()


def get_source_icon(source: str):
    """
    メッセージと一緒に表示するアイコンの種類を取得
    """
    if isinstance(source, str) and source.startswith("http"):
        return ct.LINK_SOURCE_ICON
    else:
        return ct.DOC_SOURCE_ICON


def build_error_message(message: str) -> str:
    """
    エラーメッセージと管理者問い合わせテンプレートの連結
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def _update_chat_history(user_text: str, ai_text: str):
    """
    chat_history (LLMに渡す会話の文脈) を HumanMessage / AIMessage で更新する。
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
        logger.warning("chat_history 更新に失敗しました: %s", e)


def get_llm_response(chat_message: str):
    """
    ユーザー入力(chat_message)に対する回答を生成する。

    両モードとも RAG（ベクターストア検索 + 参照文書をLLMに渡す）を使う。
    ただし、LLMに渡すシステムプロンプトはモードによって切り替える。

    - ANSWER_MODE_1（社内文書検索）:
        ct.SYSTEM_PROMPT_DOC_SEARCH を使う
        → ファイル候補や根拠の提示に向いたスタイル
    - ANSWER_MODE_2（社内問い合わせ）:
        ct.SYSTEM_PROMPT_INQUIRY を使う
        → 社内問い合わせの回答トーン・方針に向いたスタイル

    返り値は常に:
        {
            "answer": "...",       # LLMの最終回答 or エラーメッセージ
            "context": [Document, ...]  # retrieverが返したドキュメントのリスト（なければ空）
        }

    どんな場合も raise はしない。必ず上の dict を返す。
    main.py 側はこの dict を前提に表示・ログ保存する。
    """

    logger = logging.getLogger(ct.LOGGER_NAME)

    # 0. LLMインスタンスの準備
    try:
        llm = ChatOpenAI(
            model=ct.MODEL,          # langchain-openai>=0.2系では model= が正
            temperature=ct.TEMPERATURE,
        )
    except Exception as e:
        debug_answer = (
            "【LLM初期化エラー】\n"
            "ChatOpenAI の初期化に失敗しました。\n"
            "考えられる原因:\n"
            " - OPENAI_API_KEY が未設定 / 無効\n"
            " - ct.MODEL のモデル名にこのAPIキーでの権限がない\n"
            " - モデル名のタイプミス\n\n"
            f"詳細:\n{e}"
        )
        _update_chat_history(chat_message, debug_answer)
        return {
            "answer": debug_answer,
            "context": [],
        }

    # 1. いまのモードを取得（サイドバーのラジオで選ばれている想定）
    mode = getattr(st.session_state, "mode", ct.ANSWER_MODE_1)

    # 2. プロンプトをモード別に組み立てる
    #    - 「どんなふうに答えるべきか」を決める system プロンプトだけ差し替えるイメージ
    if mode == ct.ANSWER_MODE_2:
        # 社内問い合わせモード
        answer_system_prompt_text = ct.SYSTEM_PROMPT_INQUIRY
    else:
        # デフォルトは社内文書検索モード扱い
        answer_system_prompt_text = ct.SYSTEM_PROMPT_DOC_SEARCH

    # 会話履歴を踏まえた検索クエリを作るためのプロンプト
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # LLMに最終回答をまとめさせるプロンプト
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", answer_system_prompt_text),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 3. Retriever（Chromaなど）を、会話履歴込みで使えるようにする
    try:
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=st.session_state.retriever,
            prompt=contextualize_q_prompt,
        )
    except Exception as e:
        debug_answer = (
            "【RAG初期化エラー】\n"
            "社内文書を検索するためのRetrieverを初期化できませんでした。\n"
            "考えられる原因:\n"
            " - ベクターストア(Chroma)が未初期化/破損\n"
            " - Chromaの永続ディレクトリが壊れている\n\n"
            f"詳細:\n{e}"
        )
        _update_chat_history(chat_message, debug_answer)
        return {
            "answer": debug_answer,
            "context": [],
        }

    # 4. ドキュメントをまとめて最終回答を作るチェーン
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=question_answer_prompt,
    )

    # 5. 「(履歴付き)Retriever」→「回答生成」のRAGチェーンを組む
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain,
    )

    # 6. 実際にRAGチェーンを呼び出して回答を作る
    try:
        result = rag_chain.invoke(
            {
                "input": chat_message,
                "chat_history": st.session_state.chat_history,
            }
        )
        # LangChainのcreate_retrieval_chain()はだいたい:
        # {
        #   "answer": "...LLMの最終回答テキスト...",
        #   "context": [Document(...), ...]
        # }
        answer_text = result.get("answer", "")
        _update_chat_history(chat_message, answer_text)
        return result

    except Exception as e:
        # retriever呼び出しやOpenAI呼び出しで落ちた場合
        debug_answer = (
            "【LLM呼び出しでエラーが発生しました】\n"
            "内部の検索または回答生成処理でエラーが発生しました。\n"
            "考えられる原因:\n"
            " - ベクターストア(Chroma)の内部エラー/テーブル破損\n"
            " - モデル呼び出しの権限・レート制限\n"
            " - LangChain / Chroma のバージョン不整合\n\n"
            f"詳細:\n{e}"
        )
        _update_chat_history(chat_message, debug_answer)
        return {
            "answer": debug_answer,
            "context": [],
        }


def format_file_info(path: str, page_number: Optional[int] = None) -> str:
    """
    参照元ドキュメントの表示用文字列を返す。
    PDFのときだけ『（ページNo.X）』を付けて返す。
    """
    if (
        isinstance(path, str)
        and path.lower().endswith(".pdf")
        and page_number is not None
    ):
        return "%s（ページNo.%s）" % (path, page_number)
    return path
