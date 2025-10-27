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


############################################################
# UI用の小さいユーティリティ
############################################################

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


############################################################
# コア: LLM応答生成
############################################################

def get_llm_response(chat_message: str):
    """
    ユーザー入力(chat_message)に対する回答を生成する。

    モード別のふるまい:
    - ANSWER_MODE_1（社内文書検索モード）
        → ベクターストア検索(RAG) + 社内文書ベースの回答
        → ct.SYSTEM_PROMPT_DOC_SEARCH を使う
    - ANSWER_MODE_2（社内問い合わせモード）
        → ベクターストアを使わず、LLM単体で回答
        → ct.SYSTEM_PROMPT_INQUIRY を使う

    戻り値は常に:
        {
            "answer": "<LLM回答 or エラーメッセージ>",
            "context": [Document(...), ...]  # 問い合わせモードでは基本 []
        }

    ※ どんな場合でも raise はしない。
      main.py 側が try/except しなくても落ちないように、ここで握りつぶして answer に埋め込む。
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

    # 1. モード取得
    mode = getattr(st.session_state, "mode", ct.ANSWER_MODE_1)

    # =====================================================
    # モードB: 社内問い合わせ（RAGなしでLLMに直接聞く）
    # =====================================================
    if mode == ct.ANSWER_MODE_2:
        # ここでは Chroma / retriever に一切触らない。
        # → これで「no such table: collections」から完全に解放される。

        inquiry_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ct.SYSTEM_PROMPT_INQUIRY),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # ChatPromptTemplateを実際のメッセージ列に展開
        messages_for_llm = inquiry_prompt.format_messages(
            input=chat_message,
            chat_history=st.session_state.chat_history,
        )

        try:
            ai_message = llm.invoke(messages_for_llm)
            answer_text = ai_message.content
        except Exception as e:
            answer_text = (
                "【LLM呼び出しでエラーが発生しました（社内問い合わせモード）】\n"
                "モデル呼び出し時にエラーが発生しました。\n"
                "考えられる原因:\n"
                " - ct.MODEL のモデル名/権限の問題\n"
                " - APIキーの制限やレート制限\n\n"
                f"詳細:\n{e}"
            )

        _update_chat_history(chat_message, answer_text)

        return {
            "answer": answer_text,
            "context": [],
        }

    # =====================================================
    # モードA: 社内文書検索（RAGあり）
    # =====================================================

    # 会話履歴を踏まえて検索クエリを作るプロンプト
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 社内文書検索向けの回答プロンプト
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ct.SYSTEM_PROMPT_DOC_SEARCH),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 会話履歴込みの Retriever を組み立てる
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

    # 取得したドキュメントをもとに回答をまとめるチェーン
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=question_answer_prompt,
    )

    # Retriever→回答生成 というRAGチェーンを作る
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain,
    )

    # 実行（Chromaに問い合わせ、OpenAIで回答生成）
    try:
        result = rag_chain.invoke(
            {
                "input": chat_message,
                "chat_history": st.session_state.chat_history,
            }
        )
        # 期待形:
        # {
        #   "answer": "...LLM回答...",
        #   "context": [Document(...), ...]
        # }
        answer_text = result.get("answer", "")
        _update_chat_history(chat_message, answer_text)
        return result

    except Exception as e:
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


############################################################
# 表示用の補助（ファイルパスとページ番号を見やすくする）
############################################################

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
