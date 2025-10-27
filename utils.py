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

    Returns:
        dict:
            {
                "answer": "...回答テキスト...",
                "context": [Document(...), ...]  # or []
            }
        components.py / main.py はこの形式を前提にしている
    """

    logger = logging.getLogger(ct.LOGGER_NAME)

    # LLM本体を用意
    # langchain-openai>=0.2系では model_name ではなく model を使う
    llm = ChatOpenAI(
        model=ct.MODEL,
        temperature=ct.TEMPERATURE,
    )

    # どのモードで呼ばれたかによって処理を分岐
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # ==========================
        # 社内文書検索 (RAGあり)
        # ==========================

        # 1. 会話履歴を踏まえた検索クエリを作るためのプロンプト
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # 2. 会話履歴を考慮した Retriever を組み立てる
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=st.session_state.retriever,  # initialize.pyで用意/キャッシュ済みのやつ
            prompt=contextualize_q_prompt,
        )

        # 3. 実際の回答を作るためのプロンプト（ドキュメントを根拠に回答する想定）
        question_answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ct.SYSTEM_PROMPT_DOC_SEARCH),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # 4. ドキュメント群をまとめて最終回答を作るチェーン
        question_answer_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=question_answer_prompt,
        )

        # 5. 「履歴込みRetriever」→「回答生成」のRAGチェーン
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain,
        )

        # 6. 実行
        try:
            result = rag_chain.invoke(
                {
                    "input": chat_message,
                    "chat_history": st.session_state.chat_history,
                }
            )
            # result 例:
            # {
            #   "answer": "〜モデルの回答〜",
            #   "context": [Document(...), ...]
            # }

            answer_text = result.get("answer", "")
            _update_chat_history(chat_message, answer_text)
            return result

        except Exception as e:
            # RAG経由での回答生成に失敗した場合は
            # ユーザー側にもエラー内容を可視化して返す
            logger.error(f"[RAGモード] LLM呼び出しに失敗しました: {e}")

            debug_answer = (
                "【LLM呼び出しでエラーが発生しました（RAGモード）】\n"
                "内部の検索または回答生成処理でエラーが発生しました。\n"
                f"詳細: {e}\n\n"
                "考えられる原因:\n"
                " - ベクターストアの内部エラー / 破損\n"
                " - モデル呼び出しの権限・レート制限\n"
                " - LangChain/Chromaのバージョン差異による不整合\n"
            )

            _update_chat_history(chat_message, debug_answer)

            return {
                "answer": debug_answer,
                "context": [],
            }

    else:
        # ==========================
        # 社内問い合わせ (RAGなし)
        # ==========================
        #
        # ここでは Chroma / retriever に触れない。
        # つまり「no such table: collections」のような
        # ベクターストア絡みのエラーを回避できる。
        #
        # chat_historyを踏まえてLLMに直接聞く。
        # （SYSTEM_PROMPT_INQUIRYは「社内問い合わせ」用のルール）
        #

        inquiry_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ct.SYSTEM_PROMPT_INQUIRY),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # PromptTemplate を具体的なメッセージ列に展開する
        # - MessagesPlaceholder("chat_history") には
        #   st.session_state.chat_history (HumanMessage / AIMessage のリスト) が入る
        messages_for_llm = inquiry_prompt.format_messages(
            input=chat_message,
            chat_history=st.session_state.chat_history,
        )

        try:
            ai_message = llm.invoke(messages_for_llm)
            answer_text = ai_message.content

        except Exception as e:
            logger.error(f"[社内問い合わせモード] LLM呼び出しに失敗しました: {e}")

            answer_text = (
                "【LLM呼び出しでエラーが発生しました（社内問い合わせモード）】\n"
                "モデル呼び出し時にエラーが発生しました。\n"
                f"詳細: {e}\n\n"
                "考えられる原因:\n"
                " - ct.MODEL のモデル名/権限の問題\n"
                " - APIキーの制限やレートリミット\n"
            )

        # 会話履歴を更新（RAGなしなので context は返さない）
        _update_chat_history(chat_message, answer_text)

        return {
            "answer": answer_text,
            "context": [],
        }


def format_file_info(path: str, page_number: int | None = None) -> str:
    """
    参照元ドキュメントの表示用文字列を返す。
    PDFのときだけ『（ページNo.X）』を付けて返す。

    Args:
        path: 参照元のファイルパス
        page_number: PDFの場合はページ番号（1始まり想定）

    Returns:
        UI表示用のテキスト
    """
    if (
        isinstance(path, str)
        and path.lower().endswith(".pdf")
        and page_number is not None
    ):
        return f"{path}（ページNo.{page_number}）"
    return path
