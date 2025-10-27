"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""
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

# .env の読み込み
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


def get_llm_response(chat_message: str):
    """
    ユーザー入力(chat_message)に対する回答を生成する。

    2つのモードで動作:
    - ANSWER_MODE_1（社内文書検索）: RAGあり（ベクターストア検索＋根拠表示）
    - ANSWER_MODE_2（社内問い合わせ）: RAGなし（LLMのみで回答）

    戻り値は常に
        {"answer": "...", "context": [...]}
    の形。
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # LLMインスタンスを用意
    # ※ langchain-openai >= 0.2 系では ChatOpenAI(model=...) が正しい
    llm = ChatOpenAI(
        model=ct.MODEL,
        temperature=ct.TEMPERATURE,
    )

    # mode は main.py 側で st.session_state.mode に入れている想定
    mode = getattr(st.session_state, "mode", ct.ANSWER_MODE_1)

    if mode == ct.ANSWER_MODE_1:
        # ==========================
        # 社内文書検索 (RAGあり)
        # ==========================

        # 会話履歴つきの検索クエリ生成プロンプト
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # 会話履歴込み retriever
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=st.session_state.retriever,
            prompt=contextualize_q_prompt,
        )

        # 回答用プロンプト（根拠付き回答を出す）
        question_answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ct.SYSTEM_PROMPT_DOC_SEARCH),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # 参照ドキュメントをLLMに食わせて回答文を組み立てるチェーン
        question_answer_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=question_answer_prompt,
        )

        # 「履歴付きretriever」+「回答生成チェーン」をつないだRAGパイプライン
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain,
        )

        try:
            result = rag_chain.invoke(
                {
                    "input": chat_message,
                    "chat_history": st.session_state.chat_history,
                }
            )
            # result は {"answer": "...", "context": [...] } が期待される
            answer_text = result.get("answer", "")
            _update_chat_history(chat_message, answer_text)
            return result

        except Exception as e:
            logger.error(f"[RAGモード] LLM呼び出しに失敗しました: {e}")

            debug_answer = (
                "【LLM呼び出しでエラーが発生しました（RAGモード）】\n"
                "内部の検索または回答生成処理でエラーが発生しました。\n"
                f"詳細: {e}\n"
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

        inquiry_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ct.SYSTEM_PROMPT_INQUIRY),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # PromptTemplate → 実際のメッセージ列に展開
        messages_for_llm = inquiry_prompt.format_messages(
            input=chat_message,
            chat_history=st.session_state.chat_history,
        )

        try:
            ai_message = llm.invoke(messages_for_llm)
            answer_text = ai_message.content
        except Exception as e:
            logger.error(f"[問い合わせモード] LLM呼び出しに失敗しました: {e}")

            answer_text = (
                "【LLM呼び出しでエラーが発生しました（社内問い合わせモード）】\n"
                f"詳細: {e}\n"
            )

        _update_chat_history(chat_message, answer_text)

        return {
            "answer": answer_text,
            "context": [],
        }


def format_file_info(path: str, page_number: int | None = None) -> str:
    """
    参照元ドキュメントの表示用文字列を返す。
    PDFのときだけ『（ページNo.X）』を付けて返す。
    """
    if (
        isinstance(path, str)
        and path.lower().endswith(".pdf")
        and page_number is not None
    ):
        return f"{path}（ページNo.{page_number}）"
    return path
