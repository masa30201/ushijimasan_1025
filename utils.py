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
# 関数定義
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


def get_llm_response(chat_message: str):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値（テキスト）

    Returns:
        dict:
            {
                "answer": "...",
                "context": [Document(...), ...]
            }
        ※ components.py / main.py はこの形式を前提にしている
    """

    logger = logging.getLogger(ct.LOGGER_NAME)

    # 1. LLM本体の用意
    #    langchain-openai>=0.2系では model_name ではなく model を使う
    llm = ChatOpenAI(
        model=ct.MODEL,
        temperature=ct.TEMPERATURE,
    )

    # 2. 「会話履歴を踏まえた検索クエリを作る」プロンプト
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # retriever（RAG用の検索器）に、会話履歴を考慮した検索クエリを渡すチェーン
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=st.session_state.retriever,
        prompt=contextualize_q_prompt,
    )

    # 3. 回答生成用プロンプト（モードに応じて system を切り替える）
    if st.session_state.mode == ct.ANSWER_MODE_1:
        system_prompt = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        system_prompt = ct.SYSTEM_PROMPT_INQUIRY

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 4. 取得したドキュメントをまとめてLLMに渡して、回答文を組み立てるチェーン
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
    )

    # 5. (会話履歴を考慮したretriever)＋(回答生成チェーン) のRAGパイプライン
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain,
    )

    # 6. チェーン実行
    try:
        llm_response = rag_chain.invoke(
            {
                "input": chat_message,
                "chat_history": st.session_state.chat_history,
            }
        )
        # llm_response 例:
        # {
        #   "answer": "〜モデルが生成した回答〜",
        #   "context": [Document(...), ...]
        # }

        # 会話履歴を更新（次回以降の文脈のため）
        try:
            st.session_state.chat_history.extend(
                [
                    HumanMessage(content=chat_message),
                    AIMessage(content=llm_response.get("answer", "")),
                ]
            )
        except Exception as hist_err:
            logger.warning(f"chat_history 更新に失敗しました: {hist_err}")

        return llm_response

    except Exception as e:
        # ここに来る = OpenAIのチャットモデル呼び出しなどで失敗した
        # 代表例:
        #  - モデル名が無効 / 権限がない
        #  - レート制限
        #  - APIキー権限不足
        logger.error(f"LLM呼び出しに失敗しました: {e}")

        # main.py 側に例外を投げると赤い帯しか出ないので、
        # ここでは「エラー内容そのものをanswerとして返す」ことで
        # 画面に可視化してデバッグしやすくする。
        debug_answer = (
            "【LLM呼び出しでエラーが発生しました】\n"
            "おそらく OpenAI のチャットモデル呼び出し時のエラーです。\n"
            f"詳細: {e}\n\n"
            "考えられる原因:\n"
            " - ct.MODEL のモデル名が現在のAPIキーで使えない\n"
            " - 利用上限/レートリミットに達した\n"
            " - モデル名のタイプミス\n"
        )

        # 会話履歴にはエラーも一応残しておく（次の問い合わせは普通に通る可能性もある）
        try:
            st.session_state.chat_history.extend(
                [
                    HumanMessage(content=chat_message),
                    AIMessage(content=debug_answer),
                ]
            )
        except Exception as hist_err:
            logger.warning(f"chat_history 更新に失敗しました(エラー時): {hist_err}")

        # components.py 側が期待する形
        return {
            "answer": debug_answer,
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
