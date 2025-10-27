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
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if isinstance(source, str) and source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON

    return icon


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
        llm_response: dict
            例:
            {
                "answer": "〜〜〜回答テキスト〜〜〜",
                "context": [Document(...), Document(...), ...]
            }

        ※ "answer" と "context" は components.py 側で利用される想定
    """

    logger = logging.getLogger(ct.LOGGER_NAME)

    # -----------------------------------------------------
    # 1. LLMインスタンスの用意
    #    ※ langchain-openai>=0.2.x では model_name ではなく model を使う
    # -----------------------------------------------------
    llm = ChatOpenAI(
        model=ct.MODEL,
        temperature=ct.TEMPERATURE,
    )

    # -----------------------------------------------------
    # 2. 「会話履歴を踏まえても、単体で意味が通る質問文」を生成するプロンプト
    #    → 会話のニュアンスを含んだ検索クエリを作るため
    # -----------------------------------------------------
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # -----------------------------------------------------
    # 3. 回答用プロンプト（モード別に system プロンプトを切り替える）
    #    - 社内文書検索（根拠つき回答を重視）
    #    - 社内問い合わせ（社内制度Q&A想定）
    # -----------------------------------------------------
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # モードが「社内文書検索」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        # モードが「社内問い合わせ」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY

    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # -----------------------------------------------------
    # 4. 会話履歴つきRetrieverの組み立て
    #    - create_history_aware_retriever:
    #        "過去のやりとり"＋"今回の質問" → 検索用のクエリをLLMで整形
    #    - これにより st.session_state.retriever（Chroma経由）に
    #      文脈に合った問い合わせが飛ぶ
    # -----------------------------------------------------
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=st.session_state.retriever,
        prompt=question_generator_prompt,
    )

    # -----------------------------------------------------
    # 5. 取り出したドキュメントをもとに最終回答を作るチェーン
    #    - create_stuff_documents_chain:
    #        参照ドキュメントをまとめてLLMに渡し、回答文を組ませる
    # -----------------------------------------------------
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=question_answer_prompt,
    )

    # -----------------------------------------------------
    # 6. Retrievalと回答生成をまとめたRAGチェーン
    #    - create_retrieval_chain:
    #        (文脈対応retriever) → (回答生成チェーン)
    # -----------------------------------------------------
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain,
    )

    # -----------------------------------------------------
    # 7. チェーン実行（LLM呼び出し）
    #    LangChain 0.3系では chat_history に
    #    HumanMessage / AIMessage のリストを渡すことが推奨
    # -----------------------------------------------------
    try:
        llm_response = rag_chain.invoke(
            {
                "input": chat_message,
                "chat_history": st.session_state.chat_history,
            }
        )
        # llm_response は典型的に:
        # {
        #   "answer": "<モデルの回答テキスト>",
        #   "context": [Document(...), ...]
        # }

    except Exception as e:
        logger.error(f"LLM呼び出しに失敗しました: {e}")
        # ここで例外を投げ直すことで main.py 側の try/except に捕まって
        # 画面に「回答生成に失敗しました。」が出る
        raise

    # -----------------------------------------------------
    # 8. 会話履歴を更新
    #    - 次の質問時に、文脈を保持したままRetriever＆LLMに渡すため
    #    - HumanMessage / AIMessage を積む
    # -----------------------------------------------------
    try:
        st.session_state.chat_history.extend(
            [
                HumanMessage(content=chat_message),
                AIMessage(content=llm_response["answer"]),
            ]
        )
    except Exception as e:
        # 履歴の更新に失敗しても、回答自体は返す
        logger.warning(f"chat_history 更新に失敗しました: {e}")

    return llm_response


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
