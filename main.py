"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。修正しました。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
from dotenv import load_dotenv        # 「.env」ファイルから環境変数を読み込む
import logging                        # ログ出力
import streamlit as st               # Streamlit本体

import utils                         # 画面表示以外の関数
from initialize import initialize    # 起動時の初期化処理
import components as cn              # 画面表示系（タイトルやログ表示など）
import constants as ct               # 定数モジュール


############################################################
# 2. ページ設定・ロガー初期化
############################################################
st.set_page_config(
    page_title=ct.APP_NAME,
    layout="wide",
)

logger = logging.getLogger(ct.LOGGER_NAME)


############################################################
# 3. アプリ全体の初期化処理
############################################################
try:
    initialize()
except Exception as e:
    logger.error(f"{ct.INITIALIZE_ERROR_MESSAGE}\n{e}")
    st.error(utils.build_error_message(ct.INITIALIZE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    st.stop()

# 初回起動ログ（多重防止）
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)

# デフォルトモードを保証
if "mode" not in st.session_state:
    st.session_state.mode = ct.ANSWER_MODE_1


############################################################
# 4. 画面上部・サイドバー
############################################################
cn.display_app_title()
cn.display_select_mode()         # ここで st.session_state.mode を更新する想定
cn.display_initial_ai_message()


############################################################
# 5. これまでの会話ログの表示
############################################################
try:
    cn.display_conversation_log()
except Exception as e:
    logger.error(f"{ct.CONVERSATION_LOG_ERROR_MESSAGE}\n{e}")
    st.error(utils.build_error_message(ct.CONVERSATION_LOG_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    st.stop()


############################################################
# 6. チャット入力欄
############################################################
chat_message = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)


############################################################
# 7. 入力が送信されたときの処理
############################################################
if chat_message:
    # 7-1. ユーザーメッセージをまず表示
    logger.info({"message": chat_message, "application_mode": st.session_state.mode})

    with st.chat_message("user"):
        st.markdown(chat_message)

    # 7-2. LLM応答を生成
    with st.spinner(ct.SPINNER_TEXT):
        try:
            llm_response = utils.get_llm_response(chat_message)
        except Exception as e:
            logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
            llm_response = {
                "answer": utils.build_error_message(ct.GET_LLM_RESPONSE_ERROR_MESSAGE),
                "context": [],
            }

    # 7-3. アシスタントの返答描画
    content_for_history = None

    with st.chat_message("assistant"):
        try:
            # ─────────────────────────────
            # モードA: 社内文書検索（RAGあり表示UI）
            # ─────────────────────────────
            if st.session_state.mode == ct.ANSWER_MODE_1:
                # この関数はすでに正しく動いている（ファイル候補も表示されている）
                content_for_history = cn.display_search_llm_response(llm_response)

            # ─────────────────────────────
            # モードB: 社内問い合わせ（LLM単独応答）
            # ─────────────────────────────
            elif st.session_state.mode == ct.ANSWER_MODE_2:
                # ★ここが超重要★
                # components.display_contact_llm_response() は呼ばない。
                # かわりに main.py で直接表示する。

                from utils import format_file_info, get_source_icon

                # LLMの回答本文
                answer_text = llm_response.get("answer", "")
                if not answer_text:
                    answer_text = "（回答テキストが取得できませんでした）"

                st.markdown(answer_text)

                # contextがあれば「情報源」として列挙（通常は空でOK）
                context_docs = llm_response.get("context", [])
                file_info_list = []
                message_for_user = ""

                if context_docs:
                    st.markdown("---")
                    st.subheader("情報源")

                    for doc in context_docs:
                        meta = getattr(doc, "metadata", {}) or {}
                        raw_path = meta.get("source", "")
                        page_num = meta.get("page", None)

                        pretty_path = format_file_info(raw_path, page_num)
                        icon = get_source_icon(raw_path)

                        st.markdown(f"{icon} {pretty_path}")
                        file_info_list.append(pretty_path)

                    message_for_user = "参考として、関連する情報源の候補を表示しました。"

                # ログ用の構造体（会話履歴に積むやつ）
                content_for_history = {
                    "mode": ct.ANSWER_MODE_2,
                    "answer": answer_text,
                }
                # 「検索しても該当なし」みたいな固定文がINQUIRY_NO_MATCH_ANSWERとして定義されてることがある
                no_match_answer = getattr(ct, "INQUIRY_NO_MATCH_ANSWER", None)
                if answer_text != no_match_answer:
                    if message_for_user:
                        content_for_history["message"] = message_for_user
                    if file_info_list:
                        content_for_history["file_info_list"] = file_info_list

            # ─────────────────────────────
            # 想定外モード
            # ─────────────────────────────
            else:
                fallback_msg = (
                    "現在のモードが不明です。もう一度モードを選択して質問を送信してください。"
                )
                st.markdown(fallback_msg)
                content_for_history = {
                    "mode": "unknown",
                    "answer": fallback_msg,
                }

            # ログにも残す
            logger.info({
                "message": content_for_history,
                "application_mode": st.session_state.mode
            })

        except Exception as e:
            logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")
            fallback_error_msg = utils.build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE)
            st.error(fallback_error_msg, icon=ct.ERROR_ICON)

            content_for_history = {
                "mode": st.session_state.mode,
                "answer": fallback_error_msg,
            }

    # 7-4. 会話ログの記録（画面の履歴用）
    st.session_state.messages.append({
        "role": "user",
        "content": chat_message,
    })
    st.session_state.messages.append({
        "role": "assistant",
        "content": content_for_history,
    })
