"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
from dotenv import load_dotenv        # 「.env」ファイルから環境変数を読み込むための関数
import logging                        # ログ出力を行うためのモジュール
import streamlit as st               # Streamlitアプリ本体

import utils                         # （自作）画面表示以外の様々な関数
from initialize import initialize    # （自作）アプリ起動時の初期化処理
import components as cn              # （自作）画面表示系の関数
import constants as ct               # （自作）定数定義モジュール


############################################################
# 2. ページ設定・ロガー初期化
############################################################
st.set_page_config(
    page_title=ct.APP_NAME,
    layout="wide",   # ワイドレイアウトで画面を広く使う
)

logger = logging.getLogger(ct.LOGGER_NAME)


############################################################
# 3. アプリ全体の初期化処理
############################################################
try:
    # initialize() の中では主に：
    # - session_state の初期化（messages / chat_history / retriever など）
    # - logger のセットアップ
    # - ベクターストア（retriever）の読み込み or 構築
    # - OPENAI_API_KEY のチェック
    initialize()
except Exception as e:
    logger.error(f"{ct.INITIALIZE_ERROR_MESSAGE}\n{e}")
    st.error(utils.build_error_message(ct.INITIALIZE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    st.stop()

# 初回起動ログの出力（多重出力防止）
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)

# 念のため mode の初期値を保証しておく
if "mode" not in st.session_state:
    st.session_state.mode = ct.ANSWER_MODE_1


############################################################
# 4. 画面上部・サイドバーなどの初期表示
############################################################
# タイトルや案内
cn.display_app_title()

# サイドバーのラジオボタンなどを描画し、
# st.session_state.mode を更新する想定の関数
cn.display_select_mode()

# 初回ガイダンスメッセージ（「こんにちは〜」など）
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
    # ==========================================
    # 7-1. ユーザーメッセージをまず表示
    # ==========================================
    logger.info({"message": chat_message, "application_mode": st.session_state.mode})

    with st.chat_message("user"):
        st.markdown(chat_message)

    # ==========================================
    # 7-2. LLM応答を生成
    # ==========================================
    # utils.get_llm_response() は必ず
    # {"answer": "...", "context": [...] } の dict を返す設計
    # -> 例外を raise しないようにしてある
    with st.spinner(ct.SPINNER_TEXT):
        try:
            llm_response = utils.get_llm_response(chat_message)
        except Exception as e:
            # 念のためのフォールバック（ここには基本こないはず）
            logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
            llm_response = {
                "answer": utils.build_error_message(ct.GET_LLM_RESPONSE_ERROR_MESSAGE),
                "context": [],
            }

    # ==========================================
    # 7-3. アシスタントの返答を画面に描画
    # ==========================================
    content_for_history = None  # 後で st.session_state.messages に積む用のオブジェクト

    with st.chat_message("assistant"):
        try:
            # -------------------------
            # (A) モード: 社内文書検索
            # -------------------------
            if st.session_state.mode == ct.ANSWER_MODE_1:
                # 既存の表示ロジックに任せる
                # display_search_llm_response は、関連文書リストや
                # 参考ファイルなどをきれいに表示する想定の関数
                # 戻り値はログ用（文字列や dict）という既存仕様を尊重する
                content_for_history = cn.display_search_llm_response(llm_response)

            # -------------------------
            # (B) モード: 社内問い合わせ
            # -------------------------
            elif st.session_state.mode == ct.ANSWER_MODE_2:
                # ここは components.display_contact_llm_response() を
                # 直接呼ばず、main.py 側で描画する。
                #
                # 理由:
                # - 古い components.py が固定の
                #   「回答生成に失敗しました…」だけを出しており
                #   LLMの本当の回答（もしくはエラーメッセージ詳細）が
                #   画面に見えない状態になっているため。
                #
                # ここでやりたいこと:
                #   1. LLMの回答本文 (llm_response["answer"]) をそのまま表示
                #   2. context があれば "情報源" として列挙
                #   3. 会話ログ用の content(dict) を作って返す
                from utils import format_file_info, get_source_icon

                # 1. 本文
                answer_text = llm_response.get("answer", "")
                if not answer_text:
                    answer_text = "（回答テキストが取得できませんでした）"

                st.markdown(answer_text)

                # 2. 情報源 (context)
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

                        # 表示用の整形（PDFならページNo.をつける）
                        pretty_path = format_file_info(raw_path, page_num)
                        icon = get_source_icon(raw_path)

                        st.markdown(f"{icon} {pretty_path}")
                        file_info_list.append(pretty_path)

                    message_for_user = "参考として、関連する情報源の候補を表示しました。"

                # 3. 会話ログ用の dict
                content_for_history = {
                    "mode": ct.ANSWER_MODE_2,
                    "answer": answer_text,
                }

                # かつての実装では「マッチしなかった場合（INQUIRY_NO_MATCH_ANSWER）」
                # は message/file_info_list を付けない、という分岐があったので
                # それに近い挙動を保つ
                no_match_answer = getattr(ct, "INQUIRY_NO_MATCH_ANSWER", None)
                if answer_text != no_match_answer:
                    if message_for_user:
                        content_for_history["message"] = message_for_user
                    if file_info_list:
                        content_for_history["file_info_list"] = file_info_list

            # -------------------------
            # (C) 想定外のモード
            # -------------------------
            else:
                fallback_msg = (
                    "現在のモードが不明です。もう一度モードを選択して質問を送信してください。"
                )
                st.markdown(fallback_msg)
                content_for_history = {
                    "mode": "unknown",
                    "answer": fallback_msg,
                }

            # ログ出力（content_for_history は dict or str のはず）
            logger.info({
                "message": content_for_history,
                "application_mode": st.session_state.mode
            })

        except Exception as e:
            # もし描画処理自体で例外が起きた場合でも、
            # ここでユーザーにメッセージを見せておく
            logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")

            fallback_error_msg = utils.build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE)
            st.error(fallback_error_msg, icon=ct.ERROR_ICON)

            # content_for_history が None のままだと後続で困るので補完
            content_for_history = {
                "mode": st.session_state.mode,
                "answer": fallback_error_msg,
            }

    # ==========================================
    # 7-4. 会話ログ（画面再描画用）の更新
    # ==========================================
    # ユーザーの発話
    st.session_state.messages.append({
        "role": "user",
        "content": chat_message,
    })

    # アシスタント側の表示内容
    # （content_for_history は dict のはずだが、
    #   display_search_llm_response() の戻り値次第では文字列の可能性もある。
    #   display_conversation_log() 側が想定済みである前提。）
    st.session_state.messages.append({
        "role": "assistant",
        "content": content_for_history,
    })
