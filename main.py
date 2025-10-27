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
    # 重要：
    # initialize() の中で
    # - session_state初期化
    # - loggerセットアップ
    # - retriever準備（キャッシュ済み）
    # - OPENAI_API_KEYのチェック
    # などを行う
    initialize()
except Exception as e:
    # 初期化に失敗した場合はこれ以上進めないので、
    # エラーログを残してユーザーにも伝えて処理を止める
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
# タイトル・使い方の説明など
cn.display_app_title()

# サイドバーに「利用目的（社内文書検索 / 社内問い合わせ）」などを表示し、
# st.session_state.mode を更新する想定の関数
cn.display_select_mode()

# イントロの案内メッセージ（「こんにちは。私は〜」などの案内やヒント）
cn.display_initial_ai_message()


############################################################
# 5. これまでの会話ログの表示
############################################################
try:
    cn.display_conversation_log()
except Exception as e:
    # 会話ログの描画に失敗しても、ユーザーへのガイドは必要なので
    # ここで止めていい
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
    # 7-1. ユーザーからのメッセージをまず表示
    # ==========================================
    logger.info({"message": chat_message, "application_mode": st.session_state.mode})

    with st.chat_message("user"):
        st.markdown(chat_message)

    # ==========================================
    # 7-2. LLM応答を生成（utils.get_llm_response）
    # ==========================================
    # - utils.get_llm_response() は、例外を投げずに
    #   {"answer": "...", "context": [...]} を返すようにしてある
    # - それでも万一 utils 内で予期せぬ例外が出たら、
    #   ここで握ってユーザーに分かるメッセージを返す
    res_box = st.empty()

    with st.spinner(ct.SPINNER_TEXT):
        try:
            llm_response = utils.get_llm_response(chat_message)
        except Exception as e:
            logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
            # ここではアプリ全体を止めず、簡易的な疑似レスポンスを作って継続する
            llm_response = {
                "answer": utils.build_error_message(ct.GET_LLM_RESPONSE_ERROR_MESSAGE),
                "context": [],
            }

    # ==========================================
    # 7-3. アシスタント側のメッセージを画面に描画
    # ==========================================
    content = ""
    with st.chat_message("assistant"):
        try:
            if st.session_state.mode == ct.ANSWER_MODE_1:
                # 「社内文書検索」モード
                # → 入力と関連性が高い社内文書のありかを提示するUI
                content = cn.display_search_llm_response(llm_response)

            elif st.session_state.mode == ct.ANSWER_MODE_2:
                # 「社内問い合わせ」モード
                # → 社内の制度・社内向け回答を返すUI
                #   （RAGなしのLLM回答 or エラーメッセージ含む）
                content = cn.display_contact_llm_response(llm_response)

            else:
                # mode が想定外の場合も落ちないように保険
                fallback_msg = (
                    "現在のモードが不明です。もう一度モードを選択して質問を送信してください。"
                )
                st.markdown(fallback_msg)
                content = fallback_msg

            # LLM側の回答や、参照ドキュメントのヒントなどをログ出力
            logger.info({"message": content, "application_mode": st.session_state.mode})

        except Exception as e:
            # 表示処理で失敗したときは、最後の砦として人間向けエラーを出す
            logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")
            st.error(utils.build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            # content が未設定だと後続で困るので、適当な代替文を入れる
            content = utils.build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE)

    # ==========================================
    # 7-4. 会話ログ（表示用）に追加
    # ==========================================
    # ユーザーの発話
    st.session_state.messages.append({
        "role": "user",
        "content": chat_message,
    })
    # 今回アシスタントが画面に出した最終テキスト
    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
    })
