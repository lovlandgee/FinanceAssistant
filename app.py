import io
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from gigachat import GigaChat
from sklearn.ensemble import IsolationForest

# Лимиты
MAX_FILE_SIZE_MB = 10
MAX_ROWS = 200_000
ALLOWED_EXTENSIONS = {".csv"}

# Поддерживаемый формат колонок CSV
COLUMN_ALIASES = {
    "date": [
        "date",
        "transaction_date",
        "дата",
        "дата операции",
        "дата платежа",
        "operation date",
        "posted date",
        "booking date",
    ],
    "amount": [
        "amount",
        "sum",
        "value",
        "сумма",
        "операция",
        "сумма операции",
        "transaction amount",
        "debit",
        "credit",
    ],
    "category": [
        "category",
        "merchant_category",
        "категория",
        "тип",
        "mcc",
    ],
    "description": [
        "description",
        "details",
        "назначение",
        "описание",
        "merchant",
        "comment",
        "payment purpose",
    ],
}


@dataclass
class AnalysisResult:
    df: pd.DataFrame
    expenses: pd.DataFrame
    income: pd.DataFrame
    monthly: pd.DataFrame
    category_stats: pd.DataFrame
    anomaly_df: pd.DataFrame
    tips: List[str]


def normalize_col_name(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"\s+"," ", col)
    return col


def find_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    normalized = {normalize_col_name(c): c for c in columns}
    for alias in aliases:
        key = normalize_col_name(alias)
        if key in normalized:
            return normalized[key]
    return None


from typing import cast
def read_csv_with_fallback(content: bytes) -> pd.DataFrame:
    errors: list[Exception] = []
    try:
        return cast(
            pd.DataFrame,
            pd.read_csv(
                io.BytesIO(content),
                nrows=MAX_ROWS,
                sep=None,
                engine="python",
                encoding="utf-8-sig",
            ),
        )
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
        errors.append(e)
    for sep in [",", ";", "\t"]:
        try:
            return cast(
                pd.DataFrame,
                pd.read_csv(
                    io.BytesIO(content),
                    nrows=MAX_ROWS,
                    sep=sep,
                    encoding="utf-8-sig",
                ),
            )
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
            errors.append(e)
    detail = errors[-1] if errors else "неизвестная ошибка"
    raise ValueError(f"Не удалось прочитать CSV. Детали: {detail}") from detail


def validate_and_read_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("Файл не загружен.")

    filename = uploaded_file.name.lower()
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise ValueError("Разрешены только CSV-файлы.")

    file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"Файл слишком большой: {file_size_mb:.2f} MB. Максимум: {MAX_FILE_SIZE_MB} MB.")

    content = uploaded_file.getvalue()
    df = read_csv_with_fallback(content)

    if df.empty:
        raise ValueError("CSV пустой.")

    return df


def parse_amount_flexible(series: pd.Series) -> pd.Series:
    # Поддерживает данные "1 234,56", "1,234.56", "-500", итд.
    s = series.astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
    s = s.str.replace(r"[^\d,.\- ]", "", regex=True)  # оставить digits/sign/separators/spaces
    s = s.str.replace(" ", "", regex=False)

    # Если запятая есть, а точки нет -> запятая - десятичный разделитель
    only_comma_mask = s.str.contains(",", regex=False) & ~s.str.contains(r"\.", regex=True)
    s = s.where(~only_comma_mask, s.str.replace(",", ".", regex=False))

    # Если есть и запятая, и точка - удалить запятую как тысячный разделитель
    both_mask = s.str.contains(",", regex=False) & s.str.contains(r"\.", regex=True)
    s = s.where(~both_mask, s.str.replace(",", "", regex=False))

    return pd.to_numeric(s, errors="coerce")


def standardize_dataframe(
    df: pd.DataFrame,
    date_col_override: Optional[str] = None,
    amount_col_override: Optional[str] = None,
    category_col_override: Optional[str] = None,
    description_col_override: Optional[str] = None,
) -> pd.DataFrame:
    columns = list(df.columns)

    date_col = date_col_override or find_column(columns, COLUMN_ALIASES["date"])
    amount_col = amount_col_override or find_column(columns, COLUMN_ALIASES["amount"])
    category_col = category_col_override if category_col_override else find_column(columns, COLUMN_ALIASES["category"])
    description_col = (
        description_col_override if description_col_override else find_column(columns, COLUMN_ALIASES["description"])
    )

    if date_col is None or amount_col is None:
        available = ", ".join([str(c) for c in columns])
        raise ValueError(
            "Не найдены обязательные колонки. Нужны минимум: дата и сумма.\n"
            "Примеры: date/дата и amount/сумма.\n"
            f"Найденные колонки: {available}"
        )

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    out["amount"] = parse_amount_flexible(df[amount_col])

    if category_col is not None:
        out["category"] = df[category_col].fillna("Unknown").astype(str)
    else:
        out["category"] = "Unknown"

    if description_col is not None:
        out["description"] = df[description_col].fillna("").astype(str)
    else:
        out["description"] = ""

    out = out.dropna(subset=["date", "amount"]).copy()
    if out.empty:
        raise ValueError("После очистки не осталось валидных строк (дата/сумма). Проверь формат даты и суммы.")

    out["month"] = out["date"].dt.to_period("M").astype(str)
    out["abs_amount"] = out["amount"].abs()
    return out


def detect_anomalies(expenses: pd.DataFrame) -> pd.DataFrame:
    if len(expenses) < 20:
        return pd.DataFrame(columns=list(expenses.columns) + ["anomaly"])

    X = expenses[["abs_amount"]].copy()
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    labels = model.fit_predict(X)

    out = expenses.copy()
    out["anomaly"] = labels
    return out[out["anomaly"] == -1].sort_values("abs_amount", ascending=False)


def build_tips(expenses: pd.DataFrame, monthly: pd.DataFrame, category_stats: pd.DataFrame) -> List[str]:
    tips: List[str] = []

    if expenses.empty:
        return ["Расходов не обнаружено — возможно, в файле только доходы."]

    top_cat = category_stats.iloc[0] if not category_stats.empty else None
    if top_cat is not None:
        tips.append(
            f"Наибольшая доля расходов — '{top_cat['category']}' ({top_cat['share_pct']:.1f}%). "
            "Проверьте подписки и повторяющиеся платежи в этой категории."
        )

    if len(monthly) >= 2:
        first = monthly["expense_total"].iloc[0]
        last = monthly["expense_total"].iloc[-1]
        if first > 0:
            growth = (last - first) / first * 100
            if growth > 10:
                tips.append(
                    f"Расходы выросли на {growth:.1f}% относительно начала периода. "
                    "Рекомендуется установить месячный лимит и уведомления при его достижении."
                )
            elif growth < -10:
                tips.append(f"Отлично: расходы снизились на {abs(growth):.1f}% — текущая стратегия работает.")

    p95 = expenses["abs_amount"].quantile(0.95)
    big_spends = expenses[expenses["abs_amount"] >= p95]
    if len(big_spends) > 0:
        tips.append(
            f"Обнаружены крупные траты (>=95%): {len(big_spends)} операций. "
            "Планируйте такие платежи заранее через отдельный резерв."
        )

    small = expenses[expenses["abs_amount"] < expenses["abs_amount"].median() * 0.4]
    if len(small) > len(expenses) * 0.35:
        tips.append("Много мелких покупок. Попробуйте правило '24 часов' для неплановых трат.")

    if not tips:
        tips.append("Финансовая динамика стабильна. Для прогресса можно задать цель экономии 10% в месяц.")

    return tips


def build_llm_payload(result: AnalysisResult) -> dict:
    monthly_small = result.monthly.tail(12).copy()
    category_small = result.category_stats.head(10).copy()

    return {
        "kpi": {
            "months_analyzed": int(result.monthly.shape[0]),
            "transactions_total": int(result.df.shape[0]),
            "expenses_count": int(result.expenses.shape[0]),
            "income_count": int(result.income.shape[0]),
            "total_income": float(result.income["amount"].sum()) if not result.income.empty else 0.0,
            "total_expense": float(result.expenses["expense"].sum()) if not result.expenses.empty else 0.0,
        },
        "monthly": monthly_small.to_dict(orient="records"),
        "top_categories": category_small.to_dict(orient="records"),
        "anomalies_count": int(result.anomaly_df.shape[0]),
        "rule_based_tips": result.tips[:5],
    }


def llm_finance_advice(payload: dict, model: str = "GigaChat") -> str:
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        return (
            "LLM-советы недоступны: не найден `GIGACHAT_CREDENTIALS`.\n"
            "Добавьте ключ в переменные окружения и перезапустите приложение."
        )

    system_prompt = (
        "Ты финансовый ассистент для Personal Finance Manager. "
        "Давай только образовательные рекомендации по личным финансам, "
        "без инвестиционных гарантий и без рискованных советов. "
        "Пиши коротко и структурированно: "
        "1) ключевые выводы, 2) 5 конкретных шагов, 3) что отслеживать в следующем месяце. "
        "Не запрашивай персональные данные."
    )

    user_prompt = (
        "Ниже агрегированная статистика по финансам пользователя в JSON. "
        "Сделай персонализированные рекомендации по оптимизации расходов и привычек.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    verify_ssl = os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "true").lower() in ("1", "true", "yes")

    try:
        with GigaChat(
            credentials=credentials,
            verify_ssl_certs=verify_ssl,
        ) as giga:
            response = giga.chat(
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                }
            )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Ошибка GigaChat: {e}"


def analyze_finances(df: pd.DataFrame) -> AnalysisResult:
    expenses = df[df["amount"] < 0].copy()
    income = df[df["amount"] > 0].copy()

    if not expenses.empty:
        expenses["expense"] = expenses["amount"].abs()

    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            income_total=("amount", lambda s: s[s > 0].sum()),
            expense_total=("amount", lambda s: (-s[s < 0]).sum()),
            tx_count=("amount", "count"),
        )
        .sort_values("month")
    )

    category_stats = pd.DataFrame(columns=["category", "expense_total", "share_pct"])
    if not expenses.empty:
        category_stats = (
            expenses.groupby("category", as_index=False)
            .agg(expense_total=("expense", "sum"))
            .sort_values("expense_total", ascending=False)
        )
        total_exp = category_stats["expense_total"].sum()
        category_stats["share_pct"] = np.where(
            total_exp > 0,
            category_stats["expense_total"] / total_exp * 100,
            0.0,
        )

    anomaly_df = detect_anomalies(expenses)
    tips = build_tips(expenses, monthly, category_stats)

    return AnalysisResult(
        df=df,
        expenses=expenses,
        income=income,
        monthly=monthly,
        category_stats=category_stats,
        anomaly_df=anomaly_df,
        tips=tips,
    )


def main():
    st.set_page_config(page_title="Personal Finance Manager AI", layout="wide")
    st.title("Personal Finance Manager - AI Assistant")
    st.caption("Безопасный анализ CSV-выписок: статистика, паттерны, советы и визуализация")

    with st.sidebar:
        st.header("LLM-ассистент")
        llm_consent = st.checkbox(
            "Согласен отправить обезличенную статистику в GigaChat",
            value=False,
            help="В облако уходят только агрегаты: KPI, помесячные итоги, топ категорий. "
            "Сырые строки выписки и описания операций не отправляются.",
        )
        use_llm = st.checkbox(
            "Включить AI-рекомендации",
            value=False,
            disabled=not llm_consent,
        )
        llm_model = st.selectbox(
            "Модель",
            options=["GigaChat", "GigaChat-2", "GigaChat-2-Max", "GigaChat-2-Pro", "GigaChat-3B-2025-09"],
            disabled=not use_llm,
        )
        if not llm_consent:
            st.caption("Чтобы включить AI, сначала подтвердите согласие на отправку агрегированной статистики.")
        else:
            st.caption("В LLM отправляются только агрегированные данные, без сырых транзакций.")

        st.header("Загрузка данных")
        uploaded_file = st.file_uploader("Загрузите CSV из банка", type=["csv"])
        st.markdown(
            f"""
            **Ограничения безопасности:**
            - только `.csv`
            - размер до `{MAX_FILE_SIZE_MB} MB`
            - до `{MAX_ROWS:,}` строк
            """.replace(",", " ")
        )

    if uploaded_file is None:
        st.info("Загрузите CSV-файл, чтобы начать анализ.")
        return

    try:
        raw_df = validate_and_read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        return

    st.subheader("Сопоставление колонок (если автоопределение не сработало)")
    cols = list(raw_df.columns)
    auto_date = find_column(cols, COLUMN_ALIASES["date"])
    auto_amount = find_column(cols, COLUMN_ALIASES["amount"])
    auto_category = find_column(cols, COLUMN_ALIASES["category"])
    auto_description = find_column(cols, COLUMN_ALIASES["description"])

    c_map1, c_map2 = st.columns(2)
    with c_map1:
        date_col = st.selectbox("Колонка с датой", options=cols, index=cols.index(auto_date) if auto_date in cols else 0)
        amount_col = st.selectbox(
            "Колонка с суммой",
            options=cols,
            index=cols.index(auto_amount) if auto_amount in cols else min(1, len(cols) - 1),
        )
    with c_map2:
        optional_cols = ["(нет)"] + cols
        category_col_ui = st.selectbox(
            "Колонка с категорией (опционально)",
            options=optional_cols,
            index=optional_cols.index(auto_category) if auto_category in optional_cols else 0,
        )
        description_col_ui = st.selectbox(
            "Колонка с описанием (опционально)",
            options=optional_cols,
            index=optional_cols.index(auto_description) if auto_description in optional_cols else 0,
        )

    category_col = None if category_col_ui == "(нет)" else category_col_ui
    description_col = None if description_col_ui == "(нет)" else description_col_ui

    try:
        df = standardize_dataframe(
            raw_df,
            date_col_override=date_col,
            amount_col_override=amount_col,
            category_col_override=category_col,
            description_col_override=description_col,
        )
        result = analyze_finances(df)
    except Exception as e:
        st.error(f"Ошибка обработки файла: {e}")
        return

    total_income = result.income["amount"].sum() if not result.income.empty else 0.0
    total_expense = result.expenses["expense"].sum() if not result.expenses.empty else 0.0
    savings = total_income - total_expense
    savings_rate = (savings / total_income * 100) if total_income > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Доходы", f"{total_income:,.2f}".replace(",", " "))
    c2.metric("Расходы", f"{total_expense:,.2f}".replace(",", " "))
    c3.metric("Баланс", f"{savings:,.2f}".replace(",", " "))
    c4.metric("Норма сбережений", f"{savings_rate:.1f}%")

    st.subheader("Динамика по месяцам")
    if not result.monthly.empty:
        fig_monthly = px.line(
            result.monthly,
            x="month",
            y=["income_total", "expense_total"],
            markers=True,
            title="Доходы и расходы по месяцам",
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Структура расходов по категориям")
        if not result.category_stats.empty:
            fig_pie = px.pie(
                result.category_stats.head(10),
                names="category",
                values="expense_total",
                title="Топ-10 категорий расходов",
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.write("Недостаточно данных по расходам.")
    with right:
        st.subheader("Подозрительные/аномальные траты")
        if not result.anomaly_df.empty:
            show_cols = [c for c in ["date", "category", "description", "abs_amount"] if c in result.anomaly_df.columns]
            st.dataframe(result.anomaly_df[show_cols].head(20), use_container_width=True)
        else:
            st.write("Аномалии не обнаружены или данных пока мало.")

    st.subheader("Персональные советы по экономии")
    for tip in result.tips:
        st.markdown(f"- {tip}")

    if use_llm and llm_consent:
        st.subheader("AI-рекомендации (LLM)")
        payload = build_llm_payload(result)
        with st.spinner("Генерирую рекомендации..."):
            llm_text = llm_finance_advice(payload, model=llm_model)
        st.markdown(llm_text)

    st.subheader("Предпросмотр очищенных данных")
    st.dataframe(result.df.head(100), use_container_width=True)


if __name__ == "__main__":
    main()