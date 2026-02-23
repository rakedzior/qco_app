# app_qco.py
import re
import traceback
from io import BytesIO
from datetime import datetime, date
from typing import Optional, Dict, List, Any

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text, bindparam
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode


# ============================================================
# CONFIG
# ============================================================

FIELD_MAX_LENGTHS = {
    "jira_id": 50,
    "jira_link": 500,
    "project_key": 50,
    "project_name": 200,
    "issue_type": 100,
    "jira_status": 100,
    "jira_status_category": 50,
    "assignee_name": 100,
    "assignee_display": 200,
    "reporter_id": 100,
    "reporter_display": 200,
    "jira_summary": 1000,
    "qc_overall_status": 50,
    "qc_investigation_status": 50,
    "qc_investigator": 100,
    "qc_link": 500,
    "reference_id": 100,
    "project_manager_id": 100,
    "project_manager_name": 200,
    "support_lead_id": 100,
    "support_lead_name": 200,
    "responsible_analyst_id": 100,
    "responsible_analyst_name": 200,
    "region": 50,
    "updated_by": 100,
    # Add others as needed
}


st.set_page_config(page_title="QC Governance", layout="wide")

JIRA_BASE_URL = "https://fcr-jira.systems.uk.hsbc/browse/"

QC_INVESTIGATORS = ["Franek Grzybowski", "Hesam Khaksar", "Rafal Kedzior"]

QC_STATUS_OPTIONS = [
    "Not Reviewed",
    "Fully Conforms",
    "Generally Conforms",
    "Partially Conforms",
    "Does Not Conform",
]
INVESTIGATION_STATUS_OPTIONS = ["Not Started", "In-Progress", "Completed"]
QC_OVERALL_STATUS_OPTIONS = [
    "Fully Conforms",
    "Generally Conforms",
    "Partially Conforms",
    "Does Not Conform",
]
YES_NO_OPTIONS = ["No", "Yes"]

# Performance guards (Python-only optimization, no SQL changes)
MASTER_AUTOFILL_MAX_ROWS = 400

JIRA_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]+-\d+$")


# ============================================================
# THEME / CSS (HSBC + KPI CARDS)
# ============================================================
st.markdown(
    """
    <style>
      :root{
        --hsbc-red:#DB0011;
        --hsbc-red-dark:#B8000E;
        --hsbc-text:#1F1F1F;
        --hsbc-muted:#6B6B6B;
        --hsbc-border:#E6E6E6;
        --hsbc-bg:#FAFAFA;

        /* Pastels */
        --p-green:  #c6efce;
        --p-lgreen: #e2f0d9;
        --p-yellow: #fff2cc;
        --p-red:    #f8cbad;
        --p-gray:   #e7e6e6;
      }

      section.main > div { background: var(--hsbc-bg); }

      /* Primary buttons */
      div.stButton > button[kind="primary"]{
        background: var(--hsbc-red) !important;
        border: 1px solid var(--hsbc-red) !important;
        color: #fff !important;
        border-radius: 10px !important;
      }
      div.stButton > button[kind="primary"]:hover{
        background: var(--hsbc-red-dark) !important;
        border-color: var(--hsbc-red-dark) !important;
      }

      /* Download button */
      div.stDownloadButton > button {
        background: var(--hsbc-red) !important;
        color: white !important;
        border: 1px solid var(--hsbc-red) !important;
        border-radius: 10px !important;
      }
      div.stDownloadButton > button:hover {
        background: var(--hsbc-red-dark) !important;
        border-color: var(--hsbc-red-dark) !important;
      }

      .small-muted { color: var(--hsbc-muted); font-size: 12px; }

      /* KPI wrapper (center) */
      .kpi-wrap{ display:flex; justify-content:center; width:100%; }
      .kpi-row{ width: 880px; max-width: 100%; margin: 0 auto; }

      /* KPI cards (non-clickable) */
      .kpi-card{
        height: 62px;
        border-radius: 10px;
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        text-align:center;
        padding: 6px 8px;
        font-weight: 700;
        color: var(--hsbc-text);
      }
      .kpi-title{ font-size: 12px; line-height: 14px; font-weight: 700; }
      .kpi-value{ font-size: 18px; line-height: 20px; margin-top: 4px; font-weight: 800; }

      .kpi-green  { background: var(--p-green); }
      .kpi-lgreen { background: var(--p-lgreen); }
      .kpi-yellow { background: var(--p-yellow); }
      .kpi-red    { background: var(--p-red); }
      .kpi-gray   { background: var(--p-gray); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DB CONNECTION (SQL Server via pyodbc + SQLAlchemy)
# ============================================================
SQL_SERVER = "gbw25068941.hbeu.adroot.hsbc,10010"
SQL_DATABASE = "aa_global_training_ddl"

# Switched to ODBC Driver 17
SQL_ODBC_DRIVER = "ODBC Driver 17 for SQL Server"

engine = create_engine(
    f"mssql+pyodbc://{SQL_SERVER}/{SQL_DATABASE}"
    f"?trusted_connection=yes&driver={SQL_ODBC_DRIVER.replace(' ', '+')}&AnsiNPW=No",
    fast_executemany=True,
)

DB_SCHEMA = "dbo"
T_MASTER = f"{DB_SCHEMA}.qco_master"
T_CATALOG = f"{DB_SCHEMA}.qco_check_catalog"
T_GRADES = f"{DB_SCHEMA}.qco_check_grades"
T_EMPLOYEE_SOURCE = f"{DB_SCHEMA}.qco_staff"


# ============================================================
# HELPERS
# ============================================================
def jira_link(jira_id: str) -> str:
    return f"{JIRA_BASE_URL}{jira_id}"


def fetch_df(sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def exec_sql(sql: str, params: Optional[Dict] = None) -> int:
    # Truncate all string params
    if params:
        params = truncate_fields(params, FIELD_MAX_LENGTHS)
    with engine.begin() as conn:
        res = conn.execute(text(sql), params or {})
    try:
        return int(res.rowcount or 0)
    except Exception:
        return 0


def current_user() -> str:
    return st.session_state.get("user_name", "unknown")


def safe_date(v) -> Optional[date]:
    if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v.strip() == ""):
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    try:
        return pd.to_datetime(v).date()
    except Exception:
        return None


def to_yes_no(v) -> str:
    if v in [True, 1, "1", "Yes", "YES", "yes", "True", "TRUE", "true"]:
        return "Yes"
    return "No"


def from_yes_no(v: str) -> int:
    return 1 if (v or "").strip().lower() == "yes" else 0


def _norm(s: Any) -> str:
    return ("" if s is None else str(s)).strip()


def clean_none_like(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace({pd.NA: "", None: ""})
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str).replace({"None": ""})
            out.loc[out[c].str.strip().str.lower().eq("nan"), c] = ""
            out.loc[out[c].str.strip().str.lower().eq("none"), c] = ""
    return out


def export_excel_button(df: pd.DataFrame) -> None:
    xls_buffer = BytesIO()
    with pd.ExcelWriter(xls_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="qco_master")

    st.download_button(
        "Export to Excel",
        data=xls_buffer.getvalue(),
        file_name="qco_master_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def flash_success(msg: str) -> None:
    st.session_state["_flash_success"] = msg


def show_flash_messages() -> None:
    msg = st.session_state.pop("_flash_success", None)
    if msg:
        try:
            st.toast(msg, icon="✅")
        except Exception:
            st.success(msg)


def _parse_date_loose(v) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def date_str(v: Optional[date]) -> str:
    return v.isoformat() if isinstance(v, date) else ""


def parse_date_str(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def kpi_card_html(title: str, value: int, color_class: str) -> str:
    return f"""
    <div class="kpi-card {color_class}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """


def _status_to_tbd(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    s = s.replace({"": "TBD"})
    s.loc[s.str.lower().isin(["nan", "none"])] = "TBD"
    return s


def _qc_status_to_tbd(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    s.loc[s.str.lower().isin(["nan", "none"])] = ""
    s = s.replace({"Not Reviewed": "TBD", "": "TBD"})
    return s


def run_sql_safely(fn, context: str) -> bool:
    try:
        fn()
        return True
    except Exception as e:
        st.error(f"SQL operation failed ({context}). Error: {e}")
        with st.expander("Show technical details"):
            st.code(traceback.format_exc())
        return False


@st.cache_data(show_spinner=False, ttl=600)
def has_column(schema: str, table: str, col: str) -> bool:
    try:
        df = fetch_df(
            """
            SELECT 1 AS ok
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :schema
              AND TABLE_NAME = :table
              AND COLUMN_NAME = :col;
            """,
            {"schema": schema, "table": table, "col": col},
        )
        return not df.empty
    except Exception:
        return False


MASTER_SCHEMA = T_MASTER.split(".")[0]
MASTER_TABLE = T_MASTER.split(".")[1]

HAS_JIRA_CREATED_AT = has_column(MASTER_SCHEMA, MASTER_TABLE, "jira_created_at")
HAS_JIRA_UPDATED_AT = has_column(MASTER_SCHEMA, MASTER_TABLE, "jira_updated_at")
HAS_UPDATED_AT = has_column(MASTER_SCHEMA, MASTER_TABLE, "updated_at")
HAS_UPDATED_BY = has_column(MASTER_SCHEMA, MASTER_TABLE, "updated_by")


@st.cache_data(show_spinner=False, ttl=600)
def get_master_columns() -> Set[str]:
    try:
        df = fetch_df(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :schema
              AND TABLE_NAME = :table;
            """,
            {"schema": MASTER_SCHEMA, "table": MASTER_TABLE},
        )
        return {str(x).strip() for x in df["COLUMN_NAME"].tolist()}
    except Exception:
        return set()


def master_has_column(col: str) -> bool:
    return col in get_master_columns()


@st.cache_data(show_spinner=False, ttl=600)
def get_master_char_limits() -> Dict[str, int]:
    try:
        df = fetch_df(
            """
            SELECT COLUMN_NAME, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :schema
              AND TABLE_NAME = :table
              AND CHARACTER_MAXIMUM_LENGTH IS NOT NULL;
            """,
            {"schema": MASTER_SCHEMA, "table": MASTER_TABLE},
        )
        limits: Dict[str, int] = {}
        for _, row in df.iterrows():
            col = str(row.get("COLUMN_NAME") or "").strip()
            max_len = row.get("CHARACTER_MAXIMUM_LENGTH")
            if not col or max_len in (None, -1):
                continue
            try:
                limits[col] = int(max_len)
            except Exception:
                continue
        return limits
    except Exception:
        return {}


def truncate_master_row_to_db_limits(row: Dict[str, Any]) -> Dict[str, Any]:
    limits = get_master_char_limits()
    if not limits:
        return row

    out = dict(row)
    for field, val in out.items():
        if not isinstance(val, str):
            continue
        db_max = limits.get(field)
        cfg_max = FIELD_MAX_LENGTHS.get(field)

        if db_max is None and cfg_max is None:
            continue

        effective_max = db_max if cfg_max is None else min(db_max or cfg_max, cfg_max)
        if effective_max is not None and len(val) > effective_max:
            print(f"Truncating field {field}: {len(val)} -> {effective_max}")
            out[field] = val[:effective_max]

    return out


def get_effective_master_limit(col: str) -> Optional[int]:
    db_max = get_master_char_limits().get(col)
    cfg_max = FIELD_MAX_LENGTHS.get(col)

    if db_max is None and cfg_max is None:
        return None
    if db_max is None:
        return cfg_max
    if cfg_max is None:
        return db_max
    return min(db_max, cfg_max)


def enforce_master_display_limits(df_display: pd.DataFrame, display_to_internal: Dict[str, str]) -> pd.DataFrame:
    out = df_display.copy()
    for display_col, internal_col in display_to_internal.items():
        if display_col not in out.columns:
            continue
        limit = get_effective_master_limit(internal_col)
        if not limit:
            continue

        out[display_col] = out[display_col].apply(
            lambda v: v if (not isinstance(v, str) or len(v) <= limit) else v[:limit]
        )

    return out


def normalize_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str).fillna("").map(lambda x: x.strip())
        if "Date" in c or "date" in c:
            out[c] = out[c].map(lambda x: _parse_date_loose(x).isoformat() if _parse_date_loose(x) else "")
    return out


# ============================================================
# EMPLOYEES MAPS
# ============================================================
@st.cache_data(show_spinner=False, ttl=600)
def get_employees_maps() -> dict:
    queries = [
        # Source of truth for people mapping
        (
            T_EMPLOYEE_SOURCE,
            """
            SELECT
                LTRIM(RTRIM(CAST([employee_name] AS nvarchar(4000)))) AS employee_name,
                LTRIM(RTRIM(CAST([employee_id] AS varchar(100)))) AS employee_id,
                LTRIM(RTRIM(CAST([region] AS nvarchar(4000)))) AS region
            FROM {table}
            WHERE [employee_id] IS NOT NULL
              AND [employee_name] IS NOT NULL;
            """,
        ),
        (
            T_EMPLOYEE_SOURCE,
            """
            SELECT
                LTRIM(RTRIM(CAST([Employee Name] AS nvarchar(4000)))) AS employee_name,
                LTRIM(RTRIM(CAST([Employee ID] AS varchar(100)))) AS employee_id,
                LTRIM(RTRIM(CAST([Region] AS nvarchar(4000)))) AS region
            FROM {table}
            WHERE [Employee ID] IS NOT NULL
              AND [Employee Name] IS NOT NULL;
            """,
        ),
    ]

    last_err = None
    df = pd.DataFrame(columns=["employee_name", "employee_id", "region"])
    for table, sql_tpl in queries:
        try:
            df = fetch_df(sql_tpl.format(table=table))
            if not df.empty:
                break
        except Exception as e:
            last_err = e
            continue

    if df.empty and last_err is not None:
        print(f"[WARN] Could not load employee maps: {last_err}")

    if df.empty:
        return {"name_to_id": {}, "id_to_name": {}, "id_to_region": {}, "names_sorted": []}

    df["employee_name"] = df["employee_name"].astype(str).str.strip()
    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    df["region"] = df["region"].astype(str).str.strip()

    df = df[(df["employee_name"] != "") & (df["employee_id"] != "")]

    name_to_id = {n.lower(): i for n, i in zip(df["employee_name"], df["employee_id"])}
    id_to_name = {i: n for n, i in zip(df["employee_name"], df["employee_id"])}
    id_to_region = {i: r for i, r in zip(df["employee_id"], df["region"])}

    names_sorted = sorted(df["employee_name"].dropna().unique().tolist(), key=lambda x: x.lower())

    return {
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
        "id_to_region": id_to_region,
        "names_sorted": names_sorted,
    }


def _df_equal_loose(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    aa = a.fillna("").astype(str)
    bb = b.fillna("").astype(str)
    return aa.equals(bb)


def apply_master_autofill(df_display: pd.DataFrame, maps: dict) -> pd.DataFrame:
    out = df_display.copy()
    for i, r in out.iterrows():
        row = {
            "project_manager_id": _norm(r.get("Project Manager ID", "")) or None,
            "project_manager_name": _norm(r.get("Project Manager Name", "")) or None,
            "support_lead_id": _norm(r.get("Support Lead ID", "")) or None,
            "support_lead_name": _norm(r.get("Support Lead Name", "")) or None,
            "responsible_analyst_id": _norm(r.get("Responsible Analyst ID", "")) or None,
            "responsible_analyst_name": _norm(r.get("Responsible Analyst Name", "")) or None,
            "region": _norm(r.get("Region", "")) or None,
        }
        row = sync_person_row_fields(row, maps)

        out.at[i, "Project Manager ID"] = row.get("project_manager_id") or ""
        out.at[i, "Project Manager Name"] = row.get("project_manager_name") or ""
        out.at[i, "Support Lead ID"] = row.get("support_lead_id") or ""
        out.at[i, "Support Lead Name"] = row.get("support_lead_name") or ""
        out.at[i, "Responsible Analyst ID"] = row.get("responsible_analyst_id") or ""
        out.at[i, "Responsible Analyst Name"] = row.get("responsible_analyst_name") or ""
        out.at[i, "Region"] = row.get("region") or ""

    return out


def sync_person_fields(prefix: str, maps: dict, also_region_from_support_lead: bool = False) -> None:
    name_to_id = maps["name_to_id"]
    id_to_name = maps["id_to_name"]
    id_to_region = maps["id_to_region"]

    id_key = f"{prefix}_id"
    name_key = f"{prefix}_name"

    emp_id = _norm(st.session_state.get(id_key, ""))
    emp_name = _norm(st.session_state.get(name_key, ""))

    if emp_id:
        if emp_id in id_to_name:
            st.session_state[name_key] = id_to_name[emp_id]
            if also_region_from_support_lead:
                st.session_state["region"] = id_to_region.get(emp_id, "") or ""
        else:
            if also_region_from_support_lead:
                st.session_state["region"] = ""
    elif emp_name:
        k = emp_name.lower()
        if k in name_to_id:
            mapped_id = name_to_id[k]
            st.session_state[id_key] = mapped_id
            if also_region_from_support_lead:
                st.session_state["region"] = id_to_region.get(mapped_id, "") or ""
        else:
            if also_region_from_support_lead:
                st.session_state["region"] = ""
    else:
        st.session_state[id_key] = ""
        st.session_state[name_key] = ""
        if also_region_from_support_lead:
            st.session_state["region"] = ""


def sync_person_row_fields(row: Dict[str, Any], maps: dict) -> Dict[str, Any]:
    out = dict(row)
    name_to_id = maps["name_to_id"]
    id_to_name = maps["id_to_name"]
    id_to_region = maps["id_to_region"]

    def _sync_pair(id_key: str, name_key: str) -> None:
        emp_id = _norm(out.get(id_key, ""))
        emp_name = _norm(out.get(name_key, ""))

        if emp_id:
            if emp_id in id_to_name:
                out[name_key] = id_to_name[emp_id]
            return

        if emp_name:
            mapped_id = name_to_id.get(emp_name.lower())
            if mapped_id:
                out[id_key] = mapped_id

    _sync_pair("project_manager_id", "project_manager_name")
    _sync_pair("support_lead_id", "support_lead_name")
    _sync_pair("responsible_analyst_id", "responsible_analyst_name")

    support_lead_id = _norm(out.get("support_lead_id", ""))
    if support_lead_id:
        out["region"] = id_to_region.get(support_lead_id) or None
    else:
        support_lead_name = _norm(out.get("support_lead_name", ""))
        mapped_id = name_to_id.get(support_lead_name.lower()) if support_lead_name else None
        if mapped_id:
            out["support_lead_id"] = mapped_id
            out["region"] = id_to_region.get(mapped_id) or None
        elif not support_lead_name:
            out["region"] = None

    return out


# ============================================================
# AGGRID JS
# ============================================================
QC_CELLSTYLE_JS = JsCode(
    """
function(params) {
  const v = params.value;
  if (v === 'Fully Conforms') return {backgroundColor: '#c6efce'};
  if (v === 'Generally Conforms') return {backgroundColor: '#e2f0d9'};
  if (v === 'Partially Conforms') return {backgroundColor: '#fff2cc'};
  if (v === 'Does Not Conform') return {backgroundColor: '#f8cbad'};
  if (v === 'Not Reviewed') return {backgroundColor: '#e7e6e6'};
  return {};
}
"""
)

INV_CELLSTYLE_JS = JsCode(
    """
function(params) {
  const v = params.value;
  if (v === 'Completed') return {backgroundColor: '#c6efce'};
  if (v === 'In-Progress') return {backgroundColor: '#fff2cc'};
  if (v === 'Not Started') return {backgroundColor: '#f8cbad'};
  return {};
}
"""
)

LINK_RENDERER_JS = JsCode(
    """
class LinkRenderer {
  init(params) {
    this.eGui = document.createElement('a');
    if (params.value) {
      this.eGui.href = params.value;
      this.eGui.innerText = params.value;
      this.eGui.target = '_blank';
      this.eGui.rel = 'noopener noreferrer';
      this.eGui.style.color = '#1a73e8';
      this.eGui.style.textDecoration = 'underline';
    } else {
      this.eGui.innerText = '';
    }
  }
  getGui() { return this.eGui; }
}
"""
)

JIRA_ID_RENDERER_JS = JsCode(
    """
class JiraIdRenderer {
  init(params) {
    this.eGui = document.createElement('a');
    const jiraId = (params.value || '').toString();
    if (jiraId) {
      const targetUrl = `${window.location.pathname}?page=detail&jira_id=${encodeURIComponent(jiraId)}`;
      this.eGui.href = targetUrl;
      this.eGui.innerText = jiraId;
      this.eGui.style.color = '#1a73e8';
      this.eGui.style.textDecoration = 'underline';
    } else {
      this.eGui.innerText = '';
    }
  }
  getGui() { return this.eGui; }
}
"""
)

AUTOSIZE_COLUMNS_JS = JsCode(
    """
function(params) {
  const allCols = [];
  params.columnApi.getColumns().forEach((col) => allCols.push(col.getId()));
  if (allCols.length > 0) {
    params.columnApi.autoSizeColumns(allCols, false);
  }
}
"""
)


# ============================================================
# PRE-POPULATE GRADES
# ============================================================
def ensure_grades_rows_for_jira(jira_id: str) -> None:
    exec_sql(
        f"""
        INSERT INTO {T_GRADES} (jira_id, check_id, updated_at, updated_by)
        SELECT
            :jira_id,
            c.check_id,
            SYSUTCDATETIME(),
            :updated_by
        FROM {T_CATALOG} c
        LEFT JOIN {T_GRADES} g
          ON g.jira_id = :jira_id
         AND g.check_id = c.check_id
        WHERE c.is_active = 1
          AND g.check_id IS NULL;
    """,
        {"jira_id": jira_id, "updated_by": current_user()},
    )


# ============================================================
# MASTER CRUD
# ============================================================
MASTER_SELECT_COLUMNS = """
    jira_id,
    jira_summary,
    qc_overall_status,
    qc_investigation_status,
    qc_investigator,
    reference_id,
    reference_date,
    pre_check_date,
    email_after_pre_check,
    check_meeting_date,
    email_after_check_meeting,
    remediation_meeting_date,
    project_manager_id,
    project_manager_name,
    support_lead_id,
    support_lead_name,
    responsible_analyst_id,
    responsible_analyst_name,
    region,
    completed_date,
    summary_of_findings,
    check_meeting_comments,
    remediation_meeting_comments
"""
if HAS_JIRA_CREATED_AT:
    MASTER_SELECT_COLUMNS += ", jira_created_at"
if HAS_JIRA_UPDATED_AT:
    MASTER_SELECT_COLUMNS += ", jira_updated_at"
if HAS_UPDATED_AT:
    MASTER_SELECT_COLUMNS += ", updated_at"
if HAS_UPDATED_BY:
    MASTER_SELECT_COLUMNS += ", updated_by"


def get_master_df() -> pd.DataFrame:
    df = fetch_df(f"SELECT {MASTER_SELECT_COLUMNS} FROM {T_MASTER}")

    for c in ["qc_investigator", "qc_investigation_status", "qc_overall_status"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()
            df.loc[df[c].str.lower().isin(["nan", "none"]), c] = ""

    df["qc_investigator_norm"] = df["qc_investigator"].astype(str).str.strip()
    df["qc_investigation_status_norm"] = df["qc_investigation_status"].astype(str).str.strip()
    df["qc_overall_status_norm"] = df["qc_overall_status"].astype(str).str.strip()

    df.insert(1, "jira_link", df["jira_id"].apply(jira_link))

    if "email_after_pre_check" in df.columns:
        df["email_after_pre_check"] = df["email_after_pre_check"].apply(to_yes_no)
    if "email_after_check_meeting" in df.columns:
        df["email_after_check_meeting"] = df["email_after_check_meeting"].apply(to_yes_no)

    return df


def update_master_fields_many(payload_rows: List[dict]) -> None:
    payload_rows = [truncate_master_row_to_db_limits(truncate_fields(row, FIELD_MAX_LENGTHS)) for row in payload_rows]

    candidate_cols = [
        "jira_summary",
        "qc_overall_status",
        "qc_investigation_status",
        "qc_investigator",
        "pre_check_date",
        "check_meeting_date",
        "remediation_meeting_date",
        "reference_id",
        "reference_date",
        "email_after_pre_check",
        "email_after_check_meeting",
        "project_manager_id",
        "project_manager_name",
        "support_lead_id",
        "support_lead_name",
        "responsible_analyst_id",
        "responsible_analyst_name",
        "region",
        "completed_date",
        "summary_of_findings",
        "check_meeting_comments",
        "remediation_meeting_comments",
    ]

    set_clauses: List[str] = []
    for col in candidate_cols:
        if not master_has_column(col):
            continue
        if col == "jira_summary":
            set_clauses.append("jira_summary = COALESCE(:jira_summary, jira_summary)")
        else:
            set_clauses.append(f"{col} = :{col}")

    if master_has_column("updated_at"):
        set_clauses.append("updated_at = SYSUTCDATETIME()")
    if master_has_column("updated_by"):
        set_clauses.append("updated_by = :updated_by")

    if not set_clauses:
        return

    sql = f"""
        UPDATE {T_MASTER}
           SET {", ".join(set_clauses)}
         WHERE jira_id = :jira_id;
    """

    with engine.begin() as conn:
        stmt = text(sql)
        for row in payload_rows:
            conn.execute(stmt, truncate_master_row_to_db_limits({**row, "updated_by": current_user()}))

def truncate_fields(row, field_max_lengths):
    for field, max_len in field_max_lengths.items():
        if field in row and isinstance(row[field], str):
            if len(row[field]) > max_len:
                print(f"Truncating field {field}: {len(row[field])} -> {max_len}")
                row[field] = row[field][:max_len]
    return row

def update_master_partial(jira_id: str, fields: Dict[str, Any]) -> None:
    """
    Partial update used by Detail view callbacks (auto-sync between ID/Name/Region etc.).
    Updates only provided keys (from the allowed set), plus updated_at/updated_by when present.
    """
    allowed = {
        "qc_overall_status",
        "qc_investigation_status",
        "qc_investigator",
        "reference_id",
        "reference_date",
        "pre_check_date",
        "check_meeting_date",
        "remediation_meeting_date",
        "completed_date",
        "email_after_pre_check",
        "email_after_check_meeting",
        "project_manager_id",
        "project_manager_name",
        "support_lead_id",
        "support_lead_name",
        "responsible_analyst_id",
        "responsible_analyst_name",
        "region",
        "summary_of_findings",
        "check_meeting_comments",
        "remediation_meeting_comments",
    }

    fields = truncate_master_row_to_db_limits(truncate_fields(fields, FIELD_MAX_LENGTHS))
    cols: List[str] = []
    params = truncate_master_row_to_db_limits({"jira_id": jira_id, "updated_by": current_user()})

    for k, v in fields.items():
        if k not in allowed or not master_has_column(k):
            continue
        cols.append(f"{k} = :{k}")
        params[k] = v

    if master_has_column("updated_at"):
        cols.append("updated_at = SYSUTCDATETIME()")
    if master_has_column("updated_by"):
        cols.append("updated_by = :updated_by")

    if not cols:
        return

    sql = f"""
        UPDATE {T_MASTER}
           SET {", ".join(cols)}
         WHERE jira_id = :jira_id;
    """
    exec_sql(sql, params)


# ============================================================
# GRADES CRUD (UPSERT without MERGE)
# ============================================================
def upsert_grade_row(
    jira_id: str,
    check_id: str,
    status_pre: Optional[str],
    notes_pre: Optional[str],
    status_post: Optional[str],
    notes_post: Optional[str],
) -> None:
    exec_sql(
        f"""
        UPDATE {T_GRADES}
           SET
               status_pre = :status_pre,
               notes_pre = :notes_pre,
               status_post = :status_post,
               notes_post = :notes_post,
               updated_at = SYSUTCDATETIME(),
               updated_by = :updated_by
         WHERE jira_id = :jira_id
           AND check_id = :check_id;

        IF @@ROWCOUNT = 0
        BEGIN
            INSERT INTO {T_GRADES} (
                jira_id, check_id,
                status_pre, notes_pre,
                status_post, notes_post,
                updated_at, updated_by
            )
            VALUES (
                :jira_id, :check_id,
                :status_pre, :notes_pre,
                :status_post, :notes_post,
                SYSUTCDATETIME(), :updated_by
            );
        END
    """,
        {
            "jira_id": jira_id,
            "check_id": check_id,
            "status_pre": status_pre,
            "notes_pre": notes_pre,
            "status_post": status_post,
            "notes_post": notes_post,
            "updated_by": current_user(),
        },
    )


# ============================================================
# MASTER FILTER STATE + APPLY
# ============================================================
def _init_master_filters() -> None:
    st.session_state.setdefault("flt_investigator", "All")
    st.session_state.setdefault("flt_inv_status", "All")
    st.session_state.setdefault("flt_completed_overall", "All")
    st.session_state.setdefault("flt_search", "")


def _apply_master_filters(df: pd.DataFrame) -> pd.DataFrame:
    df_f = df.copy()

    inv = st.session_state.get("flt_investigator", "All")
    invs = st.session_state.get("flt_inv_status", "All")
    comp_overall = st.session_state.get("flt_completed_overall", "All")
    q = st.session_state.get("flt_search", "")

    if inv != "All":
        df_f = df_f[df_f["qc_investigator_norm"] == inv]

    if invs != "All":
        df_f = df_f[df_f["qc_investigation_status_norm"] == invs]

    if invs == "Completed" and comp_overall != "All":
        if comp_overall == "TBD":
            s = df_f["qc_overall_status_norm"].astype(str).str.strip()
            df_f = df_f[(s == "") | s.str.lower().isin(["nan", "none"])]
        else:
            df_f = df_f[df_f["qc_overall_status_norm"] == comp_overall]

    if q and q.strip():
        qq = q.strip().lower()
        mask = (
            df_f["jira_id"].astype(str).str.lower().str.contains(qq, na=False)
            | df_f["jira_summary"].astype(str).str.lower().str.contains(qq, na=False)
            | df_f["project_manager_name"].astype(str).str.lower().str.contains(qq, na=False)
            | df_f["support_lead_name"].astype(str).str.lower().str.contains(qq, na=False)
            | df_f["responsible_analyst_name"].astype(str).str.lower().str.contains(qq, na=False)
            | df_f["region"].astype(str).str.lower().str.contains(qq, na=False)
        )
        df_f = df_f[mask]

    return df_f


# ============================================================
# MASTER GRID
# ============================================================
def build_master_grid(df_display: pd.DataFrame, display_to_internal: Dict[str, str], name_options: List[str]) -> pd.DataFrame:
    gb = GridOptionsBuilder.from_dataframe(df_display)

    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        editable=False,
        wrapText=False,
        autoHeight=False,
    )

    gb.configure_column("JIRA ID", cellRenderer="JiraIdRenderer", editable=False)
    gb.configure_column("JIRA Link", cellRenderer="LinkRenderer", editable=False)

    if "QC Overall Status" in df_display.columns:
        gb.configure_column(
            "QC Overall Status",
            editable=True,
            cellEditor="agSelectCellEditor",
            cellEditorParams={"values": [""] + QC_OVERALL_STATUS_OPTIONS},
            cellStyle=QC_CELLSTYLE_JS,
        )
    if "QC Investigation Status" in df_display.columns:
        gb.configure_column(
            "QC Investigation Status",
            editable=True,
            cellEditor="agSelectCellEditor",
            cellEditorParams={"values": [""] + INVESTIGATION_STATUS_OPTIONS},
            cellStyle=INV_CELLSTYLE_JS,
        )
    if "Email After Pre-Check" in df_display.columns:
        gb.configure_column(
            "Email After Pre-Check",
            editable=True,
            cellEditor="agSelectCellEditor",
            cellEditorParams={"values": YES_NO_OPTIONS},
        )
    if "Email After Check Meeting" in df_display.columns:
        gb.configure_column(
            "Email After Check Meeting",
            editable=True,
            cellEditor="agSelectCellEditor",
            cellEditorParams={"values": YES_NO_OPTIONS},
        )

    # NOTE: JIRA Title remains read-only (avoid accidental wiping)
    editable_text_cols = [
        "QC Investigator",
        "Reference ID",
    ]
    for c in editable_text_cols:
        if c in df_display.columns:
            internal_col = display_to_internal.get(c)
            max_len = get_effective_master_limit(internal_col) if internal_col else None
            col_cfg = {"editable": True, "cellEditor": "agTextCellEditor"}
            if max_len:
                col_cfg["cellEditorParams"] = {"maxLength": int(max_len)}
            gb.configure_column(c, **col_cfg)

    for c in ["Project Manager ID", "Support Lead ID", "Responsible Analyst ID", "Region"]:
        if c in df_display.columns:
            gb.configure_column(c, editable=False)

    for c in ["Project Manager Name", "Support Lead Name", "Responsible Analyst Name"]:
        if c in df_display.columns:
            gb.configure_column(
                c,
                editable=True,
                cellEditor="agSelectCellEditor",
                cellEditorParams={"values": [""] + name_options},
            )

    editable_date_cols = [
        "Reference Date",
        "Pre-Check Date",
        "Check Meeting Date",
        "Remediation Meeting Date",
        "Completed Date",
    ]
    for c in editable_date_cols:
        if c in df_display.columns:
            gb.configure_column(c, editable=True)

    grid_options = gb.build()
    grid_options["components"] = {"LinkRenderer": LINK_RENDERER_JS, "JiraIdRenderer": JIRA_ID_RENDERER_JS}
    grid_options["headerHeight"] = 64
    grid_options["groupHeaderHeight"] = 64
    grid_options["onFirstDataRendered"] = AUTOSIZE_COLUMNS_JS
    grid_options["suppressSizeToFit"] = True

    resp = AgGrid(
        df_display,
        gridOptions=grid_options,
        height=560,
        fit_columns_on_grid_load=False,
        update_mode=GridUpdateMode.MANUAL,
        data_return_mode=DataReturnMode.AS_INPUT,
        allow_unsafe_jscode=True,
        theme="balham",
    )

    return pd.DataFrame(resp["data"])

# ============================================================
# MASTER VIEW
# ============================================================
def show_master_view() -> None:
    st.title("QC Governance — Master Sheet")
    show_flash_messages()
    _init_master_filters()

    df_all = get_master_df()

    with st.sidebar:
        st.header("Session")
        default_val = st.session_state.get("user_name", QC_INVESTIGATORS[-1])
        if default_val not in QC_INVESTIGATORS:
            default_val = QC_INVESTIGATORS[-1]
        picked = st.selectbox("QC Investigator", QC_INVESTIGATORS, index=QC_INVESTIGATORS.index(default_val))
        st.session_state["user_name"] = picked
        st.divider()
        st.header("Export")
        export_excel_button(df_all)

    inv_list = ["All"] + sorted([x for x in df_all["qc_investigator_norm"].dropna().unique().tolist() if str(x).strip()])
    status_list = ["All"] + sorted(
        [x for x in df_all["qc_investigation_status_norm"].dropna().unique().tolist() if str(x).strip()]
    )

    if st.session_state.get("flt_investigator", "All") not in inv_list:
        st.session_state["flt_investigator"] = "All"
    if st.session_state.get("flt_inv_status", "All") not in status_list:
        st.session_state["flt_inv_status"] = "All"
    valid_completed_overall = ["All"] + QC_OVERALL_STATUS_OPTIONS + ["TBD"]
    if st.session_state.get("flt_completed_overall", "All") not in valid_completed_overall:
        st.session_state["flt_completed_overall"] = "All"

    bar1, bar2, bar3, bar4 = st.columns([1.2, 1.2, 2.2, 0.9])
    with bar1:
        st.selectbox("QC Investigator", inv_list, key="flt_investigator")
    with bar2:
        st.selectbox("Investigation Status", status_list, key="flt_inv_status")
    with bar3:
        st.text_input("Search", key="flt_search", placeholder="JIRA ID / Title / People / Region")
    with bar4:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Clear", use_container_width=True):
            st.session_state["flt_investigator"] = "All"
            st.session_state["flt_inv_status"] = "All"
            st.session_state["flt_completed_overall"] = "All"
            st.session_state["flt_search"] = ""
            st.rerun()

    st.divider()

    df_f = _apply_master_filters(df_all)

    st.subheader("Overview")

    total_cnt = int(len(df_f))
    inv_counts = _status_to_tbd(df_f["qc_investigation_status_norm"]).value_counts().to_dict()

    st.markdown(
        '<div class="small-muted" style="text-align:center; margin-top:8px; font-size:16px;">QC Investigation Status - All tickets</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="kpi-wrap"><div class="kpi-row">', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_card_html("All issues", total_cnt, "kpi-gray"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card_html("Completed", int(inv_counts.get("Completed", 0)), "kpi-green"), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card_html("In-Progress", int(inv_counts.get("In-Progress", 0)), "kpi-yellow"), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card_html("Not Started", int(inv_counts.get("Not Started", 0)), "kpi-red"), unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="small-muted" style="text-align:center; margin-top:8px; font-size:16px;"> QC Overall Status - Completed tickets only</div>',
        unsafe_allow_html=True,
    )

    df_completed = df_f[df_f["qc_investigation_status_norm"] == "Completed"].copy()
    s = df_completed["qc_overall_status_norm"].fillna("").astype(str).str.strip()
    s.loc[s.str.lower().isin(["nan", "none"])] = ""
    s = s[s != ""]
    overall_counts = s.value_counts().to_dict()

    st.markdown('<div class="kpi-wrap"><div class="kpi-row">', unsafe_allow_html=True)
    o1, o2, o3, o4 = st.columns(4)
    with o1:
        st.markdown(kpi_card_html("Fully", int(overall_counts.get("Fully Conforms", 0)), "kpi-green"), unsafe_allow_html=True)
    with o2:
        st.markdown(kpi_card_html("Generally", int(overall_counts.get("Generally Conforms", 0)), "kpi-lgreen"), unsafe_allow_html=True)
    with o3:
        st.markdown(kpi_card_html("Partially", int(overall_counts.get("Partially Conforms", 0)), "kpi-yellow"), unsafe_allow_html=True)
    with o4:
        st.markdown(kpi_card_html("Does Not", int(overall_counts.get("Does Not Conform", 0)), "kpi-red"), unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.divider()

    csel1, csel2 = st.columns([2, 1])
    with csel1:
        pick = st.selectbox("Open QC Details for JIRA ID", options=[""] + df_f["jira_id"].tolist())
    with csel2:
        st.write("")
        st.write("")
        if st.button("Open Details", disabled=(pick == ""), type="primary", use_container_width=True):
            st.query_params.update({"page": "detail", "jira_id": pick})
            st.rerun()

    ordered_internal = [
        "jira_id",
        "jira_link",
        "jira_summary",
        "qc_overall_status",
        "qc_investigation_status",
        "qc_investigator",
        "reference_id",
        "reference_date",
        "pre_check_date",
        "email_after_pre_check",
        "check_meeting_date",
        "email_after_check_meeting",
        "remediation_meeting_date",
        "completed_date",
        "project_manager_id",
        "project_manager_name",
        "support_lead_id",
        "support_lead_name",
        "responsible_analyst_id",
        "responsible_analyst_name",
        "region",
    ]
    if HAS_JIRA_CREATED_AT:
        ordered_internal.append("jira_created_at")
    if HAS_JIRA_UPDATED_AT:
        ordered_internal.append("jira_updated_at")

    df_show = df_f.copy()
    for col in ordered_internal:
        if col not in df_show.columns:
            df_show[col] = ""
    df_show = df_show[ordered_internal]

    rename_map = {
        "jira_id": "JIRA ID",
        "jira_link": "JIRA Link",
        "jira_summary": "JIRA Title",
        "qc_overall_status": "QC Overall Status",
        "qc_investigation_status": "QC Investigation Status",
        "qc_investigator": "QC Investigator",
        "reference_id": "Reference ID",
        "reference_date": "Reference Date",
        "pre_check_date": "Pre-Check Date",
        "email_after_pre_check": "Email After Pre-Check",
        "check_meeting_date": "Check Meeting Date",
        "email_after_check_meeting": "Email After Check Meeting",
        "remediation_meeting_date": "Remediation Meeting Date",
        "completed_date": "Completed Date",
        "project_manager_id": "Project Manager ID",
        "project_manager_name": "Project Manager Name",
        "support_lead_id": "Support Lead ID",
        "support_lead_name": "Support Lead Name",
        "responsible_analyst_id": "Responsible Analyst ID",
        "responsible_analyst_name": "Responsible Analyst Name",
        "region": "Region",
    }
    if HAS_JIRA_CREATED_AT:
        rename_map["jira_created_at"] = "JIRA Created At"
    if HAS_JIRA_UPDATED_AT:
        rename_map["jira_updated_at"] = "JIRA Updated At"

    df_show = df_show.rename(columns=rename_map)
    df_show = clean_none_like(df_show)

    display_to_internal = {v: k for k, v in rename_map.items()}
    df_show = enforce_master_display_limits(df_show, display_to_internal)

    st.subheader("JIRA Master List")

    grid_state_key = "master_grid_working"
    grid_key_state = "master_grid_view_key"
    current_view_key = "|".join(df_show["JIRA ID"].astype(str).tolist())

    if st.session_state.get(grid_key_state) != current_view_key:
        st.session_state[grid_state_key] = df_show.copy(deep=True)
        st.session_state[grid_key_state] = current_view_key

    grid_input = st.session_state.get(grid_state_key, df_show.copy(deep=True))
    if not isinstance(grid_input, pd.DataFrame):
        grid_input = df_show.copy(deep=True)

    maps = get_employees_maps()

    df_before_edit = grid_input.copy(deep=True)
    edited_df = build_master_grid(grid_input, display_to_internal, maps.get("names_sorted", []))

    if len(edited_df) <= MASTER_AUTOFILL_MAX_ROWS:
        edited_autofilled = apply_master_autofill(edited_df, maps)
    else:
        edited_autofilled = edited_df
        st.caption(f"Autosync ID/Region in grid preview is limited to first {MASTER_AUTOFILL_MAX_ROWS} rows for performance; Save still syncs all changed rows.")

    edited_sanitized = enforce_master_display_limits(edited_autofilled, display_to_internal)

    if not _df_equal_loose(edited_sanitized, edited_df):
        st.session_state[grid_state_key] = edited_sanitized
        st.session_state[grid_key_state] = current_view_key
        st.info("People fields were auto-synced (ID/Name/Region) and overlong text was truncated.")

    edited_df = edited_sanitized
    st.session_state[grid_state_key] = edited_df.copy(deep=True)

    csave1, _ = st.columns([1, 6])
    with csave1:
        if st.button("Save", type="primary", use_container_width=True):
            base_df = normalize_for_compare(df_before_edit.copy(deep=True))
            edited_n = normalize_for_compare(edited_df.copy())

            edited_n = edited_n[base_df.columns]

            changed_mask = ~(base_df.eq(edited_n)).all(axis=1)
            changed = edited_df.loc[changed_mask].copy()

            if changed.empty:
                flash_success("No changes to save.")
                st.rerun()

            maps = get_employees_maps()

            payload_rows: List[dict] = []
            for _, r in changed.iterrows():
                jira_id = _norm(r.get("JIRA ID", ""))
                if not jira_id:
                    continue
                row = {
                    "jira_id": jira_id,
                    "jira_summary": None,  # keep title read-only
                    "qc_overall_status": _norm(r.get("QC Overall Status", "")) or None,
                    "qc_investigation_status": _norm(r.get("QC Investigation Status", "")) or None,
                    "qc_investigator": _norm(r.get("QC Investigator", "")) or None,
                    "reference_id": _norm(r.get("Reference ID", "")) or None,
                    "reference_date": _parse_date_loose(r.get("Reference Date", None)),
                    "pre_check_date": _parse_date_loose(r.get("Pre-Check Date", None)),
                    "check_meeting_date": _parse_date_loose(r.get("Check Meeting Date", None)),
                    "remediation_meeting_date": _parse_date_loose(r.get("Remediation Meeting Date", None)),
                    "completed_date": _parse_date_loose(r.get("Completed Date", None)),
                    "email_after_pre_check": from_yes_no(_norm(r.get("Email After Pre-Check", "No")) or "No"),
                    "email_after_check_meeting": from_yes_no(_norm(r.get("Email After Check Meeting", "No")) or "No"),
                    "project_manager_id": _norm(r.get("Project Manager ID", "")) or None,
                    "project_manager_name": _norm(r.get("Project Manager Name", "")) or None,
                    "support_lead_id": _norm(r.get("Support Lead ID", "")) or None,
                    "support_lead_name": _norm(r.get("Support Lead Name", "")) or None,
                    "responsible_analyst_id": _norm(r.get("Responsible Analyst ID", "")) or None,
                    "responsible_analyst_name": _norm(r.get("Responsible Analyst Name", "")) or None,
                    "region": _norm(r.get("Region", "")) or None,
                    "summary_of_findings": None,
                    "check_meeting_comments": None,
                    "remediation_meeting_comments": None,
                }

                # Autofill ID/Name/Region in master save path (mirrors detail-view behavior)
                row = sync_person_row_fields(row, maps)

                # Truncate fields before appending
                row = truncate_fields(row, FIELD_MAX_LENGTHS)
                payload_rows.append(row)

            if not payload_rows:
                flash_success("No valid rows to save.")
                st.rerun()

            # Preserve long-text fields on bulk save (only if columns exist)
            jira_ids = [p["jira_id"] for p in payload_rows]
            long_text_cols = [
                c
                for c in ["summary_of_findings", "check_meeting_comments", "remediation_meeting_comments"]
                if master_has_column(c)
            ]

            if long_text_cols:
                select_cols = ", ".join(["jira_id"] + long_text_cols)
                stmt = (
                    text(
                        f"""
                        SELECT {select_cols}
                        FROM {T_MASTER}
                        WHERE jira_id IN :jira_ids
                        """
                    )
                    .bindparams(bindparam("jira_ids", expanding=True))
                )

                with engine.begin() as conn:
                    existing = pd.read_sql(stmt, conn, params={"jira_ids": jira_ids})

                existing_map = {
                    _norm(row["jira_id"]): {col: row.get(col) for col in long_text_cols}
                    for _, row in existing.iterrows()
                }

                for p in payload_rows:
                    keep = existing_map.get(p["jira_id"], {})
                    for col in long_text_cols:
                        p[col] = keep.get(col)

            payload_rows = [truncate_fields(row, FIELD_MAX_LENGTHS) for row in payload_rows]

            for row in payload_rows:
                for field, max_len in FIELD_MAX_LENGTHS.items():
                    if field in row and isinstance(row[field], str) and len(row[field]) > max_len:
                        print(f"Field {field} too long: {len(row[field])} chars (max {max_len})")
                # No need to truncate again, row is already truncated before appending

            ok = run_sql_safely(lambda: update_master_fields_many(payload_rows), "master/update_master_fields_many")
            if not ok:
                return

            flash_success(f"Saved {len(payload_rows)} row(s).")
            st.rerun()


# ============================================================
# DETAIL VIEW
# ============================================================
def get_master_one(jira_id: str) -> Optional[pd.Series]:
    df = fetch_df(f"SELECT {MASTER_SELECT_COLUMNS} FROM {T_MASTER} WHERE jira_id = :jira_id", {"jira_id": jira_id})
    if df.empty:
        return None
    return df.iloc[0]


def get_checks_for_jira(jira_id: str) -> pd.DataFrame:
    ensure_grades_rows_for_jira(jira_id)
    return fetch_df(
        f"""
        SELECT
            c.check_id,
            c.check_group,
            c.check_name,
            c.check_description,
            c.sort_order,
            c.is_active,
            g.status_pre,
            g.notes_pre,
            g.status_post,
            g.notes_post
        FROM {T_CATALOG} c
        LEFT JOIN {T_GRADES} g
          ON g.check_id = c.check_id
         AND g.jira_id = :jira_id
        WHERE c.is_active = 1
        ORDER BY c.sort_order;
    """,
        {"jira_id": jira_id},
    )


def _init_detail_checks_state(jira_id: str, df_checks: pd.DataFrame) -> pd.DataFrame:
    key_meta = "detail_checks_meta"
    key_df = "detail_checks_working"

    meta = st.session_state.get(key_meta)
    if meta and meta.get("jira_id") == jira_id:
        working = st.session_state.get(key_df)
        if isinstance(working, pd.DataFrame) and not working.empty:
            return working

    df = df_checks.copy()
    df["status_pre"] = df["status_pre"].fillna("Not Reviewed")
    df["status_post"] = df["status_post"].fillna("Not Reviewed")
    df["notes_pre"] = df["notes_pre"].fillna("")
    df["notes_post"] = df["notes_post"].fillna("")

    st.session_state[key_meta] = {"jira_id": jira_id}
    st.session_state[key_df] = df
    return df


def build_checks_editor_grid(df_checks: pd.DataFrame) -> pd.DataFrame:
    df_edit = df_checks.copy()
    display_cols = ["check_id", "check_name", "status_pre", "notes_pre", "status_post", "notes_post"]
    df_edit = df_edit[display_cols]

    gb = GridOptionsBuilder.from_dataframe(df_edit)
    gb.configure_default_column(resizable=True, sortable=True, filter=False, editable=False, wrapText=False, autoHeight=False)

    gb.configure_column("check_id", header_name="check_id", hide=True)
    gb.configure_column("check_name", header_name="Check Name", editable=False)

    gb.configure_column(
        "status_pre",
        header_name="Pre Status",
        editable=True,
        cellEditor="agSelectCellEditor",
        cellEditorParams={"values": QC_STATUS_OPTIONS},
        cellStyle=QC_CELLSTYLE_JS,
    )
    gb.configure_column(
        "status_post",
        header_name="Post Status",
        editable=True,
        cellEditor="agSelectCellEditor",
        cellEditorParams={"values": QC_STATUS_OPTIONS},
        cellStyle=QC_CELLSTYLE_JS,
    )

    gb.configure_column("notes_pre", header_name="Pre Notes", editable=True)
    gb.configure_column("notes_post", header_name="Post Notes", editable=True)

    grid_options = gb.build()
    grid_options["domLayout"] = "normal"

    resp = AgGrid(
        df_edit,
        gridOptions=grid_options,
        height=520,
        fit_columns_on_grid_load=True,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        data_return_mode=DataReturnMode.AS_INPUT,
        allow_unsafe_jscode=True,
        theme="balham",
    )
    return pd.DataFrame(resp["data"])


def show_detail_view(jira_id: str) -> None:
    show_flash_messages()

    with st.sidebar:
        st.header("Navigation")
        if st.button("← Back to Master", type="primary", use_container_width=True):
            st.query_params.clear()
            st.rerun()

        st.markdown("### Jump to another ticket")
        all_ids = fetch_df(f"SELECT jira_id FROM {T_MASTER} ORDER BY jira_id")["jira_id"].astype(str).tolist()
        current_idx = all_ids.index(jira_id) if jira_id in all_ids else 0
        jump_id = st.selectbox("JIRA ID", options=all_ids, index=current_idx)
        if st.button("Open selected ticket", use_container_width=True, disabled=(not jump_id)):
            st.query_params.update({"page": "detail", "jira_id": jump_id})
            st.rerun()

        st.divider()
        st.header("Session")
        default_val = st.session_state.get("user_name", QC_INVESTIGATORS[-1])
        if default_val not in QC_INVESTIGATORS:
            default_val = QC_INVESTIGATORS[-1]
        picked = st.selectbox("QC Investigator", QC_INVESTIGATORS, index=QC_INVESTIGATORS.index(default_val))
        st.session_state["user_name"] = picked

    master = get_master_one(jira_id)
    if master is None:
        st.error(f"JIRA ID not found in {T_MASTER}: {jira_id}")
        return

    maps = get_employees_maps()
    names = [""] + maps["names_sorted"]

    def init_key(k: str, v: Any) -> None:
        if k not in st.session_state:
            st.session_state[k] = "" if v is None else v

    # People fields in session_state (used for callbacks + display)
    init_key("project_manager_id", master.get("project_manager_id"))
    init_key("project_manager_name", master.get("project_manager_name"))
    init_key("support_lead_id", master.get("support_lead_id"))
    init_key("support_lead_name", master.get("support_lead_name"))
    init_key("responsible_analyst_id", master.get("responsible_analyst_id"))
    init_key("responsible_analyst_name", master.get("responsible_analyst_name"))
    init_key("region", master.get("region"))

    # ---- callbacks: auto-resolve ID/Name + region, then persist to DB ----
    def _save_people_fields():
        update_master_partial(
            jira_id,
            {
                "project_manager_id": _norm(st.session_state.get("project_manager_id")) or None,
                "project_manager_name": _norm(st.session_state.get("project_manager_name")) or None,
                "support_lead_id": _norm(st.session_state.get("support_lead_id")) or None,
                "support_lead_name": _norm(st.session_state.get("support_lead_name")) or None,
                "responsible_analyst_id": _norm(st.session_state.get("responsible_analyst_id")) or None,
                "responsible_analyst_name": _norm(st.session_state.get("responsible_analyst_name")) or None,
                "region": _norm(st.session_state.get("region")) or None,
            },
        )

    def _on_pm_change():
        sync_person_fields("project_manager", maps, also_region_from_support_lead=False)
        run_sql_safely(_save_people_fields, "detail/people(pm)")

    def _on_sl_change():
        sync_person_fields("support_lead", maps, also_region_from_support_lead=True)
        run_sql_safely(_save_people_fields, "detail/people(sl)")

    def _on_ra_change():
        sync_person_fields("responsible_analyst", maps, also_region_from_support_lead=False)
        run_sql_safely(_save_people_fields, "detail/people(ra)")

    st.title(f"QC Details — {jira_id}")
    st.markdown(f"[Open in JIRA: {jira_id}]({jira_link(jira_id)})")

    created_txt = str(master.get("jira_created_at") or "") if HAS_JIRA_CREATED_AT else ""
    updated_txt = str(master.get("jira_updated_at") or "") if HAS_JIRA_UPDATED_AT else ""
    if created_txt or updated_txt:
        st.caption(f"JIRA Created At: {created_txt} | JIRA Updated At: {updated_txt}")

    if HAS_UPDATED_AT or HAS_UPDATED_BY:
        ua = str(master.get("updated_at") or "") if HAS_UPDATED_AT else ""
        ub = str(master.get("updated_by") or "") if HAS_UPDATED_BY else ""
        st.caption(f"Last DB update: {ua} | {ub}".strip(" |"))

    st.divider()
    st.subheader("General")

    overall_opts = [""] + QC_OVERALL_STATUS_OPTIONS
    inv_opts = [""] + INVESTIGATION_STATUS_OPTIONS

    def _idx(opts, val):
        v = (val or "").strip()
        return opts.index(v) if v in opts else 0

    g1, g2, g3, g4, g5 = st.columns([2.2, 1.2, 1.2, 1.2, 1.0])
    with g1:
        st.text_input("JIRA Title", value=master.get("jira_summary") or "", disabled=True)
    with g2:
        qc_overall_status = st.selectbox(
            "QC Overall Status",
            options=overall_opts,
            index=_idx(overall_opts, master.get("qc_overall_status")),
        )
    with g3:
        qc_investigation_status = st.selectbox(
            "QC Investigation Status",
            options=inv_opts,
            index=_idx(inv_opts, master.get("qc_investigation_status")),
        )
    with g4:
        qc_investigator = st.text_input("QC Investigator", value=master.get("qc_investigator") or "")
    with g5:
        st.text_input("Region", key="region", disabled=True)

    # Dates (no "Set" controls; empty string = NULL)
    st.markdown('<div class="small-muted">Dates accept empty value for NULL. Format: <b>YYYY-MM-DD</b>.</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    pre_check_date_s = d1.text_input("Pre-Check Date", value=date_str(safe_date(master.get("pre_check_date"))))
    check_meeting_date_s = d2.text_input("Check Meeting Date", value=date_str(safe_date(master.get("check_meeting_date"))))
    remediation_meeting_date_s = d3.text_input("Remediation Meeting Date", value=date_str(safe_date(master.get("remediation_meeting_date"))))
    completed_date_s = d4.text_input("Completed Date", value=date_str(safe_date(master.get("completed_date"))))

    r1, r2, r3, r4 = st.columns(4)
    reference_id = r1.text_input("Reference ID", value=master.get("reference_id") or "")
    reference_date_s = r2.text_input("Reference Date", value=date_str(safe_date(master.get("reference_date"))))
    email_after_pre_check = r3.selectbox(
        "Email After Pre-Check",
        YES_NO_OPTIONS,
        index=YES_NO_OPTIONS.index(to_yes_no(master.get("email_after_pre_check"))),
    )
    email_after_check_meeting = r4.selectbox(
        "Email After Check Meeting",
        YES_NO_OPTIONS,
        index=YES_NO_OPTIONS.index(to_yes_no(master.get("email_after_check_meeting"))),
    )

    st.divider()
    st.subheader("People")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.text_input("Project Manager ID", key="project_manager_id", disabled=True)
        st.selectbox("Project Manager Name", options=names, key="project_manager_name", on_change=_on_pm_change)
    with p2:
        st.text_input("Support Lead ID", key="support_lead_id", disabled=True)
        st.selectbox("Support Lead Name", options=names, key="support_lead_name", on_change=_on_sl_change)
        st.caption("Region auto-sync from Support Lead.")
    with p3:
        st.text_input("Responsible Analyst ID", key="responsible_analyst_id", disabled=True)
        st.selectbox("Responsible Analyst Name", options=names, key="responsible_analyst_name", on_change=_on_ra_change)

    st.divider()
    st.subheader("QC Checks (Pre / Post)")

    # ---- checks table + KPIs (KPIs ABOVE table, but computed from edited table via placeholder) ----
    checks_df_db = get_checks_for_jira(jira_id)
    checks_working = _init_detail_checks_state(jira_id, checks_df_db)

    kpi_slot = st.empty()  # KPIs will appear here (above the table)

    edited_checks_df = build_checks_editor_grid(checks_working)
    checks_working = edited_checks_df.copy()
    st.session_state["detail_checks_working"] = checks_working

    pre_counts = _qc_status_to_tbd(checks_working["status_pre"]).value_counts().to_dict()
    post_counts = _qc_status_to_tbd(checks_working["status_post"]).value_counts().to_dict()

    with kpi_slot:
        st.markdown('<div class="small-muted" style="text-align:left;">Not Reviewed is shown as <b>TBD</b>.</div>', unsafe_allow_html=True)

        st.markdown('<div class="small-muted" style="text-align:center; font-size: 16px;">Pre-Check Statuses</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-wrap"><div class="kpi-row">', unsafe_allow_html=True)
        p1c, p2c, p3c, p4c, p5c = st.columns(5)
        with p1c:
            st.markdown(kpi_card_html("Fully", int(pre_counts.get("Fully Conforms", 0)), "kpi-green"), unsafe_allow_html=True)
        with p2c:
            st.markdown(kpi_card_html("Generally", int(pre_counts.get("Generally Conforms", 0)), "kpi-lgreen"), unsafe_allow_html=True)
        with p3c:
            st.markdown(kpi_card_html("Partially", int(pre_counts.get("Partially Conforms", 0)), "kpi-yellow"), unsafe_allow_html=True)
        with p4c:
            st.markdown(kpi_card_html("Does Not", int(pre_counts.get("Does Not Conform", 0)), "kpi-red"), unsafe_allow_html=True)
        with p5c:
            st.markdown(kpi_card_html("TBD", int(pre_counts.get("TBD", 0)), "kpi-gray"), unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown('<div class="small-muted" style="text-align:center; font-size: 16px;">Post-Check Statuses</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-wrap"><div class="kpi-row" style="margin-top:10px;">', unsafe_allow_html=True)
        q1c, q2c, q3c, q4c, q5c = st.columns(5)
        with q1c:
            st.markdown(kpi_card_html("Fully", int(post_counts.get("Fully Conforms", 0)), "kpi-green"), unsafe_allow_html=True)
        with q2c:
            st.markdown(kpi_card_html("Generally", int(post_counts.get("Generally Conforms", 0)), "kpi-lgreen"), unsafe_allow_html=True)
        with q3c:
            st.markdown(kpi_card_html("Partially", int(post_counts.get("Partially Conforms", 0)), "kpi-yellow"), unsafe_allow_html=True)
        with q4c:
            st.markdown(kpi_card_html("Does Not", int(post_counts.get("Does Not Conform", 0)), "kpi-red"), unsafe_allow_html=True)
        with q5c:
            st.markdown(kpi_card_html("TBD", int(post_counts.get("TBD", 0)), "kpi-gray"), unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.subheader("Comments / Notes")
    summary_of_findings = st.text_area("Summary of Findings", value=master.get("summary_of_findings") or "", height=140)
    check_meeting_comments = st.text_area("Check Meeting Comments", value=master.get("check_meeting_comments") or "", height=120)
    remediation_meeting_comments = st.text_area(
        "Remediation Meeting Comments", value=master.get("remediation_meeting_comments") or "", height=120
    )

    st.divider()
    csave1, _ = st.columns([1, 6])
    with csave1:
        if st.button("Save", type="primary", use_container_width=True):
            payload = {
                "qc_overall_status": qc_overall_status or None,
                "qc_investigation_status": qc_investigation_status or None,
                "qc_investigator": qc_investigator or None,
                "pre_check_date": parse_date_str(pre_check_date_s),
                "check_meeting_date": parse_date_str(check_meeting_date_s),
                "remediation_meeting_date": parse_date_str(remediation_meeting_date_s),
                "completed_date": parse_date_str(completed_date_s),
                "reference_id": reference_id or None,
                "reference_date": parse_date_str(reference_date_s),
                "email_after_pre_check": from_yes_no(email_after_pre_check),
                "email_after_check_meeting": from_yes_no(email_after_check_meeting),
                "project_manager_id": _norm(st.session_state.get("project_manager_id")) or None,
                "project_manager_name": _norm(st.session_state.get("project_manager_name")) or None,
                "support_lead_id": _norm(st.session_state.get("support_lead_id")) or None,
                "support_lead_name": _norm(st.session_state.get("support_lead_name")) or None,
                "responsible_analyst_id": _norm(st.session_state.get("responsible_analyst_id")) or None,
                "responsible_analyst_name": _norm(st.session_state.get("responsible_analyst_name")) or None,
                "region": _norm(st.session_state.get("region")) or None,
                "summary_of_findings": summary_of_findings or None,
                "check_meeting_comments": check_meeting_comments or None,
                "remediation_meeting_comments": remediation_meeting_comments or None,
            }

            payload = truncate_fields(payload, FIELD_MAX_LENGTHS)
            ok = run_sql_safely(lambda: update_master_partial(jira_id, payload), "detail/update_master_partial")
            if not ok:
                return

            def _save_grades():
                for _, r in checks_working.iterrows():
                    upsert_grade_row(
                        jira_id=jira_id,
                        check_id=r["check_id"],
                        status_pre=(r["status_pre"] or None),
                        notes_pre=(r["notes_pre"] or None),
                        status_post=(r["status_post"] or None),
                        notes_post=(r["notes_post"] or None),
                    )

            ok2 = run_sql_safely(_save_grades, "detail/upsert_grade_row")
            if not ok2:
                return

            flash_success("Saved.")
            st.rerun()


# ============================================================
# ROUTING
# ============================================================
def main() -> None:
    params = st.query_params
    page = params.get("page", "master")
    jira_id = params.get("jira_id", "")

    if page == "detail" and jira_id:
        j = jira_id.strip().upper()
        if not JIRA_ID_PATTERN.match(j):
            st.error("Invalid JIRA ID format in URL.")
            st.stop()
        show_detail_view(j)
    else:
        show_master_view()


if __name__ == "__main__":
    main()
