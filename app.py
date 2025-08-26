# import streamlit as st
# APP_PASSWORD = st.secrets["APP_PASSWORD"]
# st.title("Protected App")
# password = st.text_input("Enter password", type="password")
# if password == APP_PASSWORD:
#     st.success("Access granted")
#     st.write("Your private app content goes here...")
# else:
#     st.error("Access denied")

import fitz # PyMuPDF
import re
import io
import time
import json
import pandas as pd
from typing import Dict, List, NamedTuple
from PIL import Image
import streamlit as st
import google.generativeai as genai
import requests
import xlsxwriter 
from collections import defaultdict

# ----------------- APP CONFIGURATION -----------------

st.set_page_config(page_title="Climate Heroes KPI Extractor", page_icon="🌍", layout="wide")

# --- Model & API Config ---
AVAILABLE_MODELS = {
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite (Recommended)",
    "gemini-2.5-flash": "Gemini 2.5 Flash (High Quality)",
}
DEFAULT_MODEL_KEY = "gemini-2.5-flash-lite"

# --- KPI & Ranking Definitions ---
KPI_DEFINITIONS = {
    "Scope 1 Emissions": {"keywords": ["scope 1", "scope-1", "direct ghg"], "default_selected": True},
    "Scope 2 Emissions": {"keywords": ["scope 2", "scope-2", "indirect energy"], "default_selected": True},
    "Scope 3 Emissions": {"keywords": ["scope 3", "scope-3", "value chain"], "default_selected": True},
    "Total Energy Usage": {"keywords": ["total energy consumption", "energy usage"], "default_selected": False},
    "Emissions Intensity": {"keywords": ["emissions intensity", "carbon intensity"], "default_selected": False},
    "Energy Intensity": {"keywords": ["energy intensity"], "default_selected": False}
}
EMISSIONS_KPIS = ["Scope 1 Emissions", "Scope 2 Emissions", "Scope 3 Emissions"]
GHG_UNITS = ["tco2e", "mt co2e", "co2-e", "metric tons of co2"]

class ReportInput(NamedTuple):
    company: str
    year: int
    url: str

# ----------------- API CONFIGURATION -----------------

@st.cache_resource
def configure_api():
    api_key = st.secrets.get("GEMINI_API_KEY_R5Y") or st.secrets.get("GOOGLE_API_KEY_RK") or st.secrets.get("GEMINI_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key); return True
        except Exception: return False
    return False

api_key_configured = configure_api()

@st.cache_data(show_spinner=False)
def find_and_rank_candidate_pages(pdf_content: bytes, kpis_to_search: List[str], target_year: int) -> Dict:
    candidate_pages = defaultdict(lambda: {'score': 0, 'kpis': set()})
    years_to_check = [str(y) for y in range(target_year, target_year - 3, -1)]
    with fitz.open(stream=pdf_content, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text").lower()
            if not text.strip(): continue
            score = 0
            if page.find_tables(): score += 25
            if any(year in text for year in years_to_check): score += 20
            if any(unit in text for unit in GHG_UNITS): score += 15
            found_kpis = {kpi for kpi in kpis_to_search if any(kw in text for kw in KPI_DEFINITIONS[kpi]['keywords'])}
            if found_kpis:
                score += 10 * len(found_kpis)
                candidate_pages[page_num]['score'] += score
                candidate_pages[page_num]['kpis'].update(found_kpis)
    ranked_candidates = {kpi: [] for kpi in kpis_to_search}
    if not candidate_pages: return ranked_candidates
    sorted_pages = sorted(candidate_pages.items(), key=lambda item: -item[1]['score'])
    for page_num, data in sorted_pages:
        for kpi in data.get('kpis', []):
            ranked_candidates[kpi].append({"page": page_num, "score": data['score']})
    return ranked_candidates

def generate_prompt(kpi: str, target_year: int, page_num: int) -> str:
    kpi_specific_instructions = {
        "Scope 1 Emissions": "Look for a row labeled 'Scope 1' or 'Total Scope 1'. Prioritize the single, pre-calculated total value.",
        "Scope 2 Emissions": "Search for 'Scope 2'. You MUST identify if 'market-based' and/or 'location-based' values are provided. Extract each as a separate JSON object.",
        "Scope 3 Emissions": "Look for a row labeled 'Scope 3' or 'Total Scope 3'. Find the single, pre-calculated total. Do NOT sum the individual categories yourself."
    }
    return f"""
    You are an expert AI assistant specializing in sustainability report data extraction.
    Analyze the provided image of a report page to extract the following metric.
    **Analysis Task:**
    1.  **Target KPI:** "{kpi}"
    2.  **Target Year:** "{target_year}". If this year is not present, use the most recent year available in the data.
    3.  **Source Page:** You are analyzing page `{page_num}`.
    **Instructions:**
    - Locate the specific KPI and the value corresponding to the target year (or most recent).
    - {kpi_specific_instructions.get(kpi, "Extract the total value for the specified KPI.")}
    - The value must be a number only, without commas or text.
    - If the KPI is not found on this page, return an empty array `[]`.
    **CRITICAL:** You MUST return a JSON array of objects. Do not return any other text or explanation.
    **Required JSON Format:**
    ```json
    [
      {{
        "kpi": "{kpi}",
        "value": <number, or null if not found>,
        "unit": "<string, e.g., 'tCO2e'>",
        "year": <number, the year the data is for>,
        "variant": "<'market-based', 'location-based', or null>",
        "page_number_confirmed": {page_num},
        "confidence": <float, 0.0 to 1.0, your confidence in the extraction>
      }}
    ]
    ```
    """

def extract_kpi_with_gemini(image: Image.Image, kpi: str, target_year: int, model_id: str, page_num: int) -> List[Dict]:
    """
    Extracts KPIs from a page image using a Gemini model with deterministic settings.
    Backward-compatible with older google-generativeai SDKs.
    """
    if not api_key_configured:
        return [{"kpi": kpi, "error": "API Key not configured."}]

    # Deterministic settings; fall back if SDK doesn't support response_mime_type
    try:
        generation_config = genai.GenerationConfig(
            temperature=0.0,
            top_p=0.95,
            response_mime_type='application/json'
        )
    except TypeError:
        generation_config = genai.GenerationConfig(
            temperature=0.0,
            top_p=0.95
        )

    model = genai.GenerativeModel(model_id, generation_config=generation_config)
    prompt = generate_prompt(kpi, target_year, page_num)

    try:
        response = model.generate_content([prompt, image])
        # Prefer direct JSON; otherwise try to extract a JSON array from text
        try:
            parsed_json = json.loads(getattr(response, 'text', '') or '')
        except Exception:
            text = getattr(response, 'text', '') or ''
            match = re.search(r"\[\s*{[\s\S]*}\s*\]", text)
            parsed_json = json.loads(match.group(0)) if match else []
        return parsed_json if isinstance(parsed_json, list) else [parsed_json]
    except Exception as e:
        return [{"kpi": kpi, "error": f"API/JSON Error: {str(e)}"}]

def page_to_image(doc: fitz.Document, page_num: int, dpi: int) -> Image.Image:
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png")))

# ----------------- SESSION STATE MANAGEMENT -----------------

def initialize_session_state():
    """Initializes all required session state variables."""
    if 'kpi_selections' not in st.session_state:
        st.session_state.kpi_selections = {kpi: data["default_selected"] for kpi, data in KPI_DEFINITIONS.items()}
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'diagnostics' not in st.session_state:
        st.session_state.diagnostics = defaultdict(int)
    
    # Initialize the input DataFrame for the data editor
    if 'report_df' not in st.session_state:
        st.session_state.report_df = pd.DataFrame(
            [{"Company": "", "Year": None, "URL": ""}],
            columns=["Company", "Year", "URL"]
        )
    # Initialize the number_input value from the DataFrame length
    if 'num_rows' not in st.session_state:
        st.session_state.num_rows = len(st.session_state.report_df)


def reset_session_state():
    """Clears the session state to start a new batch."""
    st.session_state.clear()
    initialize_session_state()

# ----------------- STREAMLIT UI (with Enhancements) -----------------

def main():
    st.title("🌍 Sustainability KPI Extractor (MVP2)")
    st.success("Batch-process sustainability reports to extract key metrics. Empowering Our Climate Heroes!")
    
    initialize_session_state()

    with st.sidebar:
        st.header("⚙️ Settings & Info")
        model_display_name = st.selectbox("Select AI Model", options=list(AVAILABLE_MODELS.values()), index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_KEY))
        st.session_state.dpi_setting = st.slider("PDF Rendering DPI", 100, 300, 150, help="Higher DPI improves image quality for the AI.")
        st.session_state.scan_pages = st.slider("Max pages to scan per KPI", 1, 5, 2, help="Limits API calls for each KPI to the top-ranked pages.")
        st.markdown("---")
        st.header("📊 Batch Run Stats")
        d = st.session_state.diagnostics
        c1, c2 = st.columns(2)
        c1.metric("Reports Processed", f"{d['reports_processed']} / {d['total_reports']}")
        c1.metric("API Calls Made", d['api_calls'])
        c2.metric("KPIs Found", d['kpis_found'])
        if d.get('total_kpis_targeted', 0) > 0:
            success_rate = (d.get('kpis_found', 0) / d['total_kpis_targeted']) * 100
            c2.metric("Success Rate", f"{success_rate:.1f}%")

    with st.container(border=True):
        st.subheader("📄 1. Provide Report Details")
        num_rows = st.number_input(
            "Number of Reports", 
            min_value=1, 
            max_value=500, 
            step=1, 
            key='num_rows',
            help="Set the number of reports you want to process. The table below will update automatically."
        )

        current_rows = len(st.session_state.report_df)
        if num_rows > current_rows:
            new_rows_count = num_rows - current_rows
            new_rows = pd.DataFrame([{"Company": "", "Year": None, "URL": ""}] * new_rows_count, columns=["Company", "Year", "URL"])
            st.session_state.report_df = pd.concat([st.session_state.report_df, new_rows], ignore_index=True)
        elif num_rows < current_rows:
            st.session_state.report_df = st.session_state.report_df.head(num_rows)

        df_for_display = st.session_state.report_df.copy()
        df_for_display.index = pd.RangeIndex(start=1, stop=len(df_for_display) + 1)

        edited_df = st.data_editor(
            df_for_display, 
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "Company": st.column_config.TextColumn("Company Name", required=True, help="The name of the company."),
                "Year": st.column_config.NumberColumn("Target Year", required=True, help="The 4-digit reporting year (e.g., 2023)."),
                "URL": st.column_config.LinkColumn("Report URL", required=True, help="Public URL of the PDF report.")
            }
        )
        edited_df = edited_df.reset_index(drop=True)
        st.session_state.report_df = edited_df


    with st.container(border=True):
        st.subheader("🎯 2. Select KPIs to Target")
        kpi_cols = st.columns(3)
        for i, (kpi, data) in enumerate(KPI_DEFINITIONS.items()):
            is_selected = st.session_state.kpi_selections.get(kpi, data['default_selected'])
            kpi_cols[i % 3].checkbox(kpi, value=is_selected, key=f"kpi_select_{kpi}")

    selected_kpis = [kpi for kpi, data in KPI_DEFINITIONS.items() if st.session_state.get(f"kpi_select_{kpi}", data['default_selected'])]

    c1, c2 = st.columns([2, 1])
    extract_clicked = c1.button("🚀 Extract KPIs from All Reports", type="primary", use_container_width=True, disabled=not api_key_configured)
    if c2.button("↻ Start New Batch", use_container_width=True):
        reset_session_state()
        st.rerun()

    if not api_key_configured:
        st.warning("🚨 Gemini API key not configured. Please add it to your Streamlit secrets (`secrets.toml`).")

    if extract_clicked:
        valid_reports = []
        error_messages = []
        df_to_process = st.session_state.report_df 
        
        for index, row in df_to_process.iterrows():
            company = row['Company']
            year = row['Year']
            url = row['URL']
            
            if not company or pd.isna(year) or not url:
                error_messages.append(f"- **Row {index + 1}:** Missing one or more required fields (Company, Year, URL).")
            elif not (isinstance(year, (int, float)) and 2000 <= year <= 2050):
                error_messages.append(f"- **Row {index + 1}:** Year '{year}' is not a valid 4-digit year.")
            elif not str(url).startswith('http'):
                 error_messages.append(f"- **Row {index + 1}:** URL must start with 'http' or 'https'.")
            else:
                valid_reports.append(ReportInput(company=str(company), year=int(year), url=str(url)))
        
        if len(valid_reports) != len(df_to_process):
            error_header = f"You set the number of reports to **{len(df_to_process)}**, but some information is incomplete. Please fix the following issues:"
            st.error(error_header + "\n" + "\n".join(error_messages))
            st.stop()

        if not valid_reports: st.warning("No valid reports to process."); st.stop()
        if not selected_kpis: st.warning("Please select at least one KPI."); st.stop()
        st.session_state.diagnostics['total_reports'] = len(valid_reports)
        st.session_state.diagnostics['total_kpis_targeted'] = len(valid_reports) * len(selected_kpis)
        all_results = []
        start_time = time.time()
        
        status = st.status(f"Starting batch processing for {len(valid_reports)} reports...", expanded=True)
        for i, report in enumerate(valid_reports):
            try:
                status.write(f"**Processing report {i+1}/{len(valid_reports)}: {report.company} ({report.year})**")
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(report.url, headers=headers, timeout=30)
                response.raise_for_status()
                pdf_content = response.content
                candidate_pages = find_and_rank_candidate_pages(pdf_content, selected_kpis, report.year)
                with fitz.open(stream=pdf_content, filetype="pdf") as doc:
                    model_id = next(k for k, v in AVAILABLE_MODELS.items() if v == model_display_name)

                    for kpi in selected_kpis:
                        pages_to_scan = candidate_pages.get(kpi, [])
                        if not pages_to_scan:
                            all_results.append({"Company": report.company, "Report URL": report.url, "Target Year": report.year, "kpi": kpi, "value": None, "error": "No relevant pages found."})
                            continue

                        found_kpi = False # Flag to track if we find this KPI for this report
                        for page_info in pages_to_scan[:st.session_state.scan_pages]:
                            page_num = page_info["page"]
                            status.write(f"- Analyzing page `{page_num}` (Score: {page_info['score']}) for **{kpi}**...")
                            img = page_to_image(doc, page_num, dpi=st.session_state.dpi_setting)
                            st.session_state.diagnostics['api_calls'] += 1
                            extraction_results = extract_kpi_with_gemini(img, kpi, report.year, model_id, page_num)

                            # Check if the AI returned any valid data
                            results_with_values = [res for res in extraction_results if res.get('value') is not None]

                            if results_with_values:
                                found_kpi = True # Set the flag to True
                                for res in results_with_values:
                                    res.update({"Company": report.company, "Report URL": report.url, "Target Year": report.year, "page": page_num})
                                    all_results.append(res)
                                break # Exit the page scan loop for this KPI, as we've found it

                        if found_kpi:
                            st.session_state.diagnostics['kpis_found'] += 1
                        else:
                            all_results.append({"Company": report.company, "Report URL": report.url, "Target Year": report.year, "kpi": kpi, "value": None, "error": "Not found in top pages."})
                st.session_state.diagnostics['reports_processed'] += 1
            except Exception as e:
                st.error(f"Failed to process {report.company} report: {e}")
        total_time = time.time() - start_time
        status.update(label=f"Batch processing complete in {total_time:.2f} seconds!", state="complete", expanded=False)
        st.session_state.results = all_results
        st.rerun()
    if st.session_state.get('results'):
        st.markdown("---")
        st.subheader("📊 Extraction Results")
        df = pd.DataFrame(st.session_state.results)
        df_display = df[pd.notna(df['value'])].copy()
        if not df_display.empty:
            df_display['Source Link'] = df_display.apply(lambda row: f"{row['Report URL']}#page={int(row['page'])}" if pd.notna(row.get('page')) else None, axis=1)
            final_cols = {
                "Company": "Company", "Target Year": "Year", "kpi": "Parameter Name",
                "variant": "Variant", "value": "Parameter Value", "unit": "Unit",
                "page": "Source Page", "Source Link": "Verify Link", "confidence": "Confidence"
            }
            df_display = df_display.rename(columns=final_cols)
            df_display = df_display[list(final_cols.values())].astype({'Source Page': 'Int64'})
            df_display.index = pd.RangeIndex(start=1, stop=len(df_display) + 1)
            st.dataframe(df_display, use_container_width=True, column_config={"Verify Link": st.column_config.LinkColumn("Verify", display_text="🔗 Link")})
            st.markdown("---"); st.subheader("📥 Export Results")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_display.to_excel(writer, index=False, sheet_name='KPI_Results')
                workbook  = writer.book
                worksheet = writer.sheets['KPI_Results']
                link_format = workbook.add_format({'font_color': 'blue', 'underline': 1})
                try:
                    verify_link_col_idx = df_display.columns.get_loc('Verify Link')
                    for row_num, url in enumerate(df_display['Verify Link'], 1):
                        if pd.notna(url):
                            worksheet.write_url(row_num, verify_link_col_idx, url, cell_format=link_format, string='🔗 Link')
                except KeyError:
                    pass
            st.download_button(label="📥 Download as Excel",
                            data=output.getvalue(),
                            file_name="KPI_Batch_Extraction.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True)            
        df_errors = df[pd.isna(df['value'])].copy()
        if not df_errors.empty:
            with st.expander("⚠️ View Extraction Failures"):
                st.dataframe(df_errors[['Company', 'Target Year', 'kpi', 'error']], use_container_width=True)

if __name__ == "__main__":
    main()


