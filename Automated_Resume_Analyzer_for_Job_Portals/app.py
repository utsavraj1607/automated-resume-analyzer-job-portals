import json
import os
import tempfile

import pandas as pd
import streamlit as st

from parser import ResumeParser


st.set_page_config(
    page_title="Automated Resume Analyzer",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading NLP models (first run only)...")
def load_parser():
    return ResumeParser()


def render_contact_info(contact: dict):
    cols = st.columns(4)
    labels = ["Email", "Phone", "LinkedIn", "GitHub"]
    keys = ["email", "phone", "linkedin", "github"]
    for col, label, key in zip(cols, labels, keys):
        value = contact.get(key)
        col.metric(label, value if value else "—")


def render_skills(skills: list):
    if not skills:
        st.info("No skills detected against the skill knowledge base.")
        return
    df = pd.DataFrame(skills)
    for category in sorted(df["category"].unique()):
        cat_skills = df[df["category"] == category]["skill"].tolist()
        st.markdown(f"**{category}**")
        st.write(", ".join(cat_skills))


def render_education(entries: list):
    if not entries:
        st.info("No education entries detected.")
        return
    for e in entries:
        title = e.get("degree") or "Education entry"
        date_range = " – ".join(filter(None, [e.get("start_date"), e.get("end_date")]))
        with st.container(border=True):
            st.markdown(f"**{title}**" + (f"  \n{date_range}" if date_range else ""))
            if e.get("institution"):
                st.caption(e["institution"])


def render_experience(entries: list):
    if not entries:
        st.info("No experience entries detected.")
        return
    for e in entries:
        date_range = " – ".join(filter(None, [e.get("start_date"), e.get("end_date")]))
        with st.container(border=True):
            title_line = " · ".join(filter(None, [e.get("job_title"), e.get("company")]))
            st.markdown(f"**{title_line or 'Experience entry'}**" + (f"  \n{date_range}" if date_range else ""))
            for bullet in e.get("description", []):
                st.markdown(f"- {bullet}")


def render_projects(entries: list):
    if not entries:
        st.info("No projects detected.")
        return
    for p in entries:
        with st.container(border=True):
            st.markdown(f"**{p.get('title', 'Project')}**")
            for bullet in p.get("description", []):
                st.markdown(f"- {bullet}")


def main():
    st.title("📄 Automated Resume Analyzer")
    st.caption(
        "Upload a resume (PDF or DOCX) to automatically extract contact details, "
        "education, experience, projects, and skills into structured JSON."
    )

    parser = load_parser()

    with st.sidebar:
        st.header("Upload Resume")
        uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
        st.markdown("---")
        st.markdown(
            "**How it works**\n\n"
            "• Text is extracted from the PDF/DOCX\n\n"
            "• Noise is cleaned and the text is segmented into sections\n\n"
            "• Regex + spaCy NER extract contact info & entities\n\n"
            "• Skills are matched against a skill knowledge base\n\n"
            "• Everything is structured into JSON"
        )

    if uploaded_file is None:
        st.info("👈 Upload a resume from the sidebar to get started.")
        return

    ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        with st.spinner("Parsing resume..."):
            result = parser.parse_file(tmp_path)
    except Exception as exc:
        st.error(f"Failed to parse resume: {exc}")
        return
    finally:
        os.remove(tmp_path)

    st.success(f"Parsed **{result.get('candidate_name') or 'candidate'}**'s resume successfully.")

    tab_overview, tab_edu, tab_exp, tab_proj, tab_skills, tab_json = st.tabs(
        ["Overview", "Education", "Experience", "Projects", "Skills", "Raw JSON"]
    )

    with tab_overview:
        st.subheader(result.get("candidate_name") or "Name not detected")
        render_contact_info(result.get("contact_info", {}))
        if result.get("summary"):
            st.markdown("**Summary**")
            st.write(result["summary"])
        st.caption(f"Sections detected: {', '.join(result.get('sections_detected', []))}")

    with tab_edu:
        render_education(result.get("education", []))

    with tab_exp:
        render_experience(result.get("experience", []))

    with tab_proj:
        render_projects(result.get("projects", []))

    with tab_skills:
        render_skills(result.get("skills", []))

    with tab_json:
        st.json(result)
        st.download_button(
            label="⬇ Download JSON",
            data=json.dumps(result, indent=2),
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_parsed.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()