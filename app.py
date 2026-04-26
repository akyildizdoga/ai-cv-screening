import os
import json
import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(
    page_title="AI-Powered CV Screening",
    page_icon="📄",
    layout="centered"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("AI-Powered CV Screening & Job Matching")
st.write(
    "This app analyzes a candidate CV against a job description using OpenAI embeddings "
    "and provides HR-focused recruitment insights."
)

st.divider()

uploaded_file = st.file_uploader("Upload Candidate CV as PDF", type=["pdf"])

manual_cv_text = st.text_area(
    "Or paste Candidate CV text here",
    height=180
)

job_desc = st.text_area(
    "Paste Job Description",
    height=180
)


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def extract_skills(text):
    keywords = [
        "Python", "Java", "C#", "MATLAB", "JavaScript",
        "Node.js", "React", "HTML", "CSS",
        "AWS", "Cloud", "API", "REST API",
        "AI", "Artificial Intelligence", "Machine Learning",
        "NLP", "OpenAI", "Gemini", "Chatbot",
        "SQL", "Database", "Git", "GitHub"
    ]

    found_skills = []

    for skill in keywords:
        if skill.lower() in text.lower():
            found_skills.append(skill)

    return found_skills


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_score_breakdown(cv_text, job_desc, match_score):
    cv_lower = cv_text.lower()
    job_lower = job_desc.lower()

    technical_keywords = [
        "python", "java", "c#", "matlab", "node.js", "aws",
        "api", "ai", "chatbot", "machine learning", "cloud"
    ]

    experience_keywords = [
        "internship", "project", "team", "backend",
        "development", "engineering", "autonomous", "vehicle"
    ]

    technical_matches = sum(
        1 for word in technical_keywords
        if word in cv_lower and word in job_lower
    )

    experience_matches = sum(
        1 for word in experience_keywords
        if word in cv_lower and word in job_lower
    )

    technical_score = min(round((technical_matches / len(technical_keywords)) * 100), 100)
    experience_score = min(round((experience_matches / len(experience_keywords)) * 100), 100)

    overall_fit = round(
        (technical_score * 0.5) + (experience_score * 0.3) + (match_score * 0.2),
        2
    )

    return technical_score, experience_score, overall_fit


def get_ai_analysis(cv_text, job_desc, match_score):
    prompt = f"""
You are a professional HR recruitment assistant.

Analyze the CV and job description.

Return ONLY valid JSON. Do not write any explanation outside JSON.

Use exactly this JSON structure:

{{
  "overall_evaluation": "text",
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "cv_improvement_suggestions": ["suggestion1", "suggestion2"],
  "recruiter_decision": "Recommend Interview",
  "decision_reason": "text"
}}

Recruiter decision must be one of:
- Recommend Interview
- Needs HR Review
- Not Recommended

CV:
{cv_text}

Job Description:
{job_desc}

Match Score:
{match_score}%
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You MUST return valid JSON only."},
            {"role": "user", "content": prompt}
        ]
    )

    raw_output = response.choices[0].message.content.strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        st.error("AI JSON parse failed. Raw output:")
        st.text(raw_output)
        return None


def show_weakness_message(skill):
    critical_words = ["lack", "limited", "no experience", "missing", "insufficient"]

    if any(word in skill.lower() for word in critical_words):
        st.error(f"❌ {skill}")  # Critical risk
    else:
        st.warning(f"⚠️ {skill}")  # Moderate risk


def generate_pdf_report(
    match_score,
    technical_score,
    experience_score,
    overall_fit,
    extracted_skills,
    overall_evaluation,
    matched_skills,
    missing_skills,
    suggestions,
    decision,
    reason
):
    file_path = "cv_analysis_report.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("AI CV Screening Report", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Match Score: {match_score}%", styles["Normal"]))
    content.append(Paragraph(f"Technical Skills Match: {technical_score}%", styles["Normal"]))
    content.append(Paragraph(f"Experience Match: {experience_score}%", styles["Normal"]))
    content.append(Paragraph(f"Overall Fit: {overall_fit}%", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Extracted Skills from CV", styles["Heading2"]))
    if extracted_skills:
        for skill in extracted_skills:
            content.append(Paragraph(f"- {skill}", styles["Normal"]))
    else:
        content.append(Paragraph("No predefined skills were detected.", styles["Normal"]))

    content.append(Spacer(1, 12))

    content.append(Paragraph("Overall HR Evaluation", styles["Heading2"]))
    content.append(Paragraph(overall_evaluation, styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Matched Skills", styles["Heading2"]))
    if matched_skills:
        for skill in matched_skills:
            content.append(Paragraph(f"- {skill}", styles["Normal"]))
    else:
        content.append(Paragraph("No matched skills identified.", styles["Normal"]))

    content.append(Spacer(1, 12))

    content.append(Paragraph("Missing or Weak Skills", styles["Heading2"]))
    if missing_skills:
        for skill in missing_skills:
            content.append(Paragraph(f"- {skill}", styles["Normal"]))
    else:
        content.append(Paragraph("No major missing skills identified.", styles["Normal"]))

    content.append(Spacer(1, 12))

    content.append(Paragraph("CV Improvement Suggestions", styles["Heading2"]))
    if suggestions:
        for suggestion in suggestions:
            content.append(Paragraph(f"- {suggestion}", styles["Normal"]))
    else:
        content.append(Paragraph("No specific CV improvement suggestions available.", styles["Normal"]))

    content.append(Spacer(1, 12))

    content.append(Paragraph("Recruiter Decision", styles["Heading2"]))
    content.append(Paragraph(decision, styles["Normal"]))
    content.append(Paragraph(reason, styles["Normal"]))

    doc.build(content)

    return file_path


if st.button("Analyze Match"):
    cv_text = ""

    if uploaded_file is not None:
        cv_text = extract_text_from_pdf(uploaded_file)
    elif manual_cv_text.strip():
        cv_text = manual_cv_text.strip()

    if not cv_text:
        st.warning("Please upload a PDF CV or paste CV text.")
    elif not job_desc.strip():
        st.warning("Please enter a job description.")
    else:
        with st.spinner("Analyzing CV and job description..."):
            cv_embedding = get_embedding(cv_text)
            job_embedding = get_embedding(job_desc)

            similarity = cosine_similarity(cv_embedding, job_embedding)
            match_score = round(similarity * 100, 2)

            st.subheader("Matching Result")
            st.metric("Match Score", f"{match_score}%")

            st.caption("Note: Match score is based on semantic similarity and may differ from AI evaluation.")
            st.caption("This system combines semantic similarity scoring with AI-based HR evaluation.")

            st.progress(match_score / 100)

            if match_score >= 70:
                st.success("Strong Match ✅ - Recommend interview")
            elif match_score >= 45:
                st.warning("Moderate Match ⚠️ - Needs HR review")
            else:
                st.error("Low Match ❌ - Not recommended at this stage")

            st.divider()

            st.subheader("Extracted Skills from CV")
            extracted_skills = extract_skills(cv_text)

            if extracted_skills:
                for skill in extracted_skills:
                    st.write(f"• {skill}")
            else:
                st.info("No predefined skills were detected in the CV.")

            st.divider()

            st.subheader("Score Breakdown")
            technical_score, experience_score, overall_fit = get_score_breakdown(
                cv_text,
                job_desc,
                match_score
            )

            st.write(f"Technical Skills Match: {technical_score}%")
            st.write(f"Experience Match: {experience_score}%")
            st.write(f"Overall Fit: {overall_fit}%")

            analysis = get_ai_analysis(cv_text, job_desc, match_score)

            if analysis:
                st.divider()

                overall_evaluation = analysis.get("overall_evaluation", "No evaluation available.")
                matched_skills = analysis.get("matched_skills", [])
                missing_skills = analysis.get("missing_skills", [])
                suggestions = analysis.get("cv_improvement_suggestions", [])
                decision = analysis.get("recruiter_decision", "Needs HR Review")
                reason = analysis.get("decision_reason", "")

                st.subheader("Overall HR Evaluation")
                st.write(overall_evaluation)

                st.subheader("Top Strengths")
                if matched_skills:
                    for skill in matched_skills[:3]:
                        st.success(f"✅ {skill}")
                else:
                    st.info("No top strengths identified.")

                st.subheader("Top Weaknesses")
                if missing_skills:
                    for skill in missing_skills[:3]:
                        show_weakness_message(skill)
                else:
                    st.info("No major weaknesses identified.")

                st.subheader("Matched Skills")
                if matched_skills:
                    for skill in matched_skills:
                        st.success(f"✅ {skill}")
                else:
                    st.info("No matched skills identified.")

                st.subheader("Missing or Weak Skills")
                if missing_skills:
                    for skill in missing_skills:
                        show_weakness_message(skill)
                else:
                    st.info("No major missing skills identified.")

                st.subheader("CV Improvement Suggestions")
                if suggestions:
                    for suggestion in suggestions:
                        st.write(f"• {suggestion}")
                else:
                    st.info("No specific CV improvement suggestions available.")

                st.subheader("Recruiter Decision")

                if decision == "Recommend Interview":
                    st.success(f"✅ {decision}")
                elif decision == "Needs HR Review":
                    st.warning(f"⚠️ {decision}")
                else:
                    st.error(f"❌ {decision}")

                st.write(reason)

                pdf_file = generate_pdf_report(
                    match_score,
                    technical_score,
                    experience_score,
                    overall_fit,
                    extracted_skills,
                    overall_evaluation,
                    matched_skills,
                    missing_skills,
                    suggestions,
                    decision,
                    reason
                )

                with open(pdf_file, "rb") as file:
                    st.download_button(
                        label="Download Report as PDF",
                        data=file,
                        file_name="cv_analysis_report.pdf",
                        mime="application/pdf"
                    )