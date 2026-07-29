from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet

from pathlib import Path


def create_pdf(
        patient_id,
        report_text):

    styles = getSampleStyleSheet()

    folder = Path("results") / patient_id

    pdf = folder / "report.pdf"

    doc = SimpleDocTemplate(
        str(pdf)
    )

    story = []

    story.append(
        Paragraph(
            "<b>Brain Tumor Analysis Report</b>",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1,20)
    )

    for line in report_text.split("\n"):

        story.append(
            Paragraph(
                line,
                styles["BodyText"]
            )
        )

    story.append(
        Spacer(1,20)
    )

    if (folder/"overlay.png").exists():

        story.append(
            Image(
                str(folder/"overlay.png"),
                width=350,
                height=350
            )
        )

    doc.build(story)

    return pdf