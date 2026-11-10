# Project report and editable files

The report tells the story of the project: how I chose the residual-network structure, which training changes helped, and how the validation results compared with Kaggle.

- [PDF report](Zicheng_Zhou_Deep_Learning_Project_1_Report.pdf): the current eight-page English report.
- [Word document](Zicheng_Zhou_Report_Editable.docx): editable text, tables, and embedded figures.
- [Markdown manuscript](manuscript_editable.md) and [HTML manuscript](manuscript_editable.html): convenient text versions.
- [Text without embedded figures](../稿件纯文本.md): the manuscript with figure captions kept in place.
- [Figures](figures/): PNG, PDF, and editable SVG files, together with the plotting data.
- [Structured manuscript](report_content.json): section text and layout settings for the report builder.

The current PDF and Word files follow the version reviewed on November 9. The main results are 97.52% local validation accuracy, 89.40% public accuracy, and 86.30% for the final selected private score. The results table also includes the refined robust candidate's 87.30% private score.

## Editing the report

For ordinary revisions, edit the Word document and export it as PDF. Check the page count after editing; the course limit is eight pages including figures, tables, and references.

To work with the structured text, install `requirements-report.txt` and run:

```powershell
python build_report.py
```

The builder writes to `rebuilt/`. It generates a PDF, Markdown, HTML, and a layout check. Word edits and `report_content.json` are separate editing routes, so update the version you intend to use.

The builder uses Times New Roman on Windows. On other systems, pass `--font-dir` with a directory containing `times.ttf`, `timesbd.ttf`, and `timesi.ttf`.

## Editing the figures

Each figure has an SVG for direct editing and an aggregate CSV for plotting. See [the figure guide](figures/README_zh.md) for file names and commands. Keep the manuscript's figure captions with the corresponding images when moving Markdown or HTML files.
