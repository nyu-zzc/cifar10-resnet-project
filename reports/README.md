# Final report and editable materials

The delivered PDF and Word document are the reviewed eight-page versions. The PDF is the fixed-layout submission artifact; DOCX, Markdown, HTML and report_content.json are editable working sources. The Word document contains editable text/tables and embedded figures, while each figure has an editable SVG and reproducible aggregate source table in figures/.

- `Zicheng_Zhou_Deep_Learning_Project_1_Report.pdf`: final eight-page report.
- `Zicheng_Zhou_Report_Editable.docx`: reviewed editable Word version.
- `manuscript_editable.md` / `.html`: editable text and figure links.
- `report_content.json`: structured content/layout specification.
- `build_report.py`: portable ReportLab rebuild, writing to `rebuilt/` by default.
- `figures/`: four SVG/PDF/PNG figures, five aggregate CSVs, provenance hashes and portable `plot.py`.

Install `requirements-report.txt`, then run `python build_report.py`. Windows uses locally installed Times New Roman fonts. Other systems must supply a directory containing licensed times.ttf, timesbd.ttf and timesi.ttf using `--font-dir`. Font files are not distributed. Rebuilds include layout overflow checks; inspect all pages again after edits and keep the course's eight-page maximum. PDF creation metadata can change between rebuilds. Rebuilding does not regenerate the Word file; edit the supplied DOCX directly and check its pagination in Word.

The HTML uses local relative figure links. Browser contenteditable changes are temporary unless saved; Markdown, Word or report_content.json are safer sources for persistent editing. The original final files remain unchanged when using the default rebuild output directory.

No competition images, labels, per-image predictions, account screenshots, model weights or private filesystem paths are included. All official score identities and limitations remain as stated in the report: public maximum89.4%, selected-final private86.3%, and visible unselected private maximum87.3%.
