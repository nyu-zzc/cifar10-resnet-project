"""Build the eight-page report and editable HTML/Markdown from report_content.json.
Usage: python build_report.py [--font-dir DIR] [--output-dir DIR] [--measure-only]
Defaults to rebuilt/; the delivered final report is never overwritten by default.
Requires reportlab, pypdf, Pillow. No data or model inference is performed.
"""
from pathlib import Path
import argparse, json, html, re, os
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image

HERE = Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--repository-url')
parser.add_argument('--measure-only',action='store_true')
parser.add_argument('--font-dir', type=Path, help='Directory containing licensed times.ttf, timesbd.ttf, timesi.ttf')
parser.add_argument('--output-dir', type=Path, default=HERE/'rebuilt')
args=parser.parse_args()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
obj=json.loads((HERE/'report_content.json').read_text(encoding='utf8'))
if args.repository_url:
    obj['repository_url']=args.repository_url
    (OUT/'report_content.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
FONT = args.font_dir or Path(os.environ.get('WINDIR', ''))/'Fonts'
if not all((FONT/name).is_file() for name in ('times.ttf', 'timesbd.ttf', 'timesi.ttf')):
    parser.error('Times New Roman fonts unavailable; provide --font-dir with licensed times.ttf, timesbd.ttf and timesi.ttf.')
for name, file in [('TNR','times.ttf'),('TNR-Bold','timesbd.ttf'),('TNR-Italic','timesi.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONT/file)))
pdfmetrics.registerFontFamily('TNR',normal='TNR',bold='TNR-Bold',italic='TNR-Italic',boldItalic='TNR-Bold')
W,H=612,792
M=46.8
WIDTH=W-2*M
TOP=740
BOTTOM=48
style={
 'paragraph':ParagraphStyle('body',fontName='TNR',fontSize=11,leading=14.1,spaceAfter=7.2),
 'heading':ParagraphStyle('heading',fontName='TNR-Bold',fontSize=13,leading=16,spaceBefore=6,spaceAfter=6),
 'title':ParagraphStyle('title',fontName='TNR-Bold',fontSize=22,leading=25,spaceAfter=10),
 'subtitle':ParagraphStyle('subtitle',fontName='TNR',fontSize=12,leading=15,spaceAfter=5),
 'author':ParagraphStyle('author',fontName='TNR',fontSize=11,leading=14,spaceAfter=6),
 'repository':ParagraphStyle('repository',fontName='TNR',fontSize=10,leading=12.5,spaceAfter=9),
 'caption':ParagraphStyle('caption',fontName='TNR',fontSize=9.3,leading=11.7,spaceAfter=9),
 'table':ParagraphStyle('table',fontName='TNR',fontSize=9.8,leading=11.8),
 'tablehead':ParagraphStyle('tablehead',fontName='TNR-Bold',fontSize=9.8,leading=11.8),
 'reference':ParagraphStyle('reference',fontName='TNR',fontSize=9.7,leading=11.9,spaceAfter=5),
}
def clean(s):
    return s.replace('\u2011','-').replace('–','-').replace('—','-')
def para(text,key): return Paragraph(html.escape(clean(text)),style[key])
def make_table(b):
    data=[[para(x,'tablehead') for x in b['headers']]]+[[para(x,'table') for x in r] for r in b['rows']]
    scales=b['widths']; total=sum(scales)
    t=Table(data,colWidths=[WIDTH*x/total for x in scales],hAlign='LEFT')
    t.setStyle(TableStyle([
      ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eef2')),
      ('LINEABOVE',(0,0),(-1,0),0.7,colors.HexColor('#263848')),
      ('LINEBELOW',(0,0),(-1,0),0.5,colors.HexColor('#667788')),
      ('LINEBELOW',(0,-1),(-1,-1),0.7,colors.HexColor('#263848')),
      ('LINEBELOW',(0,1),(-1,-2),0.25,colors.HexColor('#d5dadd')),
      ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
      ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    return t

pdf=OUT/'Zicheng_Zhou_Deep_Learning_Project_1_Report.pdf'
c=Canvas(str(pdf if not args.measure_only else OUT/'layout_measure_draft.pdf'),pagesize=(W,H),pageCompression=1)
c.setTitle(obj['title']);c.setAuthor('Zicheng Zhou (zz11018)')
c.setSubject('NYU Tandon Deep Learning F26 - CIFAR-10 residual networks')
measure=[]
missing=[]
htmlpages=[];mdpages=[]
FIGURE_RELATIVE = os.path.relpath(HERE/'figures', OUT).replace(os.sep, '/')
for n,blocks in enumerate(obj['pages'],1):
    y=TOP
    c.setFillColor(colors.HexColor('#53616c'));c.setFont('TNR',8.5)
    c.drawString(M,763,'DEEP LEARNING PROJECT 1  •  FALL 2026')
    c.drawRightString(W-M,763,'Zicheng Zhou  |  zz11018')
    c.setStrokeColor(colors.HexColor('#bac4cb'));c.setLineWidth(.4);c.line(M,755,W-M,755)
    c.setFillColor(colors.black)
    hp=[];mp=[]
    def put(flow,key='paragraph',gap_after=None):
        global y
        before=style[key].spaceBefore
        y-=before
        _,height=flow.wrapOn(c,WIDTH,2000)
        flow.drawOn(c,M,y-height)
        y-=height+(style[key].spaceAfter if gap_after is None else gap_after)
    for b in blocks:
        typ=b['type']
        if typ in ('paragraph','heading','title','subtitle','author'):
            text=b['text'];put(para(text,typ),typ)
            tag={'heading':'h2','title':'h1','paragraph':'p','subtitle':'p','author':'p'}[typ]
            hp.append(f'<{tag} class="{typ}">{html.escape(clean(text))}</{tag}>')
            prefix='# ' if typ=='title' else '## ' if typ=='heading' else ''
            mp.append(prefix+text)
        elif typ=='repository':
            url=obj.get('repository_url')
            txt='Public code repository: '+(url or '[Pending repository creation - draft only]')
            f=Paragraph('Public code repository: '+(f'<link href="{html.escape(url)}" color="#15527c">{html.escape(url)}</link>' if url else '[Pending repository creation - draft only]'),style['repository'])
            put(f,'repository')
            hp.append('<p class="repository">'+html.escape(txt)+'</p>');mp.append(txt)
        elif typ=='table':
            put(para(b['title'],'caption'),'caption',5)
            t=make_table(b);put(t,'table',9)
            hp.append('<p class="caption">'+html.escape(b['title'])+'</p><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in b['headers'])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(clean(x))+'</td>' for x in r)+'</tr>' for r in b['rows'])+'</tbody></table>')
            mp.append('**'+b['title']+'**\n\n'+'| '+' | '.join(b['headers'])+' |\n| '+' | '.join(['---']*len(b['headers']))+' |\n'+'\n'.join('| '+' | '.join(r)+' |' for r in b['rows']))
        elif typ=='figure':
            fp=HERE/'figures'/(b['name']+'.png'); height=b['height_inches']*72
            if fp.exists():
                im=Image.open(fp);iw,ih=im.size
                width=min(WIDTH,height*iw/ih); actualh=width*ih/iw
                c.drawImage(str(fp),M+(WIDTH-width)/2,y-actualh,width,actualh,mask='auto')
                y-=actualh+5
            else:
                missing.append(str(fp));c.setStrokeColor(colors.lightgrey);c.rect(M,y-height,WIDTH,height);y-=height+5
            put(para(b['caption'],'caption'),'caption')
            hp.append(f'<figure><img src="{FIGURE_RELATIVE}/{b["name"]}.svg" alt="{html.escape(b["caption"])}"><figcaption>{html.escape(clean(b["caption"]))}</figcaption></figure>')
            mp.append(f'![{b["caption"]}]({FIGURE_RELATIVE}/{b["name"]}.png)\n\n'+b['caption'])
        elif typ=='references':
            for i,(citation,url) in enumerate(b['items'],1):
                # The title remains human-readable and the compact arXiv link is clickable.
                label=url.rsplit('/',1)[-1]
                txt=f'[{i}] {html.escape(clean(citation))} <link href="{url}" color="#15527c">arXiv:{label}</link>.'
                put(Paragraph(txt,style['reference']),'reference')
                hp.append(f'<p class="reference">[{i}] {html.escape(clean(citation))} <a href="{url}">arXiv:{label}</a>.</p>')
                mp.append(f'[{i}] {citation} [{label}]({url}).')
    measure.append({'page':n,'bottom_y':round(y,2),'available_bottom':BOTTOM,'fits':y>=BOTTOM,'used_points':round(TOP-y,2)})
    c.setFillColor(colors.HexColor('#53616c'));c.setFont('TNR',9)
    c.drawCentredString(W/2,29,f'{n} / {len(obj["pages"])}')
    c.showPage()
    htmlpages.append('<section class="page">'+'\n'.join(hp)+'</section>')
    mdpages.append('\n\n'.join(mp))
c.save()
css='''@page{size:letter;margin:.65in} body{font:11pt/1.3 "Times New Roman",serif;color:#111;background:#f2f3f4} .page{box-sizing:border-box;width:8.5in;min-height:11in;margin:20px auto;padding:.65in;background:white} h1{font-size:23pt;line-height:1.1}h2{font-size:13pt;margin:12px 0 6px}p{margin:0 0 9px}.author,.subtitle{margin-bottom:6px}.repository{font-size:10pt}table{border-collapse:collapse;width:100%;font-size:9.8pt;margin:0 0 12px}th{text-align:left;background:#e9eef2}td,th{padding:5px;border-bottom:1px solid #ccd2d8;vertical-align:top}.caption,figcaption{font-size:9.3pt;line-height:1.25}figure{margin:8px 0 12px}figure img{width:100%;height:auto}.reference{font-size:9.7pt;margin-bottom:6px} @media print{body{background:white}.page{margin:0;padding:0;width:auto;min-height:0;break-after:page}.page:last-child{break-after:auto}}'''
(OUT/'manuscript_editable.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>'+html.escape(obj['title'])+'</title><style>'+css+'</style></head><body contenteditable="true">'+'\n'.join(htmlpages)+'</body></html>',encoding='utf8')
(OUT/'manuscript_editable.md').write_text('\n\n<!-- PAGE BREAK -->\n\n'.join(mdpages),encoding='utf8')
(OUT/'layout_measurements.json').write_text(json.dumps({'pages':measure,'missing_figures':missing,'has_repository':bool(obj.get('repository_url'))},indent=2),encoding='utf8')
print(json.dumps({'pages':measure,'missing_figure_count':len(missing),'has_repository':bool(obj.get('repository_url'))}))
if any(not p['fits'] for p in measure): raise SystemExit('Page overflow: revise content before delivery')
if missing and not args.measure_only: raise SystemExit('Figures missing; draft only')
