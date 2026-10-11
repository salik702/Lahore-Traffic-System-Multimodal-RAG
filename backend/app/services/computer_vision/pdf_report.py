"""Compact selected-vehicle report from server-owned evidence and fine records."""
import base64
from datetime import datetime,timezone,timedelta
from functools import lru_cache
from html import escape
from io import BytesIO
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
from .foreground import report_scope

BRAND=colors.HexColor('#c2410c')
INK=colors.HexColor('#292524')
MUTED=colors.HexColor('#57534e')
PALE=colors.HexColor('#f5f0e9')
LINE=colors.HexColor('#ded6cb')
WIDTH=A4[0]-88


@lru_cache(maxsize=1)
def fonts():
    import reportlab
    folder=Path(reportlab.__file__).parent/'fonts'
    pdfmetrics.registerFont(TTFont('LTI',str(folder/'Vera.ttf')))
    pdfmetrics.registerFont(TTFont('LTI-Bold',str(folder/'VeraBd.ttf')))
    pdfmetrics.registerFontFamily('LTI',normal='LTI',bold='LTI-Bold',italic='LTI',boldItalic='LTI-Bold')


def local_analysis_time(value):
    """Display the actual analysis instant in Pakistan time, never PDF-build time."""
    try:
        instant=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        if instant.tzinfo is None:return 'Unavailable'
        return instant.astimezone(timezone(timedelta(hours=5))).strftime('%d %b %Y, %I:%M %p PKT')
    except (ValueError,TypeError,OverflowError):return 'Unavailable'


def violation_detail(finding):
    if finding['type']=='helmet_violation':
        count=len(set(finding.get('rider_ids',[]))) or 1
        return ('The associated motorcycle rider has a visible bare head.' if count==1 else
                f'{count} associated motorcycle riders have visible bare heads.')
    return ' '.join(finding.get('visual_evidence',[]))


def build_report(result,vehicle_id=None):
    result=report_scope(result,vehicle_id)
    fonts()
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyLTI',fontName='LTI',fontSize=9,leading=13,textColor=INK,spaceAfter=6,wordWrap='CJK'))
    styles.add(ParagraphStyle(name='SmallLTI',parent=styles['BodyLTI'],fontSize=7.5,leading=10,textColor=MUTED))
    styles.add(ParagraphStyle(name='TitleLTI',parent=styles['BodyLTI'],fontName='LTI-Bold',fontSize=17,leading=22,textColor=BRAND,spaceAfter=10))
    styles.add(ParagraphStyle(name='HeadingLTI',parent=styles['BodyLTI'],fontName='LTI-Bold',fontSize=11,leading=15,spaceBefore=9,spaceAfter=6,keepWithNext=True))
    styles.add(ParagraphStyle(name='CellLTI',parent=styles['BodyLTI'],fontSize=8,leading=12,spaceAfter=0))

    def p(text,style='BodyLTI'):
        # OCR and evidence strings must never become executable PDF markup.
        return Paragraph(escape(str(text)).replace('\n','<br/>'),styles[style])

    def picture(url,height=255):
        data=BytesIO(base64.b64decode(url.split(',',1)[1],validate=True))
        im=Image(data)
        scale=min(WIDTH/im.imageWidth,height/im.imageHeight)
        im.drawWidth,im.drawHeight=im.imageWidth*scale,im.imageHeight*scale
        im.hAlign='CENTER'
        return im

    def table(rows,widths,header=True):
        t=Table([[p(c,'CellLTI') for c in row] for row in rows],colWidths=widths,
                repeatRows=1 if header else 0,hAlign='LEFT',splitInRow=1)
        commands=[('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),
                  ('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),7),
                  ('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.4,LINE)]
        if header:commands.append(('BACKGROUND',(0,0),(-1,0),PALE))
        t.setStyle(TableStyle(commands))
        return t

    selected=next(iter(result['vehicle_results']),None)
    plate=(selected.get('plate') or {}) if selected else {}
    # Use only confirmed OCR text, never partial candidates or manual guesses.
    plate_text=plate.get('text') if plate.get('status')=='recognized' else None
    vehicle_name=selected['vehicle_type'].title() if selected else 'No vehicle detected'
    story=[p('LAHORE TRAFFIC INTELLIGENCE','SmallLTI'),
           p('AI TRAFFIC VIOLATION ANALYSIS REPORT','TitleLTI'),
           table([['Date and time',local_analysis_time(result.get('analyzed_at'))],
                  ['Vehicle',vehicle_name],
                  ['Number plate',plate_text or ('Number plate not readable.' if selected else 'Unavailable')],
                  ['Jurisdiction',result['jurisdiction']]], [120,WIDTH-120],False),Spacer(1,8),
           KeepTogether([picture(result['original_image_url']),Spacer(1,5),
                         p('Original uploaded image - selected rider / vehicle, without YOLO annotations.','SmallLTI')]),
           p('Violation details','HeadingLTI')]
    findings=result['potential_violations']
    if findings:
        rows=[['Violation and observed evidence','Potential fine (PKR)']]
        for finding in findings:
            amount=f"{finding['fine_amount']:,}" if finding['fine_verified'] and finding['fine_amount'] is not None else 'Official fine amount requires verification.'
            rows.append([finding['name']+'\n'+violation_detail(finding),amount])
        story.append(table(rows,[WIDTH*.72,WIDTH*.28]))
    else:
        story.append(p('No supported potential violation was established.'))

    summary=result['summary']
    story += [p(f"Verified potential total: PKR {summary['verified_potential_fine']:,}",'HeadingLTI')]
    if summary['unverified_penalties']:
        story.append(p('Official fine amount requires verification for '+str(summary['unverified_penalties'])+' finding(s).','SmallLTI'))
    if result['sources']:
        fine_sources=list(dict.fromkeys(s['url'] for s in result['sources'] if s.get('url')))
        story.append(p('Legal / fine sources: '+'; '.join(fine_sources),'SmallLTI'))
    story.append(p(result['disclaimer'],'SmallLTI'))

    buffer=BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=A4,rightMargin=44,leftMargin=44,topMargin=40,bottomMargin=44,
                          title='AI Traffic Violation Analysis Report',author='Lahore Traffic Intelligence')

    def page(canvas,document):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(44,33,A4[0]-44,33)
        canvas.setFont('LTI',7)
        canvas.setFillColor(MUTED)
        canvas.drawString(44,21,'Lahore Traffic Intelligence | Preliminary AI assessment')
        canvas.drawRightString(A4[0]-44,21,f'Page {document.page}')
        canvas.restoreState()

    doc.build(story,onFirstPage=page,onLaterPages=page)
    return buffer.getvalue()
