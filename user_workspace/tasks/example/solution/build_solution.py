"""Create a reviewable example, not the unique correct legal answer."""
import sys
from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml.ns import qn

SECTIONS = {'案件与金额梳理': ['建议以解除合同、返还剩余货款为主要方案。双方没有约定首笔款项具有定金担保性质，不宜主张双倍返还。工作时点为2020年6月1日，尚未起诉。',
             '据客户情况说明及沟通摘要，2019年10月28日双方商谈购买一枚劳力士迪通拿手表，总价195,000元。2019年11月1日施某通过微信支付5,000元，备注“迪通拿订金”；11月2日银行转账190,000元。卖家一直未交货，后承诺2019年12月底不能交付则退款。2020年1月1日经催讨退回10,000元，此后至工作时点无其他付款退款。',
             '按付款退款明细：5,000＋190,000＝195,000元；195,000－10,000＝185,000元。尚未返还本金为185,000元，首笔5,000元已在其中，不能再次叠加。',
             '本报告将三份材料作为完整案情，依据所载事实提出诉前方案。'],
 '诉讼请求与风险分析': ['解除及返还：客户已支付全款，卖家长期未交货，约定的年底退款条件发生后仅部分退款，可主张违约导致合同目的不能实现，并请求解除、返还185,000元。可依据当时适用的合同法第60、94、97、107条分析；本稿以诉请解除合同为方案，不主张已有解除通知生效的事实。',
               '定金：应结合双方约定判断，不能仅凭“订金”字形判断。沟通材料明确双方没有约定担保性质或不交货双倍返还，因此首笔5,000元按预付款处理，不建议主张双倍返还。合同法第115条及当时关于定金的规则可供分析，不将民法典或2023年解释作为2020年已生效依据。',
               '185,000元已包含首笔5,000元，不得重复计入。也不能仅因不交货就推定诈骗或必然承担惩罚性赔偿。',
               '利息：建议请求以185,000元为基数，自实际起诉之日起至实际清偿日止，按同期贷款市场报价利率计算资金占用损失；具体利率期限口径、起算依据及法院是否支持须核验。实际起诉日未知，不能填入虚构日期。分段计算可采用本金×各期间年利率×期间天数÷年计息天数，核实计息口径与分段规则后求和；当前不编造最终金额。客户也可在充分说明后选择更保守方案。',
               '诉讼费可请求被告承担，最终由法院依法确定。受诉法院采用通用文书占位符，不凭空填入具体法院。'],
 '民事起诉状草稿': ['原告：施某。被告：白某。案由：买卖合同纠纷。',
             '诉讼请求：一、解除原被告关于一枚劳力士迪通拿手表的买卖合同；二、判令被告返还货款185,000元；三、判令被告支付资金占用利息，以185,000元为基数，自实际起诉之日［日期待填］至实际清偿之日止，按同期贷款市场报价利率计算（具体口径提交前核验）；四、本案诉讼费用由被告承担。',
             '事实与理由：2019年10月28日，原被告通过微信商谈购表，约定价款195,000元。原告于2019年11月1日微信支付5,000元，次日银行转账190,000元。被告未交货，后承诺如2019年12月底不能交付即退款。经原告催讨，被告于2020年1月1日退回10,000元，此后至2020年6月1日仍未交货或继续退款，尚有185,000元未返还。',
             '原告认为，被告未履行交货义务，且未履行剩余退款义务，已导致购买目的不能实现。依据当时适用的合同法关于合同履行、解除、解除后返还及违约责任的规定，提出上述请求。',
             '此致 ［人民法院］。具状人：施某。日期：［提交日期］。附件：材料目录。'],
 '材料目录与证明目的': ['一、客户情况与委托说明。支持交易标的、总价、未交货和部分退款的事实，明确客户结束交易、追回款项的委托目标，对应合同解除与返还请求。',
               '二、付款退款明细。记录2019年11月1日付款5,000元、11月2日付款190,000元及2020年1月1日退款10,000元，支持已付总款195,000元及余额185,000元的计算。',
               '三、沟通情况整理。支持买卖约定、首笔款项备注、双方未约定定金担保性质，以及年底不能交付即退款的承诺。用于分析合同履行、解除返还及不适用双倍返还的理由。']}


def build(path):
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2)
    section.left_margin = section.right_margin = Cm(2.3)
    for name in ('Normal', 'Title', 'Heading 1'):
        style = doc.styles[name]
        style.font.name = 'Songti SC'
        style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Songti SC')
        style.font.color.rgb = RGBColor(0, 0, 0)
    for element in list(doc.styles.element.iter(qn('w:pBdr'))):
        element.getparent().remove(element)
    doc.styles['Normal'].font.size = Pt(11)
    doc.styles['Normal'].paragraph_format.space_after = Pt(8)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.25
    for i, (heading, paragraphs) in enumerate(SECTIONS.items()):
        if i:
            doc.add_page_break()
        else:
            doc.add_paragraph('微信购表纠纷起诉准备', 'Title')
        doc.add_heading(heading, level=1)
        for paragraph in paragraphs:
            doc.add_paragraph(paragraph)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)

if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else '/work/complaint_preparation.docx')
