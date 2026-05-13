import cairosvg, os, re

svg_dir = r'C:\Users\user\LAU\LAU Applied AI Final Project\course_deliverables\latex\figures'
svgs = ['validation-curves.svg', 'class-distribution.svg', 'prompt-anatomy.svg', 'app-workflow.svg']

for f in svgs:
    svg_path = os.path.join(svg_dir, f)
    png_path = svg_path.replace('.svg', '.png')

    with open(svg_path, 'r', encoding='utf-8') as fh:
        content = fh.read()

    vb = re.search(r'viewBox=["\'](\d+)\s+(\d+)\s+(\d+)\s+(\d+)', content)
    if vb:
        vw, vh = int(vb.group(3)), int(vb.group(4))
        content2 = re.sub(r'width=["\'][^"\']+["\']', 'width="%d"' % vw, content, count=1)
        content2 = re.sub(r'height=["\'][^"\']+["\']', 'height="%d"' % vh, content2, count=1)

        cairosvg.svg2png(bytestring=content2.encode('utf-8'), write_to=png_path, scale=4)
        sz = os.path.getsize(png_path)
        print('%s -> %d KB  (viewBox %dx%d, rendered at %dx%d)' % (f, sz // 1024, vw, vh, vw * 4, vh * 4))
    else:
        print('%s -> NO VIEWBOX FOUND' % f)
