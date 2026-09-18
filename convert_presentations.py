import os
import markdown
from weasyprint import HTML, CSS

pres_dir = 'presentations'
files = [f for f in os.listdir(pres_dir) if f.endswith('.md')]

# Minimal CSS to make HTML and PDF look decent
css_path = os.path.join(pres_dir, 'style.css')
with open(css_path, 'w') as f:
    f.write("""
body { font-family: 'Helvetica Neue', Arial, sans-serif; margin: 40px auto; max-width: 900px; line-height: 1.6; color: #333; padding: 20px; }
h1, h2, h3 { color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }
code { background: #f8f9fa; padding: 2px 5px; border-radius: 4px; font-family: monospace; color: #e83e8c; }
pre { background: #f8f9fa; padding: 15px; border-radius: 8px; overflow-x: auto; border: 1px solid #e9ecef; }
hr { border: 0; height: 1px; background: #ddd; margin: 40px 0; }
    """)

for f in files:
    name = f.replace('.md', '')
    md_path = os.path.join(pres_dir, f)
    pdf_path = os.path.join(pres_dir, f"{name}.pdf")
    html_path = os.path.join(pres_dir, f"{name}.html")
    
    print(f"Processing {f}...")
    
    with open(md_path, 'r', encoding='utf-8') as file:
        md_text = file.read()
    
    # 1. HTML
    html_text = markdown.markdown(md_text, extensions=['extra'])
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{name}</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
{html_text}
</body>
</html>
"""
    with open(html_path, 'w', encoding='utf-8') as html_file:
        html_file.write(full_html)
        
    # 2. PDF
    try:
        HTML(string=full_html).write_pdf(pdf_path)
    except Exception as e:
        print(f"Failed to generate PDF for {f}: {e}")

print("Conversion complete!")
