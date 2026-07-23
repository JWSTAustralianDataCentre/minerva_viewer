import re, html
t = open('template.html').read()
m = re.search(r'<script type="text/x-dc" data-dc-script=""[^>]*>(.*?)</script>', t, re.S)
js = m.group(1)
open('app.js','w').write(js)
print("app.js len", len(js), "lines", js.count(chr(10)))
