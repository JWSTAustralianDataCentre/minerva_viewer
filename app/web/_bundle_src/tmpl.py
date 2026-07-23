import json
src = open('/otdata2/themiya/minerva/minerva_viewer/handoff/index.html').read().split('\n')
tmpl = json.loads(src[381])  # line 382
open('template.html','w').write(tmpl)
print("template len", len(tmpl))
print(tmpl[:2500])
