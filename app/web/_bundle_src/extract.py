import re, json, gzip, base64
src = open('/otdata2/themiya/minerva/minerva_viewer/handoff/index.html').read().split('\n')
manifest = json.loads(src[369])  # line 370 (0-indexed 369)
ext = json.loads(src[373])
print("=== ext_resources (external deps) ===")
for e in ext:
    print(e['uuid'], e['id'])
print("\n=== manifest entries ===")
for uuid, entry in manifest.items():
    data = entry['data']
    raw = base64.b64decode(data)
    if entry.get('compressed'):
        try:
            dec = gzip.decompress(raw)
        except Exception as ex:
            dec = raw
    else:
        dec = raw
    fn = uuid + '.out'
    open(fn,'wb').write(dec)
    print(f"{uuid}  mime={entry['mime']}  compressed={entry.get('compressed')}  b64len={len(data)}  declen={len(dec)}  -> {fn}")
