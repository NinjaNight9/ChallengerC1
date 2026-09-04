#!/usr/bin/env python
from __future__ import annotations
import argparse, csv, json, random, re, time, urllib.request
from pathlib import Path
from bs4 import BeautifulSoup

UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36'

def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':UA,'Accept-Language':'en-US,en;q=0.9'})
    with urllib.request.urlopen(req,timeout=40) as r:
        return r.read().decode('utf-8','replace')

def compact(s):
    return re.sub(r'\s+',' ',s).strip()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--odds',required=True); ap.add_argument('--out',required=True); ap.add_argument('--n',type=int,default=8)
    args=ap.parse_args()
    rows=list(csv.DictReader(open(args.odds,encoding='utf-8')))
    urls=[]
    for r in rows:
        u=r.get('match_url','')
        if u and u not in urls: urls.append(u)
    # deterministic spread through file rather than first N only
    if len(urls)>args.n:
        step=max(1,len(urls)//args.n); urls=urls[::step][:args.n]
    out=[]
    for i,u in enumerate(urls):
        try:
            html=fetch(u); soup=BeautifulSoup(html,'html.parser')
            page={'url':u,'title':compact(soup.title.get_text(' ',strip=True)) if soup.title else '', 'books':{}, 'contexts':{}}
            for book in ['Pinnacle','bet365','10Bet','Betway']:
                links=[a for a in soup.find_all('a') if book.casefold() in a.get_text(' ',strip=True).casefold()]
                page['books'][book]=len(links)
                if links:
                    node=links[0]
                    tr=node.find_parent('tr')
                    page['contexts'][book]=compact(tr.get_text(' ',strip=True)) if tr else compact(node.parent.get_text(' ',strip=True))
                    # include HTML of containing row for parser design
                    page['contexts'][book+'_html']=str(tr)[:6000] if tr else str(node.parent)[:6000]
            out.append(page)
            print(json.dumps(page,ensure_ascii=False)[:12000],flush=True)
        except Exception as e:
            out.append({'url':u,'error':repr(e)}); print('ERR',u,repr(e),flush=True)
        time.sleep(.6+random.random()*.5)
    Path(args.out).write_text(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
