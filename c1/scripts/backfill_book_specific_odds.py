#!/usr/bin/env python
from __future__ import annotations

import argparse, csv, random, re, time, urllib.request
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString

UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36'
BOOKS=['Pinnacle','bet365','Betway','10Bet']

def fetch(url:str, tries:int=5)->str:
    last=None
    for attempt in range(tries):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':UA,'Accept-Language':'en-US,en;q=0.9'})
            with urllib.request.urlopen(req,timeout=40) as r:
                return r.read().decode('utf-8','replace')
        except Exception as e:
            last=e
            if attempt+1<tries: time.sleep(min(20,2**attempt+random.random()))
    raise RuntimeError(f'{url}: {last!r}')

def direct_float(div):
    if div is None: return None
    for child in div.contents:
        if isinstance(child,NavigableString):
            t=str(child).strip().replace(',','.')
            if re.fullmatch(r'\d+(?:\.\d+)?',t):
                try:
                    x=float(t); return x if x>1 else None
                except: pass
    # fallback first numeric token before nested history text
    m=re.match(r'\s*(\d+(?:\.\d+)?)',div.get_text(' ',strip=True))
    if m:
        x=float(m.group(1)); return x if x>1 else None
    return None

def history(div):
    """Return (close_time, opening_odds, opening_time). Current odds itself is parsed separately."""
    if div is None: return ('',None,'')
    table=div.find('table')
    if table is None: return ('',None,'')
    trs=table.find_all('tr',recursive=False)
    close_time=''; open_time=''; opening=None
    if trs:
        tds=trs[0].find_all('td',recursive=False)
        if tds: close_time=tds[0].get_text(' ',strip=True)
    for i,tr in enumerate(trs):
        if 'opening odds' in tr.get_text(' ',strip=True).casefold() and i+1<len(trs):
            tds=trs[i+1].find_all('td',recursive=False)
            if len(tds)>=2:
                open_time=tds[0].get_text(' ',strip=True)
                try:
                    x=float(tds[1].get_text(' ',strip=True).replace(',','.')); opening=x if x>1 else None
                except: opening=None
            break
    return close_time,opening,open_time

def first_moneyline_book_row(soup,book):
    # Home/Away is the first odds market on TennisExplorer; therefore the first
    # matching bookmaker row is the moneyline row even when the book appears
    # later in totals/handicap tables too.
    for a in soup.find_all('a'):
        txt=a.get_text(' ',strip=True)
        if txt.casefold()==book.casefold():
            tr=a.find_parent('tr')
            if tr and tr.find('td',class_='k1') and tr.find('td',class_='k2'):
                return tr
    return None

def parse_book(soup,book):
    tr=first_moneyline_book_row(soup,book)
    if tr is None:
        return {'present':0}
    d1=tr.find('td',class_='k1').find('div',class_='odds-in') if tr.find('td',class_='k1') else None
    d2=tr.find('td',class_='k2').find('div',class_='odds-in') if tr.find('td',class_='k2') else None
    c1=direct_float(d1); c2=direct_float(d2)
    t1,o1,ot1=history(d1); t2,o2,ot2=history(d2)
    return {'present':1,'close1':c1,'close2':c2,'close_time1':t1,'close_time2':t2,
            'open1':o1,'open2':o2,'open_time1':ot1,'open_time2':ot2}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--odds',required=True)
    ap.add_argument('--year',type=int,required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--delay-min',type=float,default=.45)
    ap.add_argument('--delay-max',type=float,default=.80)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    src=list(csv.DictReader(open(args.odds,encoding='utf-8')))
    src=[r for r in src if str(r.get('date','')).startswith(str(args.year)) and r.get('match_url')]
    # unique match URL; keep source metadata from result-page row
    unique={}
    for r in src: unique.setdefault(r['match_url'],r)
    rows=list(unique.values())
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True)
    done=set()
    fields=['date','tournament','player1','player2','match_url','avg_odd1','avg_odd2']
    for b in BOOKS:
        p=b.lower().replace(' ','_')
        fields += [f'{p}_present',f'{p}_close1',f'{p}_close2',f'{p}_close_time1',f'{p}_close_time2',f'{p}_open1',f'{p}_open2',f'{p}_open_time1',f'{p}_open_time2']
    if args.resume and path.exists():
        done={r['match_url'] for r in csv.DictReader(open(path,encoding='utf-8'))}
    mode='a' if args.resume and path.exists() else 'w'
    with path.open(mode,'a' if False else None) if False else path.open(mode,newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if mode=='w': w.writeheader()
        n=0; pin=0; b365=0; both=0; errs=0
        for idx,r in enumerate(rows,1):
            if r['match_url'] in done: continue
            rec={'date':r.get('date',''),'tournament':r.get('tournament',''),'player1':r.get('player1',''),'player2':r.get('player2',''),
                 'match_url':r['match_url'],'avg_odd1':r.get('odd1',''),'avg_odd2':r.get('odd2','')}
            try:
                soup=BeautifulSoup(fetch(r['match_url']),'html.parser')
                parsed={b:parse_book(soup,b) for b in BOOKS}
                for b,d in parsed.items():
                    p=b.lower().replace(' ','_')
                    for k,v in d.items(): rec[f'{p}_{k}']=v
                pin += int(parsed['Pinnacle'].get('present',0) and parsed['Pinnacle'].get('close1') and parsed['Pinnacle'].get('close2'))
                b365 += int(parsed['bet365'].get('present',0) and parsed['bet365'].get('close1') and parsed['bet365'].get('close2'))
                both += int(parsed['Pinnacle'].get('close1') and parsed['Pinnacle'].get('close2') and parsed['bet365'].get('close1') and parsed['bet365'].get('close2'))
            except Exception as e:
                errs+=1; print('WARN',r['match_url'],repr(e),flush=True)
            w.writerow(rec); f.flush(); n+=1
            if n%250==0:
                print({'year':args.year,'done':n,'total':len(rows),'pinnacle':pin,'bet365':b365,'both':both,'errors':errs},flush=True)
            time.sleep(random.uniform(args.delay_min,args.delay_max))
    print({'year':args.year,'rows':len(rows),'new':n,'pinnacle':pin,'bet365':b365,'both':both,'errors':errs},flush=True)
if __name__=='__main__': main()
